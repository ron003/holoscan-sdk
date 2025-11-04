/*********************************************************************
 * Holoscan UDP demo – updated for Holoscan SDK v3.6.1
 *
 *  • Receives a UDP packet (metadata + raw float data)
 *  • Turns the raw data into a holoscan::Tensor (GPU)
 *  • Runs a tiny CUDA kernel that multiplies each element by a factor
 *  • Packs the processed data together with the original metadata
 *  • Sends the new packet out on a different UDP port
 *
 *  No Boost/ASIO – pure POSIX sockets.
 *********************************************************************/

#include <holoscan/holoscan.hpp>          // master public header (includes everything we need)
#include <cuda_runtime.h>                 // blockIdx, blockDim, threadIdx
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>   // for the simple argument map we keep in the app

/* ------------------------------------------------------------------
 *  Helper structs / functions for the UDP protocol
 * ------------------------------------------------------------------ */
struct UdpHeader {
  uint32_t meta_len;   // bytes of the metadata block
  uint32_t data_len;   // bytes of the raw data block
};

inline ssize_t recv_all(int fd, void* buf, size_t len) {
  size_t received = 0;
  while (received < len) {
    ssize_t ret = ::recv(fd, static_cast<char*>(buf) + received,
                         len - received, 0);
    if (ret <= 0) return ret;          // error or connection closed
    received += static_cast<size_t>(ret);
  }
  return static_cast<ssize_t>(received);
}
inline ssize_t send_all(int fd, const void* buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    ssize_t ret = ::send(fd, static_cast<const char*>(buf) + sent,
                         len - sent, 0);
    if (ret <= 0) return ret;
    sent += static_cast<size_t>(ret);
  }
  return static_cast<ssize_t>(sent);
}

/* ------------------------------------------------------------------
 *  (De)serialization helpers
 * ------------------------------------------------------------------ */
bool parse_udp_packet(const std::vector<char>& packet,
                      std::vector<char>& metadata,
                      std::vector<float>& tensor_data) {
  if (packet.size() < sizeof(UdpHeader)) {
    std::cerr << "[UDP] Packet too small for header\n";
    return false;
  }
  const UdpHeader* hdr = reinterpret_cast<const UdpHeader*>(packet.data());
  uint32_t meta_len = ntohl(hdr->meta_len);
  uint32_t data_len = ntohl(hdr->data_len);
  size_t expected = sizeof(UdpHeader) + meta_len + data_len;
  if (packet.size() != expected) {
    std::cerr << "[UDP] Size mismatch (expected " << expected
              << ", got " << packet.size() << ")\n";
    return false;
  }
  // copy metadata
  metadata.resize(meta_len);
  std::memcpy(metadata.data(),
              packet.data() + sizeof(UdpHeader), meta_len);
  // copy float payload
  if (data_len % sizeof(float) != 0) {
    std::cerr << "[UDP] Data length not a multiple of float32\n";
    return false;
  }
  size_t n_floats = data_len / sizeof(float);
  tensor_data.resize(n_floats);
  std::memcpy(tensor_data.data(),
              packet.data() + sizeof(UdpHeader) + meta_len,
              data_len);
  return true;
}
std::vector<char> build_udp_packet(const std::vector<char>& metadata,
                                   const std::vector<float>& payload) {
  UdpHeader hdr;
  hdr.meta_len = htonl(static_cast<uint32_t>(metadata.size()));
  hdr.data_len = htonl(static_cast<uint32_t>(payload.size() * sizeof(float)));

  size_t total = sizeof(UdpHeader) + metadata.size()
                 + payload.size() * sizeof(float);
  std::vector<char> packet(total);
  std::memcpy(packet.data(), &hdr, sizeof(UdpHeader));
  std::memcpy(packet.data() + sizeof(UdpHeader),
              metadata.data(), metadata.size());
  std::memcpy(packet.data() + sizeof(UdpHeader) + metadata.size(),
              payload.data(), payload.size() * sizeof(float));
  return packet;
}

/* ------------------------------------------------------------------
 *  Tiny CUDA kernel – element‑wise multiplication
 * ------------------------------------------------------------------ */
__global__ void mul_by_factor_kernel(float* data,
                                     size_t N,
                                     float factor) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { data[idx] = data[idx] * factor; }
}

/* ------------------------------------------------------------------
 *  Custom Holoscan operators (v3 API)
 * ------------------------------------------------------------------ */

/* --------------------------------------------------------------
 *  UDPReceiverOp
 * -------------------------------------------------------------- */
class UDPReceiverOp : public holoscan::Operator {
 public:
  // Register the operator – the macro must be inside the class definition
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UDPReceiverOp);

  UDPReceiverOp() = default;   // default ctor – the framework injects name/fragment

  /*** Operator specification ***/
  void setup(holoscan::OperatorSpec& spec) override {
    // No external trigger – we just poll the socket.
    spec.output<std::vector<char>>("metadata")
        .description("Raw metadata block (unchanged)");
    spec.output<std::shared_ptr<holoscan::Tensor>>("tensor")
        .description("GPU tensor containing the float data");
  }

  /*** One‑time initialization ***/
  void initialize() override {
    // The port is supplied as an operator argument (see compose()).
    listen_port_ = static_cast<uint16_t>(args().value<int>("listen_port"));

    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (socket_fd_ < 0) {
      HOLOSCAN_LOG_ERROR("Failed to create UDP socket: {}", std::strerror(errno));
      throw std::runtime_error("socket creation failure");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(listen_port_);
    if (::bind(socket_fd_,
              reinterpret_cast<const sockaddr*>(&addr),
              sizeof(addr))) {
      HOLOSCAN_LOG_ERROR("UDP bind failed (port {}): {}", listen_port_,
                         std::strerror(errno));
      throw std::runtime_error("bind failure");
    }
    HOLOSCAN_LOG_INFO("UDPReceiver listening on port {}", listen_port_);
  }

  /*** Per‑tick processing ***/
  void compute(holoscan::InputContext&  op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& exec_context) override {
    // --------------------------------------------------------------
    //  Non‑blocking receive
    // --------------------------------------------------------------
    std::vector<char> recv_buf(max_packet_size_);
    ssize_t nbytes = ::recv(socket_fd_, recv_buf.data(),
                            recv_buf.size(), MSG_DONTWAIT);
    if (nbytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) { return; }   // nothing yet
      HOLOSCAN_LOG_ERROR("UDP recv error: {}", std::strerror(errno));
      return;
    }
    recv_buf.resize(static_cast<size_t>(nbytes));

    // --------------------------------------------------------------
    //  Decode the packet
    // --------------------------------------------------------------
    std::vector<char> metadata;
    std::vector<float> host_data;
    if (!parse_udp_packet(recv_buf, metadata, host_data)) {
      HOLOSCAN_LOG_WARN("Failed to parse incoming UDP packet – dropping");
      return;
    }

    // --------------------------------------------------------------
    //  Create a GPU tensor and copy the host data into it
    // --------------------------------------------------------------
    const size_t N = host_data.size();

    // TensorShape is a vector of int64_t
    holoscan::TensorShape shape{static_cast<int64_t>(N)};
    // Use the fragment’s resource‑factory to create the Tensor
    auto tensor = fragment()->make_resource<holoscan::Tensor>(
        shape,
        holoscan::PrimitiveType::kFloat32,
        /*device=*/0);

    // Async copy – stream comes from the ExecutionContext
    cudaStream_t stream = exec_context.cuda_stream();

    cudaError_t err = cudaMemcpyAsync(
        tensor->data(),
        host_data.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("cudaMemcpyAsync(H2D) failed: {}", cudaGetErrorString(err));
      return;
    }

    // --------------------------------------------------------------
    //  Emit the outputs
    // --------------------------------------------------------------
    op_output.emit<std::vector<char>>("metadata", std::move(metadata));
    op_output.emit<std::shared_ptr<holoscan::Tensor>>("tensor", std::move(tensor));
  }

 private:
  int socket_fd_{-1};
  uint16_t listen_port_{0};
  static constexpr size_t max_packet_size_ = 8 * 1024 * 1024;  // 8 MiB
};

/* --------------------------------------------------------------
 *  MulTensorOp – GPU kernel wrapper
 * -------------------------------------------------------------- */
class MulTensorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MulTensorOp);
  MulTensorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::vector<char>>("metadata")
        .description("Metadata that should be passed through unchanged");
    spec.input<std::shared_ptr<holoscan::Tensor>>("tensor")
        .description("GPU tensor to be multiplied");

    spec.output<std::vector<char>>("metadata")
        .description("Same metadata block");
    spec.output<std::shared_ptr<holoscan::Tensor>>("tensor")
        .description("Tensor after multiplication");
  }

  void compute(holoscan::InputContext&  op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& exec_context) override {
    // --------------------------------------------------------------
    //  Grab inputs
    // --------------------------------------------------------------
    const auto& meta = op_input.get<std::vector<char>>("metadata");
    auto tensor = op_input.get<std::shared_ptr<holoscan::Tensor>>("tensor");

    // --------------------------------------------------------------
    //  Launch kernel (in‑place)
    // --------------------------------------------------------------
    const size_t N = static_cast<size_t>(tensor->size());
    const float factor = args().value<float>("multiply_factor");   // args() still works

    const int threads = 256;
    const int blocks = static_cast<int>((N + threads - 1) / threads);

    float* dev_ptr = static_cast<float*>(tensor->data());
    cudaStream_t stream = exec_context.cuda_stream();

    mul_by_factor_kernel<<<blocks, threads, 0, stream>>>(dev_ptr, N, factor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("CUDA kernel launch failed: {}", cudaGetErrorString(err));
      return;
    }

    // --------------------------------------------------------------
    //  Forward downstream
    // --------------------------------------------------------------
    op_output.emit<std::vector<char>>("metadata", meta);
    op_output.emit<std::shared_ptr<holoscan::Tensor>>("tensor", std::move(tensor));
  }
};

/* --------------------------------------------------------------
 *  UDPSenderOp – copy tensor back to host and send it out again
 * -------------------------------------------------------------- */
class UDPSenderOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UDPSenderOp);
  UDPSenderOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::vector<char>>("metadata")
        .description("Metadata to embed in the outgoing packet");
    spec.input<std::shared_ptr<holoscan::Tensor>>("tensor")
        .description("GPU tensor that has been processed");
  }

  void initialize() override {
    // Destination address comes from operator arguments.
    std::string dst_ip   = args().value<std::string>("dest_ip");
    uint16_t    dst_port = static_cast<uint16_t>(args().value<int>("dest_port"));

    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      HOLOSCAN_LOG_ERROR("Failed to create UDP socket for sending: {}", std::strerror(errno));
      throw std::runtime_error("sender socket failure");
    }

    std::memset(&dest_addr_, 0, sizeof(dest_addr_));
    dest_addr_.sin_family = AF_INET;
    dest_addr_.sin_port   = htons(dst_port);
    if (::inet_pton(AF_INET, dst_ip.c_str(),
                    &dest_addr_.sin_addr) != 1) {
      HOLOSCAN_LOG_ERROR("Invalid destination IP: {}", dst_ip);
      throw std::runtime_error("invalid dest IP");
    }

    HOLOSCAN_LOG_INFO("UDPSender will send to {}:{}", dst_ip, dst_port);
  }

  void compute(holoscan::InputContext&  op_input,
               holoscan::OutputContext& /*op_output*/,   // we don’t emit anything downstream
               holoscan::ExecutionContext& /*exec_context*/) override {
    const auto& meta   = op_input.get<std::vector<char>>("metadata");
    auto tensor        = op_input.get<std::shared_ptr<holoscan::Tensor>>("tensor");

    // --------------------------------------------------------------
    //  Copy tensor back to host (synchronous – fine for a demo)
    // --------------------------------------------------------------
    const size_t N = static_cast<size_t>(tensor->size());
    std::vector<float> host(N);
    cudaError_t err = cudaMemcpy(host.data(),
                                 tensor->data(),
                                 N * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("cudaMemcpy(D2H) failed: {}", cudaGetErrorString(err));
      return;
    }

    // --------------------------------------------------------------
    //  Build a new UDP packet and send it
    // --------------------------------------------------------------
    std::vector<char> packet = build_udp_packet(meta, host);
    ssize_t sent = ::sendto(socket_fd_,
                            packet.data(),
                            packet.size(),
                            0,
                            reinterpret_cast<const sockaddr*>(&dest_addr_),
                            sizeof(dest_addr_));
    if (sent < 0) {
      HOLOSCAN_LOG_ERROR("UDP sendto failed: {}", std::strerror(errno));
    } else if (static_cast<size_t>(sent) != packet.size()) {
      HOLOSCAN_LOG_WARN("Partial UDP packet sent ({} of {} bytes)",
                        sent, packet.size());
    }
  }

 private:
  int socket_fd_{-1};
  sockaddr_in dest_addr_{};
};

/* --------------------------------------------------------------
 *  Main application – the Holoscan graph
 * -------------------------------------------------------------- */
class UdpDemoApp : public holoscan::Application {
 public:
  UdpDemoApp() = default;

  // Simple POD members that we later turn into holoscan::Arg when we
  // instantiate the operators.
  void set_source_port(uint16_t p) { listen_port_ = p; }
  void set_dest_ip(const std::string& ip) { dest_ip_ = ip; }
  void set_dest_port(uint16_t p)       { dest_port_ = p; }
  void set_factor(float f)             { multiply_factor_ = f; }

 protected:
  void compose() override {
    // ----------------------------------------------------------------
    //  Operators – we hand the arguments to each operator here.
    // ----------------------------------------------------------------
    auto udp_rx = make_operator<UDPReceiverOp>(
        "udp_rx",
        holoscan::Arg("listen_port", static_cast<int>(listen_port_)));

    auto mul = make_operator<MulTensorOp>(
        "multiply",
        holoscan::Arg("multiply_factor", multiply_factor_));

    auto udp_tx = make_operator<UDPSenderOp>(
        "udp_tx",
        holoscan::Arg("dest_ip",   dest_ip_),
        holoscan::Arg("dest_port", static_cast<int>(dest_port_)));

    // ----------------------------------------------------------------
    //  Connect the graph
    // ----------------------------------------------------------------
    add_flow(udp_rx, mul,
             {{"metadata", "metadata"},
              {"tensor",   "tensor"}});
    add_flow(mul, udp_tx,
             {{"metadata", "metadata"},
              {"tensor",   "tensor"}});
  }

 private:
  uint16_t listen_port_{0};
  std::string dest_ip_;
  uint16_t dest_port_{0};
  float multiply_factor_{1.0f};
};

/* --------------------------------------------------------------
 *  Entry point
 * -------------------------------------------------------------- */
int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <listen_port> <dest_ip> <dest_port> <multiply_factor> <gpu_id>\n";
    return 1;
  }

  uint16_t listen_port = static_cast<uint16_t>(std::stoi(argv[1]));
  std::string dest_ip   = argv[2];
  uint16_t dest_port    = static_cast<uint16_t>(std::stoi(argv[3]));
  float factor          = std::stof(argv[4]);
  int   gpu_id          = std::stoi(argv[5]);

  // Choose the CUDA device before any Holoscan objects are created
  cudaSetDevice(gpu_id);

  auto app = std::make_shared<UdpDemoApp>();
  app->set_source_port(listen_port);
  app->set_dest_ip(dest_ip);
  app->set_dest_port(dest_port);
  app->set_factor(factor);

  app->run();          // run() returns void in v3
  return 0;
}
