# Ping Any

This example shows how an integer can be sent as an `int` type and received as an `std::any` type.
Additionally, this example shows how a `std::vector<IOSpec*>` in a `Parameter` can be used to 
receive a `vector` of `int` values.

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/dune_wib/cpp/dune_wib
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/dune_wib/cpp/dune_wib
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/dune_wib/cpp/dune_wib
  ```