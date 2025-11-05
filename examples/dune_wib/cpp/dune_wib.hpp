/* Copyright Me */
#include <cuda_runtime.h>                      // blockIdx, blockDim, threadIdx
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
#include <stdexcept>   // for std::runtime_error
