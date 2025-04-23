#include <cstdio>
#include <iostream>

#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "executor_interface.hpp"
#include "simulator_interface.hpp"

namespace memristor_runtime {

inline constexpr std::string_view socket_path =
    "/tmp/systemc-rram-simulator.sock";

template <typename T> bool gemm_simulator_available() {
  if constexpr (!std::is_integral_v<T>)
    return false;

  return access(socket_path.data(), F_OK) != -1;
}

template <typename T> void gemm_simulator(int32_t crossbar_id) {
  if constexpr (!std::is_integral_v<T>) {
    std::cerr << "Error: floating point operations are not supported on "
                 "simulator executor"
              << std::endl;
    std::exit(1);
  }

  auto &crossbar = crossbars[crossbar_id];
  matrix_descriptor lhs_desc{
      .bits = sizeof(T) * 8,
      .rows = static_cast<uint16_t>(crossbar.lhs.sizes[0]),
      .cols = static_cast<uint16_t>(crossbar.lhs.sizes[1])};
  matrix_descriptor rhs_desc{
      .bits = sizeof(T) * 8,
      .rows = static_cast<uint16_t>(crossbar.rhs.sizes[0]),
      .cols = static_cast<uint16_t>(crossbar.rhs.sizes[1])};

  auto client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client_fd == -1) {
    std::cerr << "Error: socket() failed" << std::endl;
    std::exit(1);
  }

  struct sockaddr_un address{};
  address.sun_family = AF_UNIX;
  strncpy(address.sun_path, socket_path.data(), socket_path.size());

  if (connect(client_fd, (struct sockaddr *)&address, sizeof(address)) == -1) {
    std::cerr << "Error: connect() failed" << std::endl;
    std::exit(1);
  }

  auto checked_write = [&](void *data, size_t size) {
    if (write(client_fd, data, size) == -1) {
      std::cerr << "Error: write() failed" << std::endl;
      std::exit(1);
    }
  };

  auto checked_read = [&](void *data, size_t size) {
    if (read(client_fd, data, size) == -1) {
      std::cerr << "Error: read() failed" << std::endl;
      std::exit(1);
    }
  };

  checked_write(&lhs_desc, sizeof(lhs_desc));
  checked_write(crossbar.lhs.data, lhs_desc.byte_count());
  checked_write(&rhs_desc, sizeof(rhs_desc));
  checked_write(crossbar.rhs.data, rhs_desc.byte_count());

  matrix_descriptor result_desc{};
  checked_read(&result_desc, sizeof(result_desc));

  if (result_desc.bits != sizeof(T) * 8) {
    std::cerr << "Error: result_desc.bits != sizeof(T) * 8 ("
              << result_desc.bits << " != " << sizeof(T) * 8 << ")"
              << std::endl;
    std::exit(1);
  }

  if (result_desc.rows != crossbar.result.sizes[0]) {
    std::cerr << "Error: result_desc.rows != crossbar.result.sizes[0] ("
              << result_desc.rows << " != " << crossbar.result.sizes[0] << ")"
              << std::endl;
    std::exit(1);
  }

  if (result_desc.cols != crossbar.result.sizes[1]) {
    std::cerr << "Error: result_desc.cols != crossbar.result.sizes[1] ("
              << result_desc.cols << " != " << crossbar.result.sizes[1] << ")"
              << std::endl;
    std::exit(1);
  }

  checked_read(crossbar.result.data, result_desc.byte_count());

  close(client_fd);
}

} // namespace memristor_runtime