#include <cstdio>
#include <iostream>

#include "executor_interface.hpp"
#include "gemm_cpu.hpp"
#include "gemm_simulator.hpp"

namespace memristor_runtime {

crossbar crossbars[16]{};

template <typename T> static void dispatch_gemm(int32_t crossbar_id) {
  if (gemm_simulator_available<T>()) {
    std::cout << "Using simulator to perform gemm" << std::endl;
    gemm_simulator<T>(crossbar_id);
    return;
  }
  if (gemm_cpu_available<T>()) {
    std::cout << "Using cpu to perform gemm" << std::endl;
    gemm_cpu<T>(crossbar_id);
    return;
  }

  std::cerr << "No available gemm implementation" << std::endl;
  std::exit(1);
}

void memristor_barrier(int32_t) {}

template <typename T>
void memristor_write_to_crossbar(int32_t crossbar_id,
                                 memref_descriptor<2> rhs) {
  crossbars[crossbar_id].rhs = rhs;
  // rhs.print<T>();
}

template <typename T>
void memristor_gemm(int32_t crossbar_id, memref_descriptor<2> lhs,
                    memref_descriptor<2> result) {
  crossbars[crossbar_id].lhs = lhs;
  crossbars[crossbar_id].result = result;
  // lhs.print<T>();
  dispatch_gemm<T>(crossbar_id);
  // result.print<T>();
}

template <typename T>
void memristor_gemv(int32_t crossbar_id, memref_descriptor<1> lhs,
                    memref_descriptor<1> result) {
  crossbars[crossbar_id].lhs = lhs;
  crossbars[crossbar_id].result = result;
  // lhs.print<T>();
  dispatch_gemm<T>(crossbar_id);
  // result.print<T>();
}

#define INSTANTIATE_FOR_TYPE(type)                                             \
  template void memristor_write_to_crossbar<type>(int32_t,                     \
                                                  memref_descriptor<2>);       \
  template void memristor_gemm<type>(int32_t, memref_descriptor<2>,            \
                                     memref_descriptor<2>);                    \
  template void memristor_gemv<type>(int32_t, memref_descriptor<1>,            \
                                     memref_descriptor<1>);

INSTANTIATE_FOR_TYPE(uint8_t)
INSTANTIATE_FOR_TYPE(uint16_t)
INSTANTIATE_FOR_TYPE(uint32_t)
INSTANTIATE_FOR_TYPE(uint64_t)
INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(double)
} // namespace memristor_runtime