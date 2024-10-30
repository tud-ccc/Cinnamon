#include <cstdio>
#include <iostream>

#include "executor_interface.hpp"

namespace memristor_runtime {

  struct crossbar {
    memref_descriptor<2> rhs;
    memref_descriptor<2> lhs;
    memref_descriptor<2> result;
  };

  static crossbar crossbars[16]{};

  template<typename T>
  static void perform_gemm(int32_t crossbar_id) {
    auto& crossbar = crossbars[crossbar_id];
    auto* lhs = (T*)crossbar.lhs.data;
    auto* rhs = (T*)crossbar.rhs.data;
    auto* result = (T*)crossbar.result.data;

    for (int64_t lhs_count = 0; lhs_count < crossbar.lhs.sizes[1]; ++lhs_count) {
      for (int64_t rhs_count = 0; rhs_count < crossbar.rhs.sizes[0]; ++rhs_count) {
        int64_t result_index = rhs_count * crossbar.result.strides[0] + lhs_count * crossbar.result.strides[1];
        result[result_index] = 0;
        for (int64_t shared = 0; shared < crossbar.lhs.sizes[0]; ++shared) {
          int64_t lhs_index = shared * crossbar.lhs.strides[0] + lhs_count * crossbar.lhs.strides[1];
          int64_t rhs_index = shared * crossbar.rhs.strides[1] + rhs_count * crossbar.rhs.strides[0];
          result[result_index] += lhs[lhs_index] * rhs[rhs_index];
        }
      }
    }
  }

  void memristor_barrier(int32_t) {
  }

  template<typename T>
  void memristor_write_to_crossbar(int32_t crossbar_id, memref_descriptor<2> rhs) {
    crossbars[crossbar_id].rhs = rhs;
    // rhs.print<T>();
  }

  template<typename T>
  void memristor_gemm(int32_t crossbar_id, memref_descriptor<2> lhs, memref_descriptor<2> result) {
    crossbars[crossbar_id].lhs = lhs;
    crossbars[crossbar_id].result = result;
    perform_gemm<T>(crossbar_id);
    // lhs.print<T>();
    // result.print<T>();
  }

  template<typename T>
  void memristor_gemv(int32_t crossbar_id, memref_descriptor<1> lhs, memref_descriptor<1> result) {
    crossbars[crossbar_id].lhs = lhs;
    crossbars[crossbar_id].result = result;
    perform_gemm<T>(crossbar_id);
    // lhs.print<T>();
    // result.print<T>();
  }

#define INSTANTIATE_FOR_TYPE(type)                                                         \
  template void memristor_write_to_crossbar<type>(int32_t, memref_descriptor<2>);          \
  template void memristor_gemm<type>(int32_t, memref_descriptor<2>, memref_descriptor<2>); \
  template void memristor_gemv<type>(int32_t, memref_descriptor<1>, memref_descriptor<1>);

  INSTANTIATE_FOR_TYPE(int8_t)
  INSTANTIATE_FOR_TYPE(int16_t)
  INSTANTIATE_FOR_TYPE(int32_t)
  INSTANTIATE_FOR_TYPE(int64_t)
  INSTANTIATE_FOR_TYPE(float)
  INSTANTIATE_FOR_TYPE(double)
}  // namespace memristor_runtime