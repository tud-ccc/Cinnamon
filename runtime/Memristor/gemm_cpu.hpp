#include <cstdio>
#include <iostream>

#include "executor_interface.hpp"

namespace memristor_runtime {

template <typename T> bool gemm_cpu_available() { return true; }

template <typename T> void gemm_cpu(int32_t crossbar_id) {
  auto &crossbar = crossbars[crossbar_id];
  auto lhs_rows = crossbar.lhs.sizes[0];
  auto lhs_cols = crossbar.lhs.sizes[1];
  auto rhs_cols = crossbar.rhs.sizes[1];
  auto *lhs = (T *)crossbar.lhs.data;
  auto *rhs = (T *)crossbar.rhs.data;
  auto *result = (T *)crossbar.result.data;

  for (int64_t lhs_count = 0; lhs_count < lhs_rows; ++lhs_count) {
    for (int64_t rhs_count = 0; rhs_count < rhs_cols; ++rhs_count) {
      int64_t result_index = lhs_count * rhs_cols + rhs_count;
      result[result_index] = 0;
      for (int64_t shared = 0; shared < lhs_cols; ++shared) {
        int64_t lhs_index = lhs_count * lhs_cols + shared;
        int64_t rhs_index = shared * rhs_cols + rhs_count;
        result[result_index] += lhs[lhs_index] * rhs[rhs_index];
      }
    }
  }
}

} // namespace memristor_runtime