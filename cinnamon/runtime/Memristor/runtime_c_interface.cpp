#include <cstdio>

#include "executor_interface.hpp"

extern "C" {

// clang-format off
#define DEF_MEMRISTOR_WRITE_TO_CROSSBAR(name, type)                                                                                                                      \
  void memristor_write_to_crossbar_##name(                                                                                                                               \
      int32_t crossbar_id,                                                                                                                                               \
      type* memref_base_ptr, type* memref_data_ptr, int64_t memref_offset, int64_t memref_size0, int64_t memref_size1, int64_t memref_stride0, int64_t memref_stride1) { \
    printf("Call to write_to_crossbar_" #name " %d %p\n", crossbar_id, memref_data_ptr);                                                                                         \
    memristor_runtime::memref_descriptor<2> rhs = {memref_base_ptr, memref_data_ptr, memref_offset, {memref_size0, memref_size1}, {memref_stride0, memref_stride1}};     \
    memristor_runtime::memristor_write_to_crossbar<type>(crossbar_id, rhs);                                                                                              \
  }

#define DEF_MEMRISTOR_GEMM(name, type)                                                                                                                                          \
  void memristor_gemm_##name(                                                                                                                                                   \
      int32_t crossbar_id,                                                                                                                                                      \
      type* memref0_base_ptr, type* memref0_data_ptr, int64_t memref0_offset, int64_t memref0_size0, int64_t memref0_size1, int64_t memref0_stride0, int64_t memref0_stride1,   \
      type* memref1_base_ptr, type* memref1_data_ptr, int64_t memref1_offset, int64_t memref1_size0, int64_t memref1_size1, int64_t memref1_stride0, int64_t memref1_stride1) { \
    printf("Call to gemm_" #name " %d %p %p\n", crossbar_id, memref0_data_ptr, memref1_data_ptr);                                                                                       \
    memristor_runtime::memref_descriptor<2> lhs = {memref0_base_ptr, memref0_data_ptr, memref0_offset, {memref0_size0, memref0_size1}, {memref0_stride0, memref0_stride1}};     \
    memristor_runtime::memref_descriptor<2> result = {memref1_base_ptr, memref1_data_ptr, memref1_offset, {memref1_size0, memref1_size1}, {memref1_stride0, memref1_stride1}};  \
    memristor_runtime::memristor_gemm<type>(crossbar_id, lhs, result);                                                                                                          \
  }

#define DEF_MEMRISTOR_GEMV(name, type)                                                                                                         \
  void memristor_gemv_##name(                                                                                                                  \
      int32_t crossbar_id,                                                                                                                     \
      type* memref0_base_ptr, type* memref0_data_ptr, int64_t memref0_offset, int64_t memref0_size0, int64_t memref0_stride0,                  \
      type* memref1_base_ptr, type* memref1_data_ptr, int64_t memref1_offset, int64_t memref1_size0, int64_t memref1_stride0) {                \
    printf("Call to gemv_" #name " %d %p %p\n", crossbar_id, memref0_data_ptr, memref1_data_ptr);                                                      \
    memristor_runtime::memref_descriptor<1> lhs = {memref0_base_ptr, memref0_data_ptr, memref0_offset, {memref0_size0}, {memref0_stride0}};    \
    memristor_runtime::memref_descriptor<1> result = {memref1_base_ptr, memref1_data_ptr, memref1_offset, {memref1_size0}, {memref1_stride0}}; \
    memristor_runtime::memristor_gemv<type>(crossbar_id, lhs, result);                                                                         \
  }
// clang-format on

void memristor_barrier(int32_t crossbar_id) {
  printf("Call to barrier %d\n", crossbar_id);
  memristor_runtime::memristor_barrier(crossbar_id);
}

DEF_MEMRISTOR_WRITE_TO_CROSSBAR(i8, uint8_t)
DEF_MEMRISTOR_WRITE_TO_CROSSBAR(i16, uint16_t)
DEF_MEMRISTOR_WRITE_TO_CROSSBAR(i32, uint32_t)
DEF_MEMRISTOR_WRITE_TO_CROSSBAR(i64, uint64_t)
DEF_MEMRISTOR_WRITE_TO_CROSSBAR(f32, float)
DEF_MEMRISTOR_WRITE_TO_CROSSBAR(f64, double)

DEF_MEMRISTOR_GEMM(i8, uint8_t)
DEF_MEMRISTOR_GEMM(i16, uint16_t)
DEF_MEMRISTOR_GEMM(i32, uint32_t)
DEF_MEMRISTOR_GEMM(i64, uint64_t)
DEF_MEMRISTOR_GEMM(f32, float)
DEF_MEMRISTOR_GEMM(f64, double)

DEF_MEMRISTOR_GEMV(i8, uint8_t)
DEF_MEMRISTOR_GEMV(i16, uint16_t)
DEF_MEMRISTOR_GEMV(i32, uint32_t)
DEF_MEMRISTOR_GEMV(i64, uint64_t)
DEF_MEMRISTOR_GEMV(f32, float)
DEF_MEMRISTOR_GEMV(f64, double)
}