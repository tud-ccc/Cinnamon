// RUN: cinm-opt %s --split-input-file | FileCheck %s

// Gemm-like ops may accumulate into a wider element type than their operands
// carry, which is what an int8 checkpoint needs: a 768-deep dot product of
// int8 values does not fit in an int8. Both operands still share one type.

// CHECK-LABEL: @gemm_i8_i32
// CHECK: cinm.op.gemm %{{.*}}, %{{.*}} : tensor<4x8xi8>, tensor<8x16xi8> -> tensor<4x16xi32>
func.func @gemm_i8_i32(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
  %c = cinm.op.gemm %a, %b : tensor<4x8xi8>, tensor<8x16xi8> -> tensor<4x16xi32>
  return %c : tensor<4x16xi32>
}

// -----

// CHECK-LABEL: @gemv_i8_i32
// CHECK: cinm.op.gemv %{{.*}}, %{{.*}} : tensor<16x8xi8>, tensor<8xi8> -> tensor<16xi32>
func.func @gemv_i8_i32(%a: tensor<16x8xi8>, %x: tensor<8xi8>) -> tensor<16xi32> {
  %y = cinm.op.gemv %a, %x : tensor<16x8xi8>, tensor<8xi8> -> tensor<16xi32>
  return %y : tensor<16xi32>
}

// -----

// CHECK-LABEL: @batch_gemm_i8_i32
// CHECK: cinm.op.batch_gemm
func.func @batch_gemm_i8_i32(%a: tensor<2x4x8xi8>, %b: tensor<2x8x16xi8>) -> tensor<2x4x16xi32> {
  %c = cinm.op.batch_gemm %a, %b : tensor<2x4x8xi8>, tensor<2x8x16xi8> -> tensor<2x4x16xi32>
  return %c : tensor<2x4x16xi32>
}

// -----

// CHECK-LABEL: @batch_gemv_i8_i32
// CHECK: cinm.op.batch_gemv
func.func @batch_gemv_i8_i32(%a: tensor<2x4x8xi8>, %x: tensor<2x8xi8>) -> tensor<2x4xi32> {
  %y = cinm.op.batch_gemv %a, %x : tensor<2x4x8xi8>, tensor<2x8xi8> -> tensor<2x4xi32>
  return %y : tensor<2x4xi32>
}

// -----

// The bias is the accumulator, so it carries the wide type. This is the form
// the tiling pass emits, where `bias` is the running accumulator tile.

// CHECK-LABEL: @gemm_bias_is_accumulator
// CHECK: cinm.op.gemm %{{.*}}, %{{.*}} plus %{{.*}} : tensor<4x8xi8>, tensor<8x16xi8> plus tensor<4x16xi32> -> tensor<4x16xi32>
func.func @gemm_bias_is_accumulator(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>,
                                    %bias: tensor<4x16xi32>) -> tensor<4x16xi32> {
  %c = cinm.op.gemm %a, %b plus %bias
    : tensor<4x8xi8>, tensor<8x16xi8> plus tensor<4x16xi32> -> tensor<4x16xi32>
  return %c : tensor<4x16xi32>
}

// -----

// The memref variant accumulates into a wide out buffer.

// CHECK-LABEL: @gemv_memref_wide_out
// CHECK: cinm.op.gemv %{{.*}}, %{{.*}} into %{{.*}} : memref<16x8xi8>, memref<8xi8> into memref<16xi32>
func.func @gemv_memref_wide_out(%a: memref<16x8xi8>, %x: memref<8xi8>, %o: memref<16xi32>) {
  cinm.op.gemv %a, %x into %o : memref<16x8xi8>, memref<8xi8> into memref<16xi32>
  return
}

// -----

// Float widening is accepted on the same rule.

// CHECK-LABEL: @gemm_f16_f32
// CHECK: cinm.op.gemm %{{.*}}, %{{.*}} : tensor<4x8xf16>, tensor<8x16xf16> -> tensor<4x16xf32>
func.func @gemm_f16_f32(%a: tensor<4x8xf16>, %b: tensor<8x16xf16>) -> tensor<4x16xf32> {
  %c = cinm.op.gemm %a, %b : tensor<4x8xf16>, tensor<8x16xf16> -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}

// -----

// An equal-width result is still the common case and must keep working.

// CHECK-LABEL: @gemm_same_type
// CHECK: cinm.op.gemm %{{.*}}, %{{.*}} : tensor<4x8xi32>, tensor<8x16xi32> -> tensor<4x16xi32>
func.func @gemm_same_type(%a: tensor<4x8xi32>, %b: tensor<8x16xi32>) -> tensor<4x16xi32> {
  %c = cinm.op.gemm %a, %b : tensor<4x8xi32>, tensor<8x16xi32> -> tensor<4x16xi32>
  return %c : tensor<4x16xi32>
}
