// RUN: cinm-opt %s --canonicalize | FileCheck %s

// The compute-op signature rewrites (passthrough yields, unused block
// arguments) are deliberately not canonicalizations anymore; they live in
// --cinm-simplify-compute-signatures and are tested there.

// ReduceOpNormalizeDim: negative dimension attribute is canonicalized to its
// positive equivalent (dimension + rank). The -1 attribute must not appear
// in the output.

// CHECK-LABEL: func.func @reduce_normalize_dim
// CHECK-NOT:   dimension = -1
// CHECK:       cinm.op.reduce add(%{{.*}}) : tensor<4x8xi32> -> tensor<4xi32>
func.func @reduce_normalize_dim(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
  %0 = cinm.compute -> tensor<4xi32> {
    %r = cinm.op.reduce add (%arg0) dim -1 : tensor<4x8xi32> -> tensor<4xi32>
    cinm.yield %r : tensor<4xi32>
  }
  return %0 : tensor<4xi32>
}

// ReduceOpNormalizeDim: already-positive dimension should not change.

// CHECK-LABEL: func.func @reduce_positive_dim_unchanged
// CHECK:       cinm.op.reduce add(%{{.*}}) dim 0 :
func.func @reduce_positive_dim_unchanged(%arg0: tensor<4x8xi32>) -> tensor<8xi32> {
  %0 = cinm.compute -> tensor<8xi32> {
    %r = cinm.op.reduce add (%arg0) dim 0 : tensor<4x8xi32> -> tensor<8xi32>
    cinm.yield %r : tensor<8xi32>
  }
  return %0 : tensor<8xi32>
}

// GemmlikeRemoveZeroBias: integer zero-splat bias on cinm.op.gemm is dropped.

// CHECK-LABEL: func.func @gemm_remove_zero_int_bias
// CHECK-NOT:   plus
// CHECK:       cinm.op.gemm %{{.*}}, %{{.*}} : tensor<4x8xi32>, tensor<8x4xi32>
func.func @gemm_remove_zero_int_bias(%A: tensor<4x8xi32>, %B: tensor<8x4xi32>) -> tensor<4x4xi32> {
  %zero = arith.constant dense<0> : tensor<4x4xi32>
  %0 = cinm.op.gemm %A, %B plus %zero : tensor<4x8xi32>, tensor<8x4xi32> plus tensor<4x4xi32> -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// GemmlikeRemoveZeroBias: float zero-splat bias on cinm.op.gemm is dropped.

// CHECK-LABEL: func.func @gemm_remove_zero_float_bias
// CHECK-NOT:   plus
// CHECK:       cinm.op.gemm %{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x4xf32>
func.func @gemm_remove_zero_float_bias(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = cinm.op.gemm %A, %B plus %zero : tensor<4x8xf32>, tensor<8x4xf32> plus tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// GemmlikeRemoveZeroBias: non-zero bias must be preserved.

// CHECK-LABEL: func.func @gemm_keep_nonzero_bias
// CHECK:       cinm.op.gemm %{{.*}}, %{{.*}} plus %{{.*}} : tensor<4x8xi32>, tensor<8x4xi32>
func.func @gemm_keep_nonzero_bias(%A: tensor<4x8xi32>, %B: tensor<8x4xi32>) -> tensor<4x4xi32> {
  %bias = arith.constant dense<1> : tensor<4x4xi32>
  %0 = cinm.op.gemm %A, %B plus %bias : tensor<4x8xi32>, tensor<8x4xi32> plus tensor<4x4xi32> -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// GemmlikeRemoveZeroBias: zero bias on cinm.op.gemv is dropped.

// CHECK-LABEL: func.func @gemv_remove_zero_bias
// CHECK-NOT:   plus
// CHECK:       cinm.op.gemv %{{.*}}, %{{.*}} : tensor<4x8xi32>, tensor<8xi32>
func.func @gemv_remove_zero_bias(%A: tensor<4x8xi32>, %B: tensor<8xi32>) -> tensor<4xi32> {
  %zero = arith.constant dense<0> : tensor<4xi32>
  %0 = cinm.op.gemv %A, %B plus %zero : tensor<4x8xi32>, tensor<8xi32> plus tensor<4xi32> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// GemmlikeRemoveZeroBias: zero bias on cinm.op.batch_gemm is dropped.

// CHECK-LABEL: func.func @batch_gemm_remove_zero_bias
// CHECK-NOT:   plus
// CHECK:       cinm.op.batch_gemm %{{.*}}, %{{.*}} : tensor<2x4x8xi32>, tensor<2x8x4xi32>
func.func @batch_gemm_remove_zero_bias(%A: tensor<2x4x8xi32>, %B: tensor<2x8x4xi32>) -> tensor<2x4x4xi32> {
  %zero = arith.constant dense<0> : tensor<2x4x4xi32>
  %0 = cinm.op.batch_gemm %A, %B plus %zero : tensor<2x4x8xi32>, tensor<2x8x4xi32> plus tensor<2x4x4xi32> -> tensor<2x4x4xi32>
  return %0 : tensor<2x4x4xi32>
}

// GemmlikeRemoveZeroBias: zero bias on cinm.op.batch_gemv is dropped.

// CHECK-LABEL: func.func @batch_gemv_remove_zero_bias
// CHECK-NOT:   plus
// CHECK:       cinm.op.batch_gemv %{{.*}}, %{{.*}} : tensor<2x4x8xi32>, tensor<2x8xi32>
func.func @batch_gemv_remove_zero_bias(%A: tensor<2x4x8xi32>, %B: tensor<2x8xi32>) -> tensor<2x4xi32> {
  %zero = arith.constant dense<0> : tensor<2x4xi32>
  %0 = cinm.op.batch_gemv %A, %B plus %zero : tensor<2x4x8xi32>, tensor<2x8xi32> plus tensor<2x4xi32> -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// GemmlikeRemoveZeroOut: zero out buffer on cinm.op.gemm is dropped.

// CHECK-LABEL: func.func @gemm_remove_zero_out
// CHECK-NOT:   into
// CHECK:       cinm.op.gemm %{{.*}}, %{{.*}} : tensor<4x8xi32>, tensor<8x4xi32>
func.func @gemm_remove_zero_out(%A: tensor<4x8xi32>, %B: tensor<8x4xi32>) -> tensor<4x4xi32> {
  %zero = arith.constant dense<0> : tensor<4x4xi32>
  %0 = cinm.op.gemm %A, %B into %zero : tensor<4x8xi32>, tensor<8x4xi32> into tensor<4x4xi32> -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// GemmlikeRemoveZeroOut: zero out buffer on cinm.op.gemv is dropped.

// CHECK-LABEL: func.func @gemv_remove_zero_out
// CHECK-NOT:   into
// CHECK:       cinm.op.gemv %{{.*}}, %{{.*}} : tensor<4x8xi32>, tensor<8xi32>
func.func @gemv_remove_zero_out(%A: tensor<4x8xi32>, %B: tensor<8xi32>) -> tensor<4xi32> {
  %zero = arith.constant dense<0> : tensor<4xi32>
  %0 = cinm.op.gemv %A, %B into %zero : tensor<4x8xi32>, tensor<8xi32> into tensor<4xi32> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// GemmlikeRemoveZeroOut: non-zero out buffer must be preserved.

// CHECK-LABEL: func.func @gemm_keep_nonzero_out
// CHECK:       cinm.op.gemm %{{.*}}, %{{.*}} into %{{.*}} : tensor<4x8xi32>, tensor<8x4xi32>
func.func @gemm_keep_nonzero_out(%A: tensor<4x8xi32>, %B: tensor<8x4xi32>) -> tensor<4x4xi32> {
  %buf = arith.constant dense<1> : tensor<4x4xi32>
  %0 = cinm.op.gemm %A, %B into %buf : tensor<4x8xi32>, tensor<8x4xi32> into tensor<4x4xi32> -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}
