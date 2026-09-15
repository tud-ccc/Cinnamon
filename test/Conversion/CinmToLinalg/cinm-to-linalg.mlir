// RUN: cinm-opt --split-input-file --convert-cinm-ops-to-linalg %s | FileCheck %s

// CHECK-LABEL: @elementwise_add
// CHECK-SAME: (%[[A:.*]]: tensor<4xi32>, %[[B:.*]]: tensor<4xi32>)
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: ins(%[[A]], %[[B]] :
// CHECK: arith.addi
// CHECK: linalg.yield
func.func @elementwise_add(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %r = cinm.op.elementwise add %a, %b : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// CHECK-LABEL: @elementwise_exp
// CHECK-SAME: (%[[A:.*]]: tensor<4xf32>)
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: ins(%[[A]] :
// CHECK: math.exp
// CHECK: linalg.yield
func.func @elementwise_exp(%a: tensor<4xf32>) -> tensor<4xf32> {
  %r = cinm.op.elementwise exp %a : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// ReduceOp: 1D add → scalar. Output is extracted from a 0-d tensor.
// CHECK-LABEL: @reduce_add
// CHECK-SAME: (%[[A:.*]]: tensor<1024xi32>)
// CHECK: %[[INIT:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins({{.*}}) outs(%[[INIT]]
// CHECK: %[[RED:.*]] = linalg.reduce
// CHECK-SAME: ins(%[[A]] :
// CHECK-SAME: outs(%[[FILL]]
// CHECK-SAME: dimensions = [0]
// CHECK: linalg.yield
// CHECK: %[[SCALAR:.*]] = tensor.extract %[[RED]]
// CHECK: return %[[SCALAR]] : i32
func.func @reduce_add(%a: tensor<1024xi32>) -> i32 {
  %r = cinm.op.reduce add (%a) : tensor<1024xi32> -> i32
  return %r : i32
}

// -----

// ReduceOp: 2D max along last dim → 1D. No tensor.extract needed.
// CHECK-LABEL: @reduce_max_2d
// CHECK-SAME: (%[[A:.*]]: tensor<8x128xi32>)
// CHECK: %[[FILL:.*]] = linalg.fill
// CHECK: %[[RED:.*]] = linalg.reduce
// CHECK-SAME: ins(%[[A]] :
// CHECK-SAME: outs(%[[FILL]]
// CHECK-SAME: dimensions = [1]
// CHECK: linalg.yield
// CHECK-NOT: tensor.extract
// CHECK: return %[[RED]] : tensor<8xi32>
func.func @reduce_max_2d(%a: tensor<8x128xi32>) -> tensor<8xi32> {
  %r = cinm.op.reduce maxsi (%a) : tensor<8x128xi32> -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// GemvOp: matrix-vector multiply → linalg.matvec.
// CHECK-LABEL: @gemv
// CHECK-SAME: (%[[A:.*]]: tensor<8x1024xi32>, %[[x:.*]]: tensor<1024xi32>)
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<8xi32>
// CHECK: %[[FILL:.*]] = linalg.fill ins({{.*}}) outs(%[[INIT]]
// CHECK: %[[R:.*]] = linalg.matvec {{.*}}ins(%[[A]], %[[x]] :
// CHECK-SAME: outs(%[[FILL]]
// CHECK: return %[[R]] : tensor<8xi32>
func.func @gemv(%A: tensor<8x1024xi32>, %x: tensor<1024xi32>) -> tensor<8xi32> {
  %r = cinm.op.gemv %A, %x : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
  return %r : tensor<8xi32>
}

// -----

// GemmOp: matrix-matrix multiply → linalg.matmul.
// CHECK-LABEL: @gemm
// CHECK-SAME: (%[[A:.*]]: tensor<8x1024xi32>, %[[B:.*]]: tensor<1024x128xi32>)
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<8x128xi32>
// CHECK: %[[FILL:.*]] = linalg.fill ins({{.*}}) outs(%[[INIT]]
// CHECK: %[[R:.*]] = linalg.matmul {{.*}}ins(%[[A]], %[[B]] :
// CHECK-SAME: outs(%[[FILL]]
// CHECK: return %[[R]] : tensor<8x128xi32>
func.func @gemm(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {
  %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

// -----

// GemmOp with bias: bias is used directly as the outs init, no fill needed.
// CHECK-LABEL: @gemm_bias
// CHECK-SAME: (%[[A:.*]]: tensor<8x1024xi32>, %[[B:.*]]: tensor<1024x128xi32>, %[[bias:.*]]: tensor<8x128xi32>)
// CHECK-NOT: linalg.fill
// CHECK: %[[R:.*]] = linalg.matmul {{.*}}ins(%[[A]], %[[B]] :
// CHECK-SAME: outs(%[[bias]]
// CHECK: return %[[R]] : tensor<8x128xi32>
func.func @gemm_bias(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>, %bias: tensor<8x128xi32>) -> tensor<8x128xi32> {
  %r = cinm.op.gemm %A, %B plus %bias : tensor<8x1024xi32>, tensor<1024x128xi32> plus tensor<8x128xi32> -> tensor<8x128xi32>
  return %r : tensor<8x128xi32>
}

// -----

// BatchGemmOp: batched matrix-matrix multiply → linalg.batch_matmul.
// CHECK-LABEL: @batch_gemm
// CHECK-SAME: (%[[A:.*]]: tensor<2x8x1024xi32>, %[[B:.*]]: tensor<2x1024x128xi32>)
// CHECK: %[[FILL:.*]] = linalg.fill
// CHECK: %[[R:.*]] = linalg.batch_matmul {{.*}}ins(%[[A]], %[[B]] :
// CHECK-SAME: outs(%[[FILL]]
// CHECK: return %[[R]] : tensor<2x8x128xi32>
func.func @batch_gemm(%A: tensor<2x8x1024xi32>, %B: tensor<2x1024x128xi32>) -> tensor<2x8x128xi32> {
  %r = cinm.op.batch_gemm %A, %B : tensor<2x8x1024xi32>, tensor<2x1024x128xi32> -> tensor<2x8x128xi32>
  return %r : tensor<2x8x128xi32>
}

// -----

// BatchGemvOp: (B×M×K) * (B×K) → (B×M) via linalg.generic with a reduction dim.
// CHECK-LABEL: @batch_gemv
// CHECK-SAME: (%[[A:.*]]: tensor<2x8x1024xi32>, %[[x:.*]]: tensor<2x1024xi32>)
// CHECK: %[[FILL:.*]] = linalg.fill
// CHECK: %[[R:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[A]], %[[x]] :
// CHECK-SAME: outs(%[[FILL]]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: linalg.yield
// CHECK: return %[[R]] : tensor<2x8xi32>
func.func @batch_gemv(%A: tensor<2x8x1024xi32>, %x: tensor<2x1024xi32>) -> tensor<2x8xi32> {
  %r = cinm.op.batch_gemv %A, %x : tensor<2x8x1024xi32>, tensor<2x1024xi32> -> tensor<2x8xi32>
  return %r : tensor<2x8xi32>
}

// -----

// TransposeOp: permute dimensions → linalg.transpose.
// CHECK-LABEL: @transpose
// CHECK-SAME: (%[[A:.*]]: tensor<8x4xi32>)
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x8xi32>
// CHECK: %[[R:.*]] = linalg.transpose {{.*}}ins(%[[A]] :
// CHECK-SAME: outs(%[[INIT]]
// CHECK-SAME: permutation = [1, 0]
// CHECK: return %[[R]] : tensor<4x8xi32>
func.func @transpose(%A: tensor<8x4xi32>) -> tensor<4x8xi32> {
  %r = "cinm.op.transpose"(%A) {permutation = array<i64: 1, 0>} : (tensor<8x4xi32>) -> tensor<4x8xi32>
  return %r : tensor<4x8xi32>
}
