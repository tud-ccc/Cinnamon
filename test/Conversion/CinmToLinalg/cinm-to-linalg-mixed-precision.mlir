// RUN: cinm-opt --split-input-file --convert-cinm-ops-to-linalg %s | FileCheck %s

// The named linalg contractions already widen narrow `ins` into a wider
// `outs`, so gemm/gemv/batch_gemm carry the mixed types straight through.

// CHECK-LABEL: @gemm_i8_i32
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x16xi32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%[[INIT]] : tensor<4x16xi32>)
// CHECK: linalg.matmul {{.*}}ins(%{{.*}}, %{{.*}} : tensor<4x8xi8>, tensor<8x16xi8>) outs(%[[FILL]] : tensor<4x16xi32>)
func.func @gemm_i8_i32(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
  %c = cinm.op.gemm %a, %b : tensor<4x8xi8>, tensor<8x16xi8> -> tensor<4x16xi32>
  return %c : tensor<4x16xi32>
}

// -----

// CHECK-LABEL: @gemv_i8_i32
// CHECK: linalg.matvec {{.*}}ins(%{{.*}}, %{{.*}} : tensor<16x8xi8>, tensor<8xi8>) outs(%{{.*}} : tensor<16xi32>)
func.func @gemv_i8_i32(%a: tensor<16x8xi8>, %x: tensor<8xi8>) -> tensor<16xi32> {
  %y = cinm.op.gemv %a, %x : tensor<16x8xi8>, tensor<8xi8> -> tensor<16xi32>
  return %y : tensor<16xi32>
}

// -----

// CHECK-LABEL: @batch_gemm_i8_i32
// CHECK: linalg.batch_matmul {{.*}}ins(%{{.*}}, %{{.*}} : tensor<2x4x8xi8>, tensor<2x8x16xi8>) outs(%{{.*}} : tensor<2x4x16xi32>)
func.func @batch_gemm_i8_i32(%a: tensor<2x4x8xi8>, %b: tensor<2x8x16xi8>) -> tensor<2x4x16xi32> {
  %c = cinm.op.batch_gemm %a, %b : tensor<2x4x8xi8>, tensor<2x8x16xi8> -> tensor<2x4x16xi32>
  return %c : tensor<2x4x16xi32>
}

// -----

// batch_gemv builds its contraction body by hand, so it has to widen the
// operands itself. Extending before the multiply is what keeps the product
// from overflowing the operand type.

// CHECK-LABEL: @batch_gemv_i8_i32
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<2x4x8xi8>, tensor<2x8xi8>)
// CHECK-SAME: outs(%{{.*}} : tensor<2x4xi32>)
// CHECK: ^bb0(%[[A:.*]]: i8, %[[X:.*]]: i8, %[[ACC:.*]]: i32):
// CHECK-DAG: %[[AE:.*]] = arith.extsi %[[A]] : i8 to i32
// CHECK-DAG: %[[XE:.*]] = arith.extsi %[[X]] : i8 to i32
// CHECK: %[[MUL:.*]] = arith.muli %[[AE]], %[[XE]] : i32
// CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[ACC]] : i32
// CHECK: linalg.yield %[[ADD]] : i32
func.func @batch_gemv_i8_i32(%a: tensor<2x4x8xi8>, %x: tensor<2x8xi8>) -> tensor<2x4xi32> {
  %y = cinm.op.batch_gemv %a, %x : tensor<2x4x8xi8>, tensor<2x8xi8> -> tensor<2x4xi32>
  return %y : tensor<2x4xi32>
}

// -----

// The equal-width case must not grow an extension.

// CHECK-LABEL: @batch_gemv_same_type
// CHECK-NOT: arith.extsi
// CHECK: arith.muli
func.func @batch_gemv_same_type(%a: tensor<2x4x8xi32>, %x: tensor<2x8xi32>) -> tensor<2x4xi32> {
  %y = cinm.op.batch_gemv %a, %x : tensor<2x4x8xi32>, tensor<2x8xi32> -> tensor<2x4xi32>
  return %y : tensor<2x4xi32>
}
