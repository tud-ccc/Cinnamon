// RUN: cinm-opt %s --cinm-simplify-compute-signatures | FileCheck %s

// These rewrites are deliberately NOT canonicalizations: the accelerator
// search's trial pipelines run the canonicalizer and must not change a
// block's signature (see the pass description).

// ComputeOpSimplifyYield: result that yields a value defined outside the
// compute block is removed; the compute op result is replaced with the
// external value directly.

// CHECK-LABEL: func.func @compute_yield_external
// CHECK-NOT:   cinm.compute
// CHECK:       return %arg0
func.func @compute_yield_external(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = cinm.compute -> tensor<8xf32> {
    cinm.yield %arg0 : tensor<8xf32>
  }
  return %0 : tensor<8xf32>
}

// ComputeOpSimplifyYield: one result is external (removed), one is internal
// (kept). The compute op shrinks from two results to one.

// CHECK-LABEL: func.func @compute_yield_mixed
// CHECK-NOT:   cinm.compute -> tensor<8xf32>, tensor<8xf32>
// CHECK:       %[[R:.*]] = cinm.compute -> tensor<8xf32>
// CHECK:       return %arg0, %[[R]]
func.func @compute_yield_mixed(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %0:2 = cinm.compute -> tensor<8xf32>, tensor<8xf32> {
    %x = arith.addf %arg0, %arg0 : tensor<8xf32>
    cinm.yield %arg0, %x : tensor<8xf32>, tensor<8xf32>
  }
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// ComputeBlockOpSimplifyYield: result that re-yields a block argument
// (pass-through) is replaced with the corresponding outer operand.

// CHECK-LABEL: func.func @compute_block_yield_passthrough
// CHECK-NOT:   cinm.compute_block
// CHECK:       return %arg0
func.func @compute_block_yield_passthrough(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = cinm.compute_block (%a = %arg0 : tensor<8xf32>) -> tensor<8xf32> {
    cinm.yield %a : tensor<8xf32>
  }
  return %0 : tensor<8xf32>
}

// ComputeBlockOpSimplifyYield: one pass-through result (removed) and one
// genuinely computed result (kept).

// CHECK-LABEL: func.func @compute_block_yield_mixed
// CHECK:       %[[R:.*]] = cinm.compute_block (%{{.*}} = %arg0 : tensor<8xf32>) -> tensor<8xf32>
// CHECK:       return %arg1, %[[R]]
func.func @compute_block_yield_mixed(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %0:2 = cinm.compute_block (%a = %arg0 : tensor<8xf32>, %b = %arg1 : tensor<8xf32>) -> tensor<8xf32>, tensor<8xf32> {
    %x = arith.addf %a, %a : tensor<8xf32>
    cinm.yield %b, %x : tensor<8xf32>, tensor<8xf32>
  }
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// ComputeBlockOpDeleteUnusedArgs: block argument that is never used inside
// the body is dropped together with the corresponding outer operand.

// CHECK-LABEL: func.func @compute_block_unused_arg
// CHECK:       cinm.compute_block (%{{.*}} = %arg0 : tensor<8xf32>) -> tensor<8xf32>
func.func @compute_block_unused_arg(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = cinm.compute_block (%a = %arg0 : tensor<8xf32>, %b = %arg1 : tensor<8xf32>) -> tensor<8xf32> {
    %x = arith.addf %a, %a : tensor<8xf32>
    cinm.yield %x : tensor<8xf32>
  }
  return %0 : tensor<8xf32>
}

// The canonicalizer must NOT perform these rewrites anymore: a trial
// pipeline canonicalization must keep the block's boundary intact.

// RUN: cinm-opt %s --canonicalize | FileCheck %s --check-prefix=CANON

// CANON-LABEL: func.func @canonicalize_keeps_signature
// CANON:       cinm.compute_block (%{{.*}} = %arg0 : tensor<8xf32>, %{{.*}} = %arg1 : tensor<8xf32>) -> tensor<8xf32>, tensor<8xf32>
func.func @canonicalize_keeps_signature(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %0:2 = cinm.compute_block (%a = %arg0 : tensor<8xf32>, %b = %arg1 : tensor<8xf32>) -> tensor<8xf32>, tensor<8xf32> {
    %x = arith.addf %a, %a : tensor<8xf32>
    cinm.yield %b, %x : tensor<8xf32>, tensor<8xf32>
  }
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}
