// RUN: cinm-opt %s --split-input-file --cinm-annotate-operand-reuse | FileCheck %s
// RUN: cinm-opt %s --split-input-file --cinm-annotate-operand-reuse --cinm-absorb-static-slices | FileCheck %s --check-prefix=ABSORB

// Grouped-query attention: two query heads read the same KV-head slice (the
// second through a slice of its own at the same offsets), a third head the
// other slice. Each block reads its operands and writes its result whole:
// the first block touches 1024 + 16 + 256 bytes, the second as many.

// CHECK-LABEL: func.func @shared_slice
//       CHECK:   cinm.compute_block {{.*}} attributes {cinm.operand_reuse_bytes = array<i64: -1, -1>}
//       CHECK:   cinm.compute_block {{.*}} attributes {cinm.operand_reuse_bytes = array<i64: 0, -1>}
//       CHECK:   cinm.compute_block {{.*}} attributes {cinm.operand_reuse_bytes = array<i64: -1, 1296>}
func.func @shared_slice(%kv: tensor<64x32xi8>, %q0: tensor<16xi8>, %q1: tensor<16xi8>) -> (tensor<64xi32>, tensor<64xi32>, tensor<64xi32>) {
  %k = tensor.extract_slice %kv[0, 0] [64, 16] [1, 1] : tensor<64x32xi8> to tensor<64x16xi8>
  %s0 = cinm.compute_block (%a = %k : tensor<64x16xi8>, %b = %q0 : tensor<16xi8>) -> tensor<64xi32> {
    %g = cinm.op.gemv %a, %b : tensor<64x16xi8>, tensor<16xi8> -> tensor<64xi32>
    cinm.yield %g : tensor<64xi32>
  }
  %k_again = tensor.extract_slice %kv[0, 0] [64, 16] [1, 1] : tensor<64x32xi8> to tensor<64x16xi8>
  %s1 = cinm.compute_block (%a = %k_again : tensor<64x16xi8>, %b = %q1 : tensor<16xi8>) -> tensor<64xi32> {
    %g = cinm.op.gemv %a, %b : tensor<64x16xi8>, tensor<16xi8> -> tensor<64xi32>
    cinm.yield %g : tensor<64xi32>
  }
  %k1 = tensor.extract_slice %kv[0, 16] [64, 16] [1, 1] : tensor<64x32xi8> to tensor<64x16xi8>
  %s2 = cinm.compute_block (%a = %k1 : tensor<64x16xi8>, %b = %q0 : tensor<16xi8>) -> tensor<64xi32> {
    %g = cinm.op.gemv %a, %b : tensor<64x16xi8>, tensor<16xi8> -> tensor<64xi32>
    cinm.yield %g : tensor<64xi32>
  }
  return %s0, %s1, %s2 : tensor<64xi32>, tensor<64xi32>, tensor<64xi32>
}

// -----

// A layer loop, annotated as an iteration in its middle runs. The layer's
// weight is a slice at the induction variable, so new data in every
// iteration; the activation comes from the previous iteration's last block;
// the table row is the same slice in every iteration, last read by the
// previous iteration's first block (which touches 256 + 16 + 16 + 64 bytes,
// the second 64 + 16).
//
// Absorbed, the weight's slice becomes the stack and the index, which both
// take over its distance.

// CHECK-LABEL: func.func @layers
//       CHECK:   scf.for
//       CHECK:     cinm.compute_block {{.*}} attributes {cinm.operand_reuse_bytes = array<i64: -1, 0, 80>}
//       CHECK:     cinm.compute_block {{.*}} attributes {cinm.operand_reuse_bytes = array<i64: 0>}

// ABSORB-LABEL: func.func @layers
//       ABSORB:   cinm.compute_block (%{{.*}} = %arg0 : tensor<4x16x16xi8>, %{{.*}} = %{{.*}} : index, %{{.*}} : tensor<16xi8>, %{{.*}} : tensor<16xi8>) -> tensor<16xi32> attributes {cinm.operand_reuse_bytes = array<i64: -1, -1, 0, 80>}
func.func @layers(%W: tensor<4x16x16xi8> {cinm.static}, %x0: tensor<16xi8>, %table: tensor<8x16xi8>, %pos: index) -> tensor<16xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %out = scf.for %l = %c0 to %c4 step %c1 iter_args(%x = %x0) -> (tensor<16xi8>) {
    %w = tensor.extract_slice %W[%l, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<4x16x16xi8> to tensor<16x16xi8>
    %r = tensor.extract_slice %table[%pos, 0] [1, 16] [1, 1] : tensor<8x16xi8> to tensor<16xi8>
    %y = cinm.compute_block (%a = %w : tensor<16x16xi8>, %b = %x : tensor<16xi8>, %c = %r : tensor<16xi8>) -> tensor<16xi32> {
      %s = arith.addi %b, %c : tensor<16xi8>
      %g = cinm.op.gemv %a, %s : tensor<16x16xi8>, tensor<16xi8> -> tensor<16xi32>
      cinm.yield %g : tensor<16xi32>
    }
    %z = cinm.compute_block (%a = %y : tensor<16xi32>) -> tensor<16xi8> {
      %t = arith.trunci %a : tensor<16xi32> to tensor<16xi8>
      cinm.yield %t : tensor<16xi8>
    }
    scf.yield %z : tensor<16xi8>
  }
  return %out : tensor<16xi8>
}
