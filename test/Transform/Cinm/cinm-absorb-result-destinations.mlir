// RUN: cinm-opt %s --cinm-absorb-result-destinations --split-input-file | FileCheck %s

// The insertion of the result moves inside the block: the destination (and
// the loop index) become block operands and the block yields the full
// tensor. This is a finalization rewrite -- it bakes offsets into the body,
// so it must only run after the search's signature-based deduplication.

// CHECK-LABEL: func @gemv_loop
// CHECK:       scf.for %[[I:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %{{.*}})
// CHECK:         %[[R:.*]] = cinm.compute_block (%{{.*}} = %{{.*}} : tensor<768x768xf32>, %{{.*}} = %{{.*}} : tensor<768xf32>, %[[D:.*]] = %[[ACC]] : tensor<6x1024x768xf32>, %[[BI:.*]] = %[[I]] : index) -> tensor<6x1024x768xf32>
// CHECK:           %[[V:.*]] = cinm.op.gemv
// CHECK:           %[[INS:.*]] = tensor.insert_slice %[[V]] into %[[D]][%[[BI]], 3, 0] [1, 1, 768] [1, 1, 1]
// CHECK:           cinm.yield %[[INS]]
// CHECK:         scf.yield %[[R]]
func.func @gemv_loop(%w: tensor<768x768xf32>, %x: tensor<768xf32>, %out: tensor<6x1024x768xf32>) -> tensor<6x1024x768xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %r = scf.for %i = %c0 to %c6 step %c1 iter_args(%acc = %out) -> tensor<6x1024x768xf32> {
    %v = cinm.compute_block (%a = %w : tensor<768x768xf32>, %b = %x : tensor<768xf32>) -> tensor<768xf32> {
      %g = cinm.op.gemv %a, %b : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
      cinm.yield %g : tensor<768xf32>
    }
    %ins = tensor.insert_slice %v into %acc[%i, 3, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    scf.yield %ins : tensor<6x1024x768xf32>
  }
  return %r : tensor<6x1024x768xf32>
}

// -----

// A bufferization.materialize_in_destination is a destination-style consumer
// too: absorbed, the block writes the destination buffer directly.

// CHECK-LABEL: func @materialize
// CHECK:       %[[R:.*]] = cinm.compute_block (%{{.*}}, %[[D:.*]] = %{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK:         %[[M:.*]] = bufferization.materialize_in_destination %{{.*}} in %[[D]]
// CHECK:         cinm.yield %[[M]]
// CHECK:       return %[[R]]
func.func @materialize(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %d: tensor<8xf32>) -> tensor<8xf32> {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %m = bufferization.materialize_in_destination %r in %d : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %m : tensor<8xf32>
}

// -----

// A destination defined after the block is fine as long as its pure producer
// can be moved before the block.

// CHECK-LABEL: func @dest_defined_later
// CHECK:       %[[O:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:       cinm.compute_block ({{.*}}, %[[O2:.*]] = %[[O]] : tensor<64xf32>) -> tensor<64xf32>
// CHECK:         tensor.insert_slice %{{.*}} into %[[O2]][0] [8] [1]
func.func @dest_defined_later(%a: tensor<8x8xf32>, %b: tensor<8xf32>) -> tensor<64xf32> {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %dest = tensor.empty() : tensor<64xf32>
  %i = tensor.insert_slice %r into %dest[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i : tensor<64xf32>
}

// -----

// The result is used elsewhere: moving the insertion in would hide it.

// CHECK-LABEL: func @result_used_twice
// CHECK:       %[[R:.*]] = cinm.compute_block ({{.*}}) -> tensor<8xf32>
// CHECK:       %[[I:.*]] = tensor.insert_slice %[[R]]
// CHECK:       return %[[I]], %[[R]]
func.func @result_used_twice(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %d: tensor<64xf32>) -> (tensor<64xf32>, tensor<8xf32>) {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %i = tensor.insert_slice %r into %d[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %i, %r : tensor<64xf32>, tensor<8xf32>
}

// -----

// The insertion is conditional, it must stay where it is.

// CHECK-LABEL: func @conditional_insert
// CHECK:       %[[R:.*]] = cinm.compute_block ({{.*}}) -> tensor<8xf32>
// CHECK:       scf.if
// CHECK:         tensor.insert_slice %[[R]]
func.func @conditional_insert(%a: tensor<8x8xf32>, %b: tensor<8xf32>, %d: tensor<64xf32>, %p: i1) -> tensor<64xf32> {
  %r = cinm.compute_block (%x = %a : tensor<8x8xf32>, %y = %b : tensor<8xf32>) -> tensor<8xf32> {
    %v = cinm.op.gemv %x, %y : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %v : tensor<8xf32>
  }
  %i = scf.if %p -> tensor<64xf32> {
    %t = tensor.insert_slice %r into %d[0] [8] [1] : tensor<8xf32> into tensor<64xf32>
    scf.yield %t : tensor<64xf32>
  } else {
    scf.yield %d : tensor<64xf32>
  }
  return %i : tensor<64xf32>
}
