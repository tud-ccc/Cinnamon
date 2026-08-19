// RUN: cinm-opt %s --cinm-complete-compute-graph --split-input-file --verify-diagnostics | FileCheck %s
// RUN: cinm-opt %s --cinm-complete-compute-graph=demote-nested-compute --split-input-file 2>/dev/null | FileCheck %s --check-prefix=DEMOTE

// The host ops between the two compute ops become a single host block. Its
// results are the values that escape the group: the inserted tensor (used by
// the return) and the extracted slice (used by the second compute op).

// CHECK-LABEL: func @bridge
// CHECK:       %[[V:.*]] = cinm.compute -> tensor<8xf32>
// CHECK:       %[[H:.*]]:2 = cinm.compute on platform #cinm.host_platform -> tensor<64xf32>, tensor<8xf32> attributes {cinm.available_platforms = [#cinm.host_platform]}
// CHECK:         %[[S:.*]] = tensor.insert_slice %[[V]]
// CHECK:         %[[E:.*]] = tensor.extract_slice %[[S]]
// CHECK:         cinm.yield %[[S]], %[[E]]
// CHECK:       %[[W:.*]] = cinm.compute -> tensor<8xf32>
// CHECK:         cinm.op.elementwise mul %[[H]]#1, %[[H]]#1
// CHECK:       return %[[H]]#0, %[[W]]
func.func @bridge(%a: tensor<8x8xf32>, %x: tensor<8xf32>, %d: tensor<64xf32>) -> (tensor<64xf32>, tensor<8xf32>) {
  %v = cinm.compute -> tensor<8xf32> {
    %g = cinm.op.gemv %a, %x : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %g : tensor<8xf32>
  }
  %s = tensor.insert_slice %v into %d[8] [8] [1] : tensor<8xf32> into tensor<64xf32>
  %e = tensor.extract_slice %s[0] [8] [1] : tensor<64xf32> to tensor<8xf32>
  %w = cinm.compute -> tensor<8xf32> {
    %m = cinm.op.elementwise mul %e, %e : tensor<8xf32>
    cinm.yield %m : tensor<8xf32>
  }
  return %s, %w : tensor<64xf32>, tensor<8xf32>
}

// -----

// A control flow op without compute ops inside is absorbed whole into a host
// block. Constants stay outside.

// CHECK-LABEL: func @absorb_loop
// CHECK:       arith.constant
// CHECK:       %[[V:.*]] = cinm.compute -> tensor<8xf32>
// CHECK:       %[[H:.*]] = cinm.compute on platform #cinm.host_platform -> tensor<8xf32>
// CHECK-NOT:     arith.constant
// CHECK:         %[[L:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[V]])
// CHECK:         cinm.yield %[[L]]
// CHECK:       return %[[H]]
func.func @absorb_loop(%t: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %f = arith.constant 5.000000e-01 : f32
  %v = cinm.compute -> tensor<8xf32> {
    %m = cinm.op.elementwise mul %t, %t : tensor<8xf32>
    cinm.yield %m : tensor<8xf32>
  }
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %v) -> tensor<8xf32> {
    %x = tensor.extract %acc[%i] : tensor<8xf32>
    %y = arith.mulf %x, %f : f32
    %ins = tensor.insert %y into %acc[%i] : tensor<8xf32>
    scf.yield %ins : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}

// -----

// A control flow op with a compute op inside cannot be wrapped: by default it
// is left in place with a warning, with demote-nested-compute the nested
// compute op is dissolved and the loop becomes a single host block.

// CHECK-LABEL: func @barrier
// CHECK-NOT:   #cinm.host_platform
// CHECK:       scf.for
// CHECK:         cinm.compute
// CHECK:       return

// DEMOTE-LABEL: func @barrier
// DEMOTE:       cinm.compute on platform #cinm.host_platform -> tensor<8xf32>
// DEMOTE:         scf.for
// DEMOTE-NOT:       cinm.compute
// DEMOTE:           cinm.op.elementwise add
// DEMOTE:         cinm.yield
// DEMOTE:       return
func.func @barrier(%t: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  // expected-warning @below {{op contains compute ops and cannot be wrapped}}
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %t) -> tensor<8xf32> {
    %w = cinm.compute -> tensor<8xf32> {
      %m = cinm.op.elementwise add %acc, %acc : tensor<8xf32>
      cinm.yield %m : tensor<8xf32>
    }
    scf.yield %w : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}

// -----

// A function without compute ops has no graph to complete and stays as it is.

// CHECK-LABEL: func @no_compute
// CHECK-NOT:   cinm.compute
func.func @no_compute(%t: tensor<8xf32>, %d: tensor<64xf32>) -> tensor<64xf32> {
  %s = tensor.insert_slice %t into %d[8] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %s : tensor<64xf32>
}

// -----

// A call whose callee contains compute ops is a hole in the graph, like
// control flow that cannot be wrapped. The host ops around it still get
// their blocks.

// CHECK-LABEL: func @opaque_call
// CHECK:       %[[V:.*]] = cinm.compute -> tensor<64xf32>
// CHECK:       %[[H1:.*]] = cinm.compute on platform #cinm.host_platform -> tensor<8xf32>
// CHECK:         tensor.extract_slice %[[V]]
// CHECK:       %[[C:.*]] = call @callee(%[[H1]])
// CHECK:       %[[H2:.*]] = cinm.compute on platform #cinm.host_platform -> tensor<64xf32>
// CHECK:         tensor.insert_slice %[[C]]
// CHECK:       return %[[H2]]
func.func @opaque_call(%t: tensor<64xf32>) -> tensor<64xf32> {
  %v = cinm.compute -> tensor<64xf32> {
    %m = cinm.op.elementwise mul %t, %t : tensor<64xf32>
    cinm.yield %m : tensor<64xf32>
  }
  %e = tensor.extract_slice %v[0] [8] [1] : tensor<64xf32> to tensor<8xf32>
  // expected-warning @below {{callee contains compute ops}}
  %c = func.call @callee(%e) : (tensor<8xf32>) -> tensor<8xf32>
  %s = tensor.insert_slice %c into %v[8] [8] [1] : tensor<8xf32> into tensor<64xf32>
  return %s : tensor<64xf32>
}

func.func @callee(%t: tensor<8xf32>) -> tensor<8xf32> {
  %m = cinm.compute -> tensor<8xf32> {
    %r = cinm.op.elementwise mul %t, %t : tensor<8xf32>
    cinm.yield %r : tensor<8xf32>
  }
  return %m : tensor<8xf32>
}
