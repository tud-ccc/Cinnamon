// RUN: cinm-opt %s --cinm-deisolate-compute-blocks | FileCheck %s

// -----

// No captures: no block args to clean up — should work.
// CHECK-LABEL: func @no_captures
// CHECK:      %[[C0:.*]] = arith.constant 0 : i32
// CHECK:      %{{.*}} = cinm.compute -> tensor<2xi32> {
// CHECK-NOT:    ^bb0(
// CHECK:          tensor.generate
// CHECK:          tensor.yield %[[C0]]
// CHECK:          cinm.yield
func.func @no_captures() -> tensor<2xi32> {
  %x = cinm.compute_block () -> tensor<2xi32> {
    %c0 = arith.constant 0 : i32
    %t = tensor.generate {
      ^bb0(%i: index):
        tensor.yield %c0 : i32
    } : tensor<2xi32>
    cinm.yield %t : tensor<2xi32>
  }
  return %x : tensor<2xi32>
}

// -----

// Single capture: block arg must be deleted and replaced by the operand.
// CHECK-LABEL: func @single_capture
// CHECK:      %[[V:.*]] = cinm.compute -> tensor<6x6xi32> {
// CHECK-NOT:    ^bb0(
// CHECK:          cinm.yield %[[ARG0:.*]] : tensor<6x6xi32>
// CHECK:      return %[[V]]
func.func @single_capture(%arg0: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %x = cinm.compute_block (%barg0 = %arg0: tensor<6x6xi32>) -> tensor<6x6xi32> {
    cinm.yield %barg0 : tensor<6x6xi32>
  }
  return %x : tensor<6x6xi32>
}

// -----

// Multiple captures: all block args replaced, none remain.
// CHECK-LABEL: func @multi_capture
// CHECK:      cinm.compute -> tensor<4xi32> {
// CHECK-NOT:    ^bb0(
// CHECK:          %[[SUM:.*]] = arith.addi %[[A:.*]], %[[B:.*]]
// CHECK:          cinm.yield %[[SUM]]
func.func @multi_capture(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %x = cinm.compute_block (%ba = %a : tensor<4xi32>, %bb = %b : tensor<4xi32>) -> tensor<4xi32> {
    %sum = arith.addi %ba, %bb : tensor<4xi32>
    cinm.yield %sum : tensor<4xi32>
  }
  return %x : tensor<4xi32>
}

// -----

// Attributes on compute_block are preserved on the resulting compute.
// CHECK-LABEL: func @attrs_preserved
// CHECK:      cinm.compute -> tensor<2xi32> attributes {some.attr = 42 : i32}
func.func @attrs_preserved() -> tensor<2xi32> {
  %x = cinm.compute_block () -> tensor<2xi32> attributes {some.attr = 42 : i32}  {
    %c0 = arith.constant 0 : i32
    %t = tensor.generate {
      ^bb0(%i: index):
        tensor.yield %c0 : i32
    } : tensor<2xi32>
    cinm.yield %t : tensor<2xi32>
  }
  return %x : tensor<2xi32>
}

// -----

// Round-trip: isolate followed by deisolate must recover the original shape.
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --cinm-deisolate-compute-blocks | FileCheck %s --check-prefix=ROUNDTRIP

// ROUNDTRIP-LABEL: func @round_trip
// ROUNDTRIP:          arith.constant dense<0> : tensor<2xi32>
// ROUNDTRIP:      cinm.compute -> tensor<2xi32> {
// ROUNDTRIP-NOT:    ^bb0(
func.func @round_trip(%t: tensor<6x6xi32>) -> tensor<2xi32> {
  %c0 = arith.constant 0 : i32
  %x = cinm.compute -> tensor<2xi32> {
    %t2 = tensor.splat %c0 : tensor<2xi32>
    cinm.yield %t2 : tensor<2xi32>
  }
  return %x : tensor<2xi32>
}
