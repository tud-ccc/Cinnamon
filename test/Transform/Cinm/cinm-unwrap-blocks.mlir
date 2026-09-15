// RUN: cinm-opt %s --cinm-unwrap-compute-blocks | FileCheck %s

// -----

// The body's ops are hoisted to where the compute was, and the result is
// replaced by what was yielded.
// CHECK-LABEL: func @unwrap_compute
// CHECK-NOT:   cinm.compute
// CHECK:       %[[SUM:.*]] = arith.addi %[[A:.*]], %[[B:.*]]
// CHECK:       return %[[SUM]]
func.func @unwrap_compute(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %x = cinm.compute -> tensor<4xi32> {
    %sum = arith.addi %a, %b : tensor<4xi32>
    cinm.yield %sum : tensor<4xi32>
  }
  return %x : tensor<4xi32>
}

// -----

// cinm.compute is not IsolatedFromAbove, so yielding a value defined outside
// it is legal -- and there is then nothing in the body to hoist for that
// result. The unwrap has to forward the outside value itself; looking it up
// as though the body had produced it asserts.
// CHECK-LABEL: func @yield_value_from_above
// CHECK-SAME:  (%[[ARG:.*]]: tensor<4xi32>)
// CHECK-NOT:   cinm.compute
// CHECK:       return %[[ARG]]
func.func @yield_value_from_above(%a: tensor<4xi32>) -> tensor<4xi32> {
  %x = cinm.compute -> tensor<4xi32> {
    cinm.yield %a : tensor<4xi32>
  }
  return %x : tensor<4xi32>
}

// -----

// The same, with one result from the body and one from above, so the two
// paths are exercised in a single op.
// CHECK-LABEL: func @yield_mixed
// CHECK-SAME:  (%[[A:.*]]: tensor<4xi32>, %[[B:.*]]: tensor<4xi32>)
// CHECK-NOT:   cinm.compute
// CHECK:       %[[SUM:.*]] = arith.addi %[[A]], %[[B]]
// CHECK:       return %[[SUM]], %[[A]]
func.func @yield_mixed(%a: tensor<4xi32>, %b: tensor<4xi32>)
    -> (tensor<4xi32>, tensor<4xi32>) {
  %x, %y = cinm.compute -> tensor<4xi32>, tensor<4xi32> {
    %sum = arith.addi %a, %b : tensor<4xi32>
    cinm.yield %sum, %a : tensor<4xi32>, tensor<4xi32>
  }
  return %x, %y : tensor<4xi32>, tensor<4xi32>
}

// -----

// A compute_block is IsolatedFromAbove, so its operands reach the body only
// as block arguments; unwrapping substitutes the operand for the argument.
// CHECK-LABEL: func @unwrap_compute_block
// CHECK-SAME:  (%[[A:.*]]: tensor<4xi32>, %[[B:.*]]: tensor<4xi32>)
// CHECK-NOT:   cinm.compute_block
// CHECK:       %[[SUM:.*]] = arith.addi %[[A]], %[[B]]
// CHECK:       return %[[SUM]]
func.func @unwrap_compute_block(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %x = cinm.compute_block (%ba = %a : tensor<4xi32>, %bb = %b : tensor<4xi32>)
      -> tensor<4xi32> {
    %sum = arith.addi %ba, %bb : tensor<4xi32>
    cinm.yield %sum : tensor<4xi32>
  }
  return %x : tensor<4xi32>
}

// -----

// Yielding a block argument straight through: the operand it stands for is
// what the result becomes.
// CHECK-LABEL: func @block_yields_its_argument
// CHECK-SAME:  (%[[ARG:.*]]: tensor<4xi32>)
// CHECK-NOT:   cinm.compute_block
// CHECK:       return %[[ARG]]
func.func @block_yields_its_argument(%a: tensor<4xi32>) -> tensor<4xi32> {
  %x = cinm.compute_block (%ba = %a : tensor<4xi32>) -> tensor<4xi32> {
    cinm.yield %ba : tensor<4xi32>
  }
  return %x : tensor<4xi32>
}
