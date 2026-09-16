// RUN: cinm-opt %s --split-input-file -verify-diagnostics

// Only widening is allowed, and only within one domain. Narrowing the
// accumulator would silently drop the high bits of every product.

func.func @narrowing(%a: tensor<4x8xi32>, %b: tensor<8x16xi32>) -> tensor<4x16xi8> {
  // expected-error @below {{inferred type(s) 'tensor<4x16xi32>' are incompatible with return type(s) of operation 'tensor<4x16xi8>'}}
  // expected-error @below {{failed to infer returned types}}
  %c = cinm.op.gemm %a, %b : tensor<4x8xi32>, tensor<8x16xi32> -> tensor<4x16xi8>
  return %c : tensor<4x16xi8>
}

// -----

func.func @int_to_float(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xf32> {
  // expected-error @below {{inferred type(s) 'tensor<4x16xi8>' are incompatible with return type(s) of operation 'tensor<4x16xf32>'}}
  // expected-error @below {{failed to infer returned types}}
  %c = cinm.op.gemm %a, %b : tensor<4x8xi8>, tensor<8x16xi8> -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}

// -----

// Widening does not license a shape change.

func.func @wrong_shape(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x8xi32> {
  // expected-error @below {{inferred type(s) 'tensor<4x16xi8>' are incompatible with return type(s) of operation 'tensor<4x8xi32>'}}
  // expected-error @below {{failed to infer returned types}}
  %c = cinm.op.gemm %a, %b : tensor<4x8xi8>, tensor<8x16xi8> -> tensor<4x8xi32>
  return %c : tensor<4x8xi32>
}

// -----

// The two multiplied operands must still share one element type: a mixed
// i8 x i16 product has no single natural interpretation here.

func.func @operands_disagree(%a: tensor<4x8xi8>, %b: tensor<8x16xi16>) -> tensor<4x16xi32> {
  // expected-error @below {{operand types are not compatible}}
  // expected-error @below {{failed to infer returned types}}
  %c = cinm.op.gemm %a, %b : tensor<4x8xi8>, tensor<8x16xi16> -> tensor<4x16xi32>
  return %c : tensor<4x16xi32>
}
