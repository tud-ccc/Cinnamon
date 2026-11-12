// RUN: cinm-opt %s --verify-diagnostics

// The reuse distances follow the operands one to one.

func.func @wrong_length(%a: tensor<16xi8>) -> tensor<16xi8> {
  // expected-error @below {{"cinm.operand_reuse_bytes" has 2 entries for 1 operands}}
  %0 = cinm.compute_block (%x = %a : tensor<16xi8>) -> tensor<16xi8> attributes {cinm.operand_reuse_bytes = array<i64: -1, 0>} {
    cinm.yield %x : tensor<16xi8>
  }
  return %0 : tensor<16xi8>
}
