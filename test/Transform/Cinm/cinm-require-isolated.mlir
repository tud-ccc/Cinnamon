// RUN: cinm-opt %s --split-input-file --verify-diagnostics --cinm-absorb-static-slices
// RUN: cinm-opt %s --split-input-file --verify-diagnostics --cinm-expand-compute-scope
// RUN: cinm-opt %s --split-input-file --verify-diagnostics --cinm-complete-compute-graph

// A pass that takes compute ops in the isolated form fails on a cinm.compute
// instead of skipping it.

func.func @not_isolated(%a: tensor<8x8xf32>, %x: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @below {{takes compute ops isolated from above (cinm.compute_block): run --cinm-isolate-compute-blocks first}}
  %v = cinm.compute -> tensor<8xf32> {
    %g = cinm.op.gemv %a, %x : tensor<8x8xf32>, tensor<8xf32> -> tensor<8xf32>
    cinm.yield %g : tensor<8xf32>
  }
  return %v : tensor<8xf32>
}
