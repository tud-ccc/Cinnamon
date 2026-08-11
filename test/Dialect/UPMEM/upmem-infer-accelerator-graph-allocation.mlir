// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1" | FileCheck %s

// The graph-level two-level solve (docs/GraphOptimizationDesign.md): profile
// each program-identity class over the rank menu, allocate the grid exactly,
// stamp each group's winning configuration onto its members. The platform has
// two ranks of 8 DPUs, so the menu is {8, 16} and the grid budget is 16.
//
// What is pinned here is the *structure* of the outcome -- who shares a
// configuration, that the grid is not oversubscribed -- not which menu point
// wins: that is the solver's decision over measured costs.

#upmem = #upmem.platform<type = v1A, dimensions = 2x8x16>

// Two different-shape gemvs: two classes, each a singleton. The grid must be
// split between them (8 + 8): timesharing one of them would pay the 40 ms
// program reload against sub-ms kernels. This is the design's headline
// partition-vs-timeshare case.
// CHECK-LABEL: func.func @chained
func.func @chained(%A: tensor<256x256xi32>, %x: tensor<256xi32>, %B: tensor<128x128xi32>) -> tensor<128xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x8x4
  %r = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  %s = tensor.extract_slice %r[0] [128] [1] : tensor<256xi32> to tensor<128xi32>
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x8x4
  %r2 = cinm.compute -> tensor<128xi32> {
    %g = cinm.op.gemv %B, %s : tensor<128x128xi32>, tensor<128xi32> -> tensor<128xi32>
    cinm.yield %g : tensor<128xi32>
  }
  return %r2 : tensor<128xi32>
}

// Two same-shape gemvs over different (static) weights: one class of two.
// Program identity (C8) requires both members to commit the *same*
// configuration -- whether the solver co-locates them on one set or gives
// each its own equally-sized set.
// CHECK-LABEL: func.func @qk
func.func @qk(%Wq: tensor<256x256xi32> {cinm.static}, %Wk: tensor<256x256xi32> {cinm.static}, %x: tensor<256xi32>)
    -> (tensor<256xi32>, tensor<256xi32>)
    attributes {cinm.available_platforms = [#upmem]} {
  // CHECK: cinm.compute_block on accelerator #upmem.array<[[SHAPE:[0-9x]+]],
  %q = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %Wq, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  // CHECK: cinm.compute_block on accelerator #upmem.array<[[SHAPE]],
  %k = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %Wk, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %q, %k : tensor<256xi32>, tensor<256xi32>
}
