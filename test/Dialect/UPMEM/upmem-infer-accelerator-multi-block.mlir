// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=2 n-init=2 fixed-dpus=16 fixed-tasklets=4" | FileCheck %s

// Inference is driven over every compute block the platform owns, not over one
// per function: the graph the whole thing optimizes (see
// docs/GraphOptimizationDesign.md) is a connected component of the dataflow
// between blocks, and a component has several blocks in it. Blocks the
// platform does not own are left untouched.
//
// What is checked here is coverage -- every owned block ends up configured and
// lowered -- not what the search converges to, hence the pinned dpus/tasklets
// and the cheap simulator.

#upmem = #upmem.platform<type = v1A, dimensions = 1x16x16>

// Chained through %r: one graph, two blocks.
// CHECK-LABEL: func.func @chained
func.func @chained(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32> {
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x16x4
  // CHECK-NOT: cinm.op.
  // CHECK: upmem.alloc_dpus
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  // The second block of the same function is searched too -- it used to be
  // reached only because the walk happened to continue.
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x16x4
  // CHECK-NOT: cinm.op.
  // CHECK: upmem.alloc_dpus
  %r2 = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %r : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r2 : tensor<256xi32>
}

// Only the function's arguments are shared: the two blocks are still one graph
// (the parallel-2MM shape), and both are configured.
// CHECK-LABEL: func.func @shared_input
func.func @shared_input(%A: tensor<256x256xi32>, %x: tensor<256xi32>, %y: tensor<256xi32>)
    -> (tensor<256xi32>, tensor<256xi32>) {
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x16x4
  // CHECK: upmem.alloc_dpus
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x16x4
  // CHECK: upmem.alloc_dpus
  %r2 = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %y : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r, %r2 : tensor<256xi32>, tensor<256xi32>
}

// The enclosing scope does not offer upmem, so this block is none of our
// business: no accelerator is committed and the cinm op survives.
// CHECK-LABEL: func.func @host_only
func.func @host_only(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32> {
  // CHECK-NOT: on accelerator
  // CHECK: cinm.op.gemv
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r : tensor<256xi32>
}
