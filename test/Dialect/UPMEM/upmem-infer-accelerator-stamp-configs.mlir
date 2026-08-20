// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1 allocation-granularity=4 stamp-configs=1" | FileCheck %s

// Commit by stamping: the winning configuration lands on the block as
// attributes and the body stays in the converted linalg form -- no lowered
// code is spliced. What is pinned here is the *shape* of the artifact: the
// accelerator on the block, the resolved tiling attributes on the distributed
// op, the forwarded workgroup, and that the body is still linalg on tensors.
// Which menu point wins is the solver's decision over measured costs.

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

// CHECK-LABEL: func.func @gemv
// CHECK:       %[[WG:.*]] = upmem.alloc_dpus : !upmem.hierarchy<
// CHECK:       cinm.compute_block on accelerator #upmem.array<
// CHECK-SAME:    (%{{.*}} = %arg0 : tensor<256x256xi32>, %{{.*}} = %arg1 : tensor<256xi32>, %{{.*}} = %[[WG]] : !upmem.hierarchy<
// CHECK-NOT:     cnm.launch
// CHECK:         cinm.debug_tag = "cinm.op.gemv"
// CHECK-SAME:      cnm.tile_sizes = array<i64:
// CHECK-SAME:      cnm.workgroup_dim_order = array<i64:
// CHECK-SAME:      upmem.leaf_tile_sizes = array<i64:
// CHECK-NOT:       upmem.outer_tile_params
// CHECK:         cinm.yield
// CHECK:       upmem.free_dpus %[[WG]]
func.func @gemv(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r : tensor<256xi32>
}
