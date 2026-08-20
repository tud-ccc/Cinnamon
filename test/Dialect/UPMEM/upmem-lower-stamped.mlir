// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=4 n-init=4 fixed-tasklets=4 graph-allocation=1 allocation-granularity=4 stamp-configs=1" --upmem-lower-stamped | FileCheck %s

// The two halves of the stamped flow chained: the search stamps
// configurations, --upmem-lower-stamped lowers the whole module in one go.
// One global bufferization crosses the block and function boundaries, so the
// block receives the function's memref arguments directly -- no defensive
// copy at the block edge -- and the lowered body launches on the forwarded
// workgroup.

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

// CHECK-LABEL: func.func @gemv
// CHECK-SAME:    (%[[A:.*]]: memref<256x256xi32>, %[[X:.*]]: memref<256xi32>
// CHECK:       %[[WG:.*]] = upmem.alloc_dpus
// CHECK-NOT:   memref.copy
// CHECK:       cinm.compute_block on accelerator #upmem.array<
// CHECK-SAME:    %[[A]] : memref<256x256xi32>
// CHECK-SAME:    %[[WG]] : !upmem.hierarchy<
// CHECK-NOT:     linalg.generic
// CHECK:         upmem.load_program
// CHECK:         upmem.wait_for
// CHECK:         upmem.gather
// CHECK:       upmem.free_dpus %[[WG]]
func.func @gemv(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<256xi32> {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r : tensor<256xi32>
}
