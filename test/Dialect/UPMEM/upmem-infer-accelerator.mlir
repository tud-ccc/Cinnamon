// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="lowering=templates simulator=op-count max-evals=2 n-init=2 fixed-dpus=16 fixed-tasklets=4" | FileCheck %s
// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="lowering=generic simulator=op-count max-evals=2 n-init=2 fixed-dpus=16 fixed-tasklets=4" | FileCheck %s --check-prefix=GENERIC

// Inference picks a configuration and commits the module lowered with it. The
// search is pinned down here (fixed dpus/tasklets, two evaluations, the cheap
// simulator) because what is under test is that each lowering path produces a
// well-formed UPMEM program, not what the search converges to.
//
// The shapes must be static: the search space is built from the problem
// dimensions.

#upmem = #upmem.platform<type = v1A, dimensions = 1x16x16>

// CHECK-LABEL: func.func @gemv
// GENERIC-LABEL: func.func @gemv
func.func @gemv(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32> {
  // The chosen accelerator is committed on the compute block.
  // CHECK: cinm.compute_block on accelerator #upmem.array<1x16x4
  // GENERIC: cinm.compute_block on accelerator #upmem.array<1x16x4

  // Host side.
  // CHECK: upmem.alloc_dpus with program
  // CHECK: upmem.scatter_on_array
  // CHECK: upmem.gather_from_array
  // GENERIC: upmem.alloc_dpus with program
  // GENERIC: upmem.scatter_on_array
  // GENERIC: upmem.gather_from_array

  // Device side. Both paths produce an MRAM/WRAM split; the generic one gets
  // there through cnm and --upmem-tile-mram-buffers rather than from a
  // hand-written generator, so nothing from the middle of the stack may
  // survive.
  // CHECK: upmem.dpu_program @{{.*}}() tasklets(4)
  // CHECK: upmem.static_alloc @{{.*}}(mram)
  // CHECK: upmem.local_transfer

  // GENERIC: upmem.dpu_program @{{.*}}() tasklets(4)
  // GENERIC: upmem.static_alloc @{{.*}}(mram)
  // GENERIC: memref.alloca() : memref<{{.*}}, #upmem.wram>
  // GENERIC: upmem.local_transfer
  // GENERIC-NOT: cnm.
  // GENERIC-NOT: cinm.op.
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r : tensor<256xi32>
}
