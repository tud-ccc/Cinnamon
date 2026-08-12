// RUN: cinm-opt %s --cinm-isolate-compute-blocks --upmem-infer-accelerator="simulator=op-count max-evals=2 n-init=2 fixed-dpus=16 fixed-tasklets=4" | FileCheck %s

// Inference picks a configuration and commits the module lowered with it. The
// search is pinned down here (fixed dpus/tasklets, two evaluations, the cheap
// simulator) because what is under test is that the lowering produces a
// well-formed UPMEM program, not what the search converges to.
//
// The shapes must be static: the search space is built from the problem
// dimensions.

#upmem = #upmem.platform<type = v1A, dpus = 16, tasklets = 16>

// CHECK-LABEL: func.func @gemv
func.func @gemv(%A: tensor<256x256xi32>, %x: tensor<256xi32>) -> tensor<256xi32> {
  // The chosen accelerator is committed on the compute block.
  // CHECK: cinm.compute_block on accelerator #upmem.array<16x4

  // Host side. Which *form* the result transfer takes -- one block per leaf or
  // a single array -- follows from the tiling the search happened to pick, so
  // it is not pinned here: doing so would make this a test of where the search
  // lands, which the note above says it deliberately is not.
  // CHECK: upmem.alloc_dpus
  // CHECK: upmem.load_program
  // CHECK: upmem.scatter_on_array
  // CHECK: upmem.gather_{{from_array|blocks}}

  // Device side. The MRAM/WRAM split is reached through cnm and
  // --upmem-tile-mram-buffers, so nothing from the middle of the stack may
  // survive.
  // CHECK: upmem.dpu_program @{{.*}}() tasklets(4)
  // The two allocations have no order between them, and which comes first
  // depends on the configuration; what matters is that both levels are there.
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram)
  // CHECK-DAG: memref.alloca() : memref<{{.*}}, #upmem.wram>
  // CHECK: upmem.local_transfer
  // CHECK-NOT: cnm.
  // CHECK-NOT: cinm.op.
  %r = cinm.compute -> tensor<256xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<256x256xi32>, tensor<256xi32> -> tensor<256xi32>
    cinm.yield %g : tensor<256xi32>
  }
  return %r : tensor<256xi32>
}
