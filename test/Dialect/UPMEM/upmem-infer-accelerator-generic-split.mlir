// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M0=8,gemv.K0=128,gemv.M1=8,gemv.K1=64,gemv.order=0" \
// RUN: | FileCheck %s

// The configuration that motivated the whole §G redesign, lowered end to end
// by the plugin rather than by pass flags.
//
// It is the gemv_64MB optimum an independent autotuner found
// (experiments/gemv_microbenchmark/dodo.py), expressed in the generic space:
// its mramRow=64, mramCol=128, taskletCols=1, wramRow=8, wramCol=64 read as
// gemv.M0 = mramRow*taskletCols/tasklets = 8, gemv.K0 = mramCol/taskletCols =
// 128, and the leaf tile (level 1) straight from the WRAM tile (design §H5).
//
// Before §G this failed in --convert-cinm-to-cnm with
// "numParallelElts (64) % numWgItems (16384) != 0": the plugin fed per-DPU
// block sizes where a per-workgroup tile was wanted, and even corrected, the
// conversion could not split a reduction across the workgroup at all. Both are
// now expressible, so what this test guards is that the projection in
// handleGemv keeps agreeing with what --convert-linalg-to-cnm does.

#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

// CHECK-LABEL: func.func @gemv_64MB
func.func @gemv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  // CHECK: upmem.alloc_dpus with program @{{.*}} : !upmem.hierarchy<1x2048x8>
  // CHECK: upmem.dpu_program @{{.*}}() tasklets(8) {

  // Per-DPU MRAM, matching the template's own constraint
  // mramRow*mramCol + mramCol + mramRow = 64*128 + 128 + 64:
  //   A  8 tasklets x (8 x 1 x 128) = 8192
  //   x                    1 x 128  =  128   <- shared across the tasklets
  //   y  8 tasklets x (1 x 8)       =   64
  // The WRAM tiles A and x are staged into hold one wramCol=64 chunk each.
  // They are allocas, so they hoist to the top of the kernel instead of being
  // allocated once per trip of the K loop -- which is why they are checked
  // here, in one order-free group with the MRAM buffers, rather than inside
  // the loop below.
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<8x8x1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<8x1x8xi32, #upmem.mram>
  // CHECK-DAG: memref.alloca() : memref<8x1x64xi32, #upmem.wram>
  // CHECK-DAG: memref.alloca() : memref<1x64xi32, #upmem.wram>

  // The 128-wide K tile is walked in those chunks, with the output staged
  // once outside the loop. This is also what checks that the leaf tile sizes
  // survived the reduction split: without `per-dim-attrs` they arrive one
  // entry short and there is no loop here at all.
  //
  // The accumulator arrives zeroed rather than loaded: its seed is a uniform
  // constant, so --cnm-scatter-optimizations dropped the host transfer for a
  // fill on the launch parameter and --upmem-tile-mram-buffers folded that
  // into the staging buffer. A tasklet writes its own 8 elements instead of
  // reading MRAM it is about to overwrite.
  // CHECK: %[[WY:.*]] = memref.alloca() : memref<1x8xi32, #upmem.wram>
  // CHECK: scf.for %[[Z:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: memref.store %{{.*}}, %[[WY]][%{{.*}}, %[[Z]]]
  // CHECK: }
  // CHECK-NOT: upmem.local_transfer %{{.*}} into %[[WY]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: upmem.local_transfer %{{.*}} into %{{.*}} : memref<8x1x64xi32, {{.*}}#upmem.mram> to memref<8x1x64xi32, #upmem.wram>
  // CHECK: }
  // CHECK: upmem.local_transfer %[[WY]] into
  // CHECK: upmem.return

  // Nothing from the middle of the stack survives.
  // CHECK-NOT: cnm.
  // CHECK-NOT: cinm.op.
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
