// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="lowering=generic simulator=op-count eval-solution=dpus=2048,tasklets=8,op0.block0=8,op0.block1=128,op0.leaf0=8,op0.leaf1=64" \
// RUN: | FileCheck %s

// The configuration that motivated the whole §G redesign, lowered end to end
// by the plugin rather than by pass flags.
//
// It is the gemv_64MB optimum an independent autotuner found
// (experiments/gemv_microbenchmark/dodo.py), expressed in the generic space:
// its mramRow=64, mramCol=128, taskletCols=1, wramRow=8, wramCol=64 read as
// block0 = mramRow*taskletCols/tasklets = 8, block1 = mramCol/taskletCols =
// 128, and the leaf tile straight from the WRAM tile (design §H5).
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
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<8x8x1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<8x1x8xi32, #upmem.mram>

  // The 128-wide K tile is walked in wramCol=64 chunks, with the output staged
  // once outside the loop. This is also what checks that the leaf tile sizes
  // survived the reduction split: without `per-dim-attrs` they arrive one
  // entry short and there is no loop here at all.
  // CHECK: %[[WY:.*]] = upmem.pwram_alloc() : memref<1x8xi32, #upmem.wram>
  // CHECK: upmem.local_transfer %{{.*}} into %[[WY]]
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK-DAG: upmem.pwram_alloc() : memref<8x1x64xi32, #upmem.wram>
  // CHECK-DAG: upmem.pwram_alloc() : memref<1x64xi32, #upmem.wram>
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
