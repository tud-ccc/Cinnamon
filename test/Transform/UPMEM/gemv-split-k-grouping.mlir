// RUN: cinm-opt %s \
// RUN:   --convert-linalg-to-cnm=cnm-buffer-level=mram --canonicalize --cse \
// RUN:   --eliminate-empty-tensors --one-shot-bufferize --cse --canonicalize \
// RUN:   --upmem-tile-mram-buffers --canonicalize --cse \
// RUN:   --cnm-ensure-scatter-gather-contiguous \
// RUN:   --convert-cnm-to-upmem \
// RUN: | FileCheck %s

// Pins the *grouping* a split reduction produces, not just its correctness.
//
// This is the gemv_64MB configuration an independent autotuner found optimal
// (experiments/gemv_microbenchmark/dodo.py): dpus=2048, tasklets=8,
// taskletCols=1, mramRow=64, mramCol=128, wramRow=8, wramCol=64. Projected
// onto this pass's parameters (design §G2):
//
//   b_m = mramRow * taskletCols / tasklets = 64 * 1 / 8 = 8
//   b_k = mramCol / taskletCols            = 128 / 1    = 128
//   leaf tile = [wramRow, wramCol]         = [8, 64]
//
// `taskletCols = 1` means the tasklets of one DPU split the *parallel*
// dimension and share their slice of the vector. That is only reproducible
// because §G3 orders the split dimension outermost; with it innermost the
// tasklets would split K instead, and the vector would be replicated eight
// times in MRAM.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 64x64>
#acc = #upmem.array<1x2048x8, #pf>

// CHECK-LABEL: func.func @gemv_64MB
func.func @gemv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>, %y: tensor<4096xi32>) -> tensor<4096xi32> {
  // CHECK: upmem.dpu_program @program() tasklets(8) {

  // The per-DPU footprint matches the template's MRAM constraint exactly
  // (mramRow*mramCol + mramCol + mramRow = 64*128 + 128 + 64):
  //
  //   A  8 tasklets x (8 x 1 x 128) = 8192
  //   x                    1 x 128  =  128   <- shared, no tasklet dimension
  //   y  8 tasklets x (1 x 8)       =   64
  //
  // The absence of a leading tasklet dimension on the vector is the load
  // bearing part: it is what `isMramBroadcastOverThreads` decides, and it only
  // holds because the scatter map for the vector simplifies to `dpu floordiv
  // 64` -- no tasklet term at all.
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<8x8x1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc @{{.*}}(mram) {{.*}} : memref<8x1x8xi32, #upmem.mram>

  // CHECK: upmem.return
  // CHECK-NOT: cnm.
  %r = cinm.compute on accelerator #acc -> tensor<4096xi32> {
    %g = linalg.contract indexing_maps = [#m, #v, #r]
      {cnm.tile_sizes = array<i64: 8, 128>,
       upmem.leaf_tile_sizes = array<i64: 8, 64>}
      ins(%A, %x : tensor<4096x4096xi32>, tensor<4096xi32>)
      outs(%y : tensor<4096xi32>) -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  func.return %r : tensor<4096xi32>
}
