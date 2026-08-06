// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order=1" \
// RUN: | FileCheck %s --check-prefixes=CHECK,SHARED
// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order=2" \
// RUN: | FileCheck %s --check-prefixes=CHECK,REPLICATED
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order=3" 2>&1 \
// RUN: | FileCheck %s --check-prefix=RANGE

// `<op>.order` reaching the device program, on the configuration of
// upmem-infer-accelerator-generic-split.mlir.
//
// This is the parameter design §G3 argued for and then declined to add: which
// tile dimension varies fastest across the leaves. It is not a speed knob. The
// leaves of one DPU are its tasklets, MRAM is per-DPU, and
// --convert-cnm-to-upmem decides syntactically -- from whether the scatter map
// mentions the tasklet dimension -- whether a buffer is stored once per DPU or
// once per tasklet. So the order changes the *footprint*, and with it what
// fits.
//
// The vector is the operand that shows it: indexed by the reduction dimension
// alone, it is what the tasklets of one DPU either share or replicate.

#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

// CHECK-LABEL: func.func @gemv_64MB
func.func @gemv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  // CHECK: upmem.dpu_program

  // order 1, the default rule: the k-tile index is outermost, so the eight
  // tasklets of a DPU differ in their m-tile and share one k-tile of the
  // vector. 128 elements per DPU. This is the grouping the independent
  // autotuner found best (taskletCols = 1).
  // SHARED-DAG: upmem.static_alloc {{.*}} : memref<1x128xi32, #upmem.mram>

  // order 2: the k-tile index is innermost, so adjacent leaves differ in it and
  // each tasklet needs its own k-tile. Same configuration otherwise, 8x the
  // vector storage.
  // REPLICATED-DAG: upmem.static_alloc {{.*}} : memref<8x1x128xi32, #upmem.mram>

  // The matrix is tiled per tasklet either way, and the accumulator likewise.
  // CHECK-DAG: upmem.static_alloc {{.*}} : memref<8x8x1x128xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc {{.*}} : memref<8x1x8xi32, #upmem.mram>

  // Two dimensions are spread over the workgroup here (4096/8 m-tiles and
  // 4096/128 k-tiles), so the space offers two orders and no more -- ranked
  // from 1, like every other parameter. The space rejects the rest rather than
  // the pass: an index the op has no order for is a configuration the search is
  // never offered, not a trial that fails.
  // RANGE: Configuration is not one this space contains
  // RANGE: gemv.order=3 is not a value this parameter can take
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
