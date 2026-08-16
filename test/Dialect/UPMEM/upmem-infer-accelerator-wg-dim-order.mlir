// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order[0]=2,gemv.order[1]=1" \
// RUN: | FileCheck %s --check-prefixes=CHECK,SHARED
// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order[0]=1,gemv.order[1]=2" \
// RUN: | FileCheck %s --check-prefixes=CHECK,REPLICATED
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order[0]=3,gemv.order[1]=1" 2>&1 \
// RUN: | FileCheck %s --check-prefix=RANGE
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order[0]=1,gemv.order[1]=1" 2>&1 \
// RUN: | FileCheck %s --check-prefix=REPEATED

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

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: func.func @gemv_64MB
func.func @gemv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  // CHECK: upmem.dpu_program

  // The order is one place per iteration dimension: `gemv.order[0]` is where
  // m goes and `gemv.order[1]` where k goes, place 1 being the outermost
  // workgroup axis.

  // K is staged 64 at a time out of the 128 a leaf holds, so every buffer
  // carrying K is cut into two chunks with the chunk dimension outermost --
  // that is what makes one staged chunk a contiguous run. The element counts
  // below are unaffected by it; only the order is.

  // k outermost (place 1), the default rule: the eight tasklets of a DPU
  // differ in their m-tile and share one k-tile of the vector. This is the
  // grouping the independent autotuner found best (taskletCols = 1).
  //
  // The vector nonetheless takes a tasklet dimension here, so 1024 elements
  // per DPU rather than the 128 the sharing would allow. Its transfer is
  // fragmented, and a fragmented transfer is repacked -- the block form it
  // would otherwise keep is one the SDK mishandles. The repack reorders into
  // workgroup x buffer order, which mentions the tasklet, and
  // isMramBroadcastOverThreads reads the sharing off the map. So the sharing
  // is lost to a repack that only the *fragmentation* requires: packing over
  // the dimensions the map actually uses would keep both. Until then this is
  // a real cost on exactly the configuration the evaluation cares about.
  // SHARED-DAG: upmem.static_alloc {{.*}} : memref<8x2x1x64xi32, #upmem.mram>

  // k innermost, so adjacent leaves differ in it and each tasklet needs its
  // own k-tile. Same configuration otherwise, and the same storage -- for a
  // different reason, this one intrinsic.
  // REPLICATED-DAG: upmem.static_alloc {{.*}} : memref<8x2x1x64xi32, #upmem.mram>

  // The matrix is tiled per tasklet either way, and the accumulator likewise.
  // CHECK-DAG: upmem.static_alloc {{.*}} : memref<8x2x8x1x64xi32, #upmem.mram>
  // CHECK-DAG: upmem.static_alloc {{.*}} : memref<8x1x8xi32, #upmem.mram>

  // Two dimensions are spread over the workgroup here (4096/8 m-tiles and
  // 4096/128 k-tiles), so there are two places and two orders. The space
  // rejects everything else rather than the pass: an order the op does not
  // have is a configuration the search is never offered, not a trial that
  // fails. Two ways to not have one, and they fail differently --
  //
  // a place outside the domain, which is a per-dimension check:
  // RANGE: Configuration is not one this space contains
  // RANGE: gemv.order[0]=3 is not a value this parameter can take
  //
  // ...and two items in the same place, where every value is one its
  // dimension can take and what rules it out is the distinctness the solver
  // posts across them. Nothing is left to reject it after the solve, so this
  // is the general statement rather than a named constraint.
  // REPEATED: Configuration is not one this space contains
  // REPEATED: no configuration in this space assigns these values together
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
