// RUN: not cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=96,gemv.order[0]=2,gemv.order[1]=1" \
// RUN:   2>&1 | FileCheck %s --check-prefix=GUARDED
// RUN: not cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="eval-solution-force=true simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=96,gemv.order[0]=2,gemv.order[1]=1" \
// RUN:   2>&1 | FileCheck %s --check-prefix=FORCED

// gemv.K.wram=96 does not divide the MRAM tile, so the space rejects the
// configuration. By default that rejection is the answer: membership is the
// whole check, and the diagnostic names the offending value. With
// eval-solution-force the membership check is skipped and the configuration
// meets the lowering itself, whose verdict -- here a genuine failure, for a
// point that also lowers fine it would be success -- is what the
// rejected-region experiment counts. The force run must not die on the
// membership message; whatever it reports comes from the pipeline.

// GUARDED: Configuration is not one this space contains
// GUARDED: gemv.K.wram=96

// FORCED: eval-solution-force: configuration is outside the feasible set, attempting the lowering anyway
// FORCED-NOT: Configuration is not one this space contains
// FORCED: Pipeline failed

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @gemv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
