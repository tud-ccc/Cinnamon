// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order[0]=2,gemv.order[1]=1" \
// RUN: | FileCheck %s --check-prefix=SPEC
// RUN: cinm-opt %s --cinm-isolate-compute-blocks \
// RUN:   --upmem-infer-accelerator="enable-scatter-specialisation=false simulator=op-count eval-solution=dpus=2048,tasklets=8,gemv.M.mram=8,gemv.K.mram=128,gemv.M.wram=8,gemv.K.wram=64,gemv.order[0]=2,gemv.order[1]=1" \
// RUN: | FileCheck %s --check-prefix=NOSPEC

// The capability-ablation switch (paper A1). Scatter specialisation lets a
// uniform buffer skip the host transfer: gemv's zero accumulator is
// initialized by the leaves themselves, so only A and x are scattered. With
// enable-scatter-specialisation=false both rewrite sites are off and the
// accumulator's init rides a third host scatter, like any other operand.
// The space itself is identical either way -- the same eval-solution vector
// is accepted by both runs.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// SPEC-LABEL: func.func @gemv_64MB
// SPEC-COUNT-2: upmem.scatter
// SPEC-NOT: upmem.scatter

// NOSPEC-LABEL: func.func @gemv_64MB
// NOSPEC-COUNT-3: upmem.scatter
// NOSPEC-NOT: upmem.broadcast
func.func @gemv_64MB(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
