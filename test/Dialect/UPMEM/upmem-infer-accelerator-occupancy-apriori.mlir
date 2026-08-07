// RUN: cinm-opt %s --cinm-isolate-compute-blocks --verify-diagnostics \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=1024,tasklets=8,gemv.M.mram=8,gemv.K.mram=256,gemv.M.wram=8,gemv.K.wram=256,gemv.order[0]=2,gemv.order[1]=1"

// The same shape as upmem-infer-accelerator-occupancy.mlir, one leaf tile
// larger: 8 * (8*256 + 256 + 8) = 18496 i32 against the 14336 a DPU has.
//
// Charging one copy per tasklet is what rejects this before any lowering runs.
// Charging one copy per DPU -- assuming the tasklets share every buffer -- put
// it in the space, and only the occupancy check on the final IR threw it out.

#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @gemv_leaf_tile_rejected_by_the_space(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  // expected-error @below {{Configuration is not one this space contains}}
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
