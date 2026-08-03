// RUN: cinm-opt %s --cinm-isolate-compute-blocks --verify-diagnostics \
// RUN:   --upmem-infer-accelerator="lowering=generic simulator=op-count eval-solution=dpus=1024,tasklets=8,gemv.M0=8,gemv.K0=256,gemv.M1=8,gemv.K1=256,gemv.order=0"

// A configuration the search space accepts and the lowered program refutes.
//
// The space's a-priori capacity bound assumes maximal sharing, so for WRAM it
// charges one copy of each leaf tile: 8*256 + 256 + 8 = 2312 i32, well inside
// the 14336 a DPU has. That bound is deliberately loose -- it exists only to
// prune what could not fit under *any* layout, because whether the tasklets of
// a DPU share a buffer or replicate it is decided during lowering, not by the
// configuration (docs/CnmMemoryLevelsDesign.md §H3).
//
// Here they replicate: each of the 8 tasklets gets its own staging buffers, so
// the real figure is 8 x 10272 bytes against 57344. --upmem-check-occupancy
// measures that on the final IR and rejects the trial, which is what keeps the
// loose bound honest.

#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

func.func @gemv_too_big_a_leaf_tile(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  // expected-error @below {{Pipeline failed: WRAM occupancy of 82176 bytes exceeds the 57344 bytes a DPU has (8 tasklets x 10272 bytes of stack, plus 0 bytes of static buffers)}}
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
