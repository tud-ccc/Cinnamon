// RUN: cinm-opt %s --cinm-isolate-compute-blocks --verify-diagnostics \
// RUN:   --upmem-infer-accelerator="simulator=op-count eval-solution=dpus=1024,tasklets=8,gemv.M.mram=4,gemv.K.mram=512,gemv.M.wram=2,gemv.K.wram=512,gemv.order[0]=2,gemv.order[1]=1"

// A configuration the search space accepts and the lowered program refutes.
//
// The space's capacity bound charges one private copy of every leaf tile per
// tasklet: 8 * (2*512 + 512 + 2) = 12304 i32, inside the 14336 a DPU has. What
// it does not model is what lowering allocates on top -- staging buffers,
// hoisting -- which here is another 1024 bytes per tasklet, enough to put the
// real figure at 8 x 7176 bytes against 57344.
//
// So the bound is necessary and not sufficient, and --upmem-check-occupancy
// measuring the final IR is what keeps it honest. The margin is 64 bytes: a
// configuration has to sit against the ceiling to get this far.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @gemv_too_big_a_leaf_tile(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32> {
  // expected-error @below {{Pipeline failed: WRAM occupancy of 57408 bytes exceeds the 57344 bytes a DPU has (8 tasklets x 7176 bytes of stack, plus 0 bytes of static buffers)}}
  %r = cinm.compute -> tensor<4096xi32> attributes {cinm.available_platforms = [#upmem]} {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
