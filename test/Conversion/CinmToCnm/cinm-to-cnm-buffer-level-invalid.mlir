// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm=cnm-buffer-level=nosuchlevel %s -verify-diagnostics

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<1x16x1, #upmem_platform>

func.func @unknown_level(%A: tensor<16x1024xi32>, %x: tensor<1024xi32>) -> tensor<16xi32> {
  %r0 = cinm.compute on accelerator #upmem -> tensor<16xi32> {
    // The pattern reports the bad level, then declines the op, so the
    // conversion driver additionally reports it as unlegalized.
    // expected-error @+2 {{unknown memory level 'nosuchlevel' for platform 'upmem'; known levels are 'mram', 'wram'}}
    // expected-error @+1 {{failed to legalize operation 'cinm.op.gemv'}}
    %r = cinm.op.gemv %A, %x : tensor<16x1024xi32>, tensor<1024xi32> -> tensor<16xi32>
    cinm.yield %r : tensor<16xi32>
  }
  func.return %r0 : tensor<16xi32>
}
