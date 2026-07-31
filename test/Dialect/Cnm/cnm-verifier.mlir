// RUN: cinm-opt %s --split-input-file -verify-diagnostics

func.func @transfer_shape_mismatch(%a: memref<64xi32, #upmem.mram>, %b: memref<32xi32, #upmem.wram>) {
  // expected-error @below {{source shape 64 does not match target shape 32}}
  cnm.local_transfer %a into %b : memref<64xi32, #upmem.mram> to memref<32xi32, #upmem.wram>
  return
}

// -----

func.func @transfer_elt_mismatch(%a: memref<64xi32, #upmem.mram>, %b: memref<64xf32, #upmem.wram>) {
  // expected-error @below {{source element type 'i32' does not match target element type 'f32'}}
  cnm.local_transfer %a into %b : memref<64xi32, #upmem.mram> to memref<64xf32, #upmem.wram>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#upmem_1_16_1 = #upmem.array<1x16x1, #upmem>

// A launch block argument's memory space must be the buffer's level: that is
// what makes the level visible to the code inside the launch.
func.func @launch_arg_level_mismatch() {
  %wg = cnm.workgroup : !cnm.workgroup<#upmem_1_16_1>
  %buf = cnm.alloc() for %wg : !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.mram>
  // expected-error @below {{Mismatched type for launch argument, expected 'memref<64xi32, #upmem.mram>', got 'memref<64xi32, #upmem.wram>'}}
  "cnm.launch"(%wg, %buf) <{operandSegmentSizes = array<i32: 1, 1, 0>}> ({
  ^bb0(%a: memref<64xi32, #upmem.wram>):
    "cnm.terminator"() : () -> ()
  }) : (!cnm.workgroup<#upmem_1_16_1>, !cnm.buffer<64xi32 on #upmem_1_16_1, #upmem.mram>) -> ()
  cnm.free_workgroup %wg : !cnm.workgroup<#upmem_1_16_1>
  return
}
