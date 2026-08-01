// RUN: cinm-opt %s --split-input-file -verify-diagnostics

func.func @transfer_shape_mismatch(%a: memref<64xi32, #upmem.mram>, %b: memref<32xi32, #upmem.wram>) {
  // expected-error @below {{source shape 64 is not compatible with target shape 32}}
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


// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

// A map may name a host index for every buffer element, or leave a suffix of
// the buffer dimensions implicit and transfer them as a block. Both extremes
// are well-formed, and describe the same transfer here.
func.func @scatter_map_forms(%host: memref<8x16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  // pointwise: leaf (r, d, t) element i comes from host[d * 2 + t, i]
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2, i) -> (d1 * 2 + d2, i)>] of %wg
      : memref<8x16xi32> into !cnm.buffer<16xi32 on #wg4x2>
  // one 16-element block per leaf, covering the trailing host dimension
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2) -> (d1 * 2 + d2)>] of %wg
      : memref<8x16xi32> into !cnm.buffer<16xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

func.func @scatter_map_wrong_result_count(%host: memref<8x16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  // expected-error @below {{map has 1 result(s) and leaves 0 buffer dimension(s) implicit, which does not add up to the 2 dimension(s) of the host value}}
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2, i) -> (d1 * 2 + d2)>] of %wg
      : memref<8x16xi32> into !cnm.buffer<16xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

func.func @scatter_map_too_many_dims(%host: memref<8x16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  // expected-error @below {{map has 5 dimension(s); expected the workgroup's 3, optionally followed by up to 1 leading buffer dimension(s)}}
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2, i, j) -> (d1 * 2 + d2, i + j)>] of %wg
      : memref<8x16xi32> into !cnm.buffer<16xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

// An implicit block is a whole sub-array of the host value, so the dimensions
// it covers have to match it extent for extent.
func.func @scatter_block_shape_mismatch(%host: memref<8x32xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  // expected-error @below {{the implicit block has shape 16 but the host dimensions it covers have shape 32}}
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2) -> (d1 * 2 + d2)>] of %wg
      : memref<8x32xi32> into !cnm.buffer<16xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

// The old contract made this unrepresentable; now it has to be computed. Leaf
// (r, 3, 1) starts its block one row past the end.
func.func @scatter_out_of_bounds(%host: memref<8x16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  // expected-error @below {{transfer reaches element 143 of a host value that has only 128}}
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2) -> (d1 * 2 + d2 + 1)>] of %wg
      : memref<8x16xi32> into !cnm.buffer<16xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

// The bound comes from interval arithmetic, so the floordiv/mod that
// linearizing a tile space introduces do not defeat it.
func.func @scatter_out_of_bounds_through_mod(%host: memref<8x4xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<4xi32 on #wg4x2>
  // expected-error @below {{transfer reaches element 47 of a host value that has only 32}}
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2) -> ((d1 * 2 + d2) mod 8 + 4)>] of %wg
      : memref<8x4xi32> into !cnm.buffer<4xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

// A scatter may be non-injective -- that is a broadcast, and the point of it.
// Every leaf gets the whole host value, so the map has nothing left to name.
func.func @scatter_may_broadcast(%host: memref<16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  cnm.scatter %host into %buf[affine_map<(d0, d1, d2) -> ()>] of %wg
      : memref<16xi32> into !cnm.buffer<16xi32 on #wg4x2>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}

// -----

#upmem = #upmem.platform<type = v1A, dimensions = 32x128x1>
#wg4x2 = #upmem.array<1x4x2, #upmem>

// A gather may not: all 8 leaves would write the same 16 host elements.
func.func @gather_must_be_injective(%host: memref<16xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg4x2>
  %buf = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #wg4x2>
  // expected-error @below {{map is not injective: two leaves would write the same host element}}
  cnm.gather %buf[affine_map<(d0, d1, d2) -> ()>] of %wg into %host
      : !cnm.buffer<16xi32 on #wg4x2> into memref<16xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg4x2>
  return
}
