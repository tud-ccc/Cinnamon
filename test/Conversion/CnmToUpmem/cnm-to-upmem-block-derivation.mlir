// RUN: cinm-opt %s --convert-cnm-to-upmem --mlir-print-local-scope | FileCheck %s

// A cnm map is pointwise and says nothing about blocks. What travels as one
// block is derived here, and a block has to be one run in memory, so the
// answer depends on the host value's layout as much as on the map.
//
// Both scatters below have the same map and the same shapes; only the strides
// differ. The packed host gives each leaf its whole 2x8 buffer in one
// transfer; the strided one has a gap between its rows, so the derivation
// stops at the 8-element run the layout actually has and the leaf takes two
// blocks instead.

#wg = #upmem.array<4x2, <type = v1A, dpus = 4096, tasklets = 1>>

// CHECK-LABEL: func.func @layout_bounds_the_block
// CHECK: upmem.scatter_blocks %{{.*}}[16 elts, {{.*}}, 1 blocks] {{.*}} : memref<4x2x8xi32> onto
// CHECK: upmem.scatter_blocks %{{.*}}[8 elts, {{.*}}, 2 blocks] {{.*}} : memref<4x2x8xi32, strided<[64, 16, 1]>> onto
func.func @layout_bounds_the_block(%packed: memref<4x2x8xi32>,
                                   %strided: memref<4x2x8xi32, strided<[64, 16, 1]>>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %packedBuf = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg>
  %stridedBuf = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg>
  cnm.scatter %packed into %packedBuf[affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>] of %wg
      : memref<4x2x8xi32> into !cnm.buffer<2x8xi32 on #wg>
  cnm.scatter %strided into %stridedBuf[affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>] of %wg
      : memref<4x2x8xi32, strided<[64, 16, 1]>> into !cnm.buffer<2x8xi32 on #wg>
  cnm.launch %wg ins(%a = %packedBuf : <2x8xi32>, %b = %stridedBuf : <2x8xi32>)
      on !cnm.workgroup<#wg> {
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}
