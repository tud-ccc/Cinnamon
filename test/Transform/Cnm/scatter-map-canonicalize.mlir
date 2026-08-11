// RUN: cinm-opt %s --split-input-file --canonicalize --mlir-print-local-scope | FileCheck %s
// RUN: cinm-opt %s --split-input-file --canonicalize --mlir-print-op-generic --mlir-print-local-scope | FileCheck %s --check-prefix=STORED
// The shorthand is what the parser takes, so canonicalizing the printed form
// again reproduces it exactly.
// RUN: cinm-opt %s --split-input-file --canonicalize | cinm-opt --split-input-file --canonicalize --mlir-print-local-scope | FileCheck %s

// The canonical scatter/gather map is the fully explicit one: it names a host
// index for every buffer element, so nothing downstream has to work out which
// trailing dimensions were left to the block. What is *printed* is the
// shorthand -- the block dimensions dropped again -- which is also what the
// parser accepts, so the text of a canonicalized op is a fixpoint even though
// the attribute stored in it is not the one written.

#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// CHECK-LABEL: func.func @block_is_implicit
// CHECK:       cnm.scatter %{{.*}}[affine_map<(d0, d1) -> (d0, d1 * 2)>]
// STORED:      "cnm.scatter"{{.*}}<{scatterMap = affine_map<(d0, d1, d2) -> (d0, d1 * 2, d2)>}> : (tensor<4x3x8xi32>
func.func @block_is_implicit(%a: tensor<4x3x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<8xi32 on #wg>
  cnm.scatter %a into %buf[affine_map<(d0, d1) -> (d0, d1 * 2)>] of %wg
      : tensor<4x3x8xi32> into !cnm.buffer<8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// Writing the explicit form by hand changes nothing: it is what
// canonicalization produces anyway, and it prints back the same shorthand.

// CHECK-LABEL: func.func @explicit_is_the_same_op
// CHECK:       cnm.gather %{{.*}}[affine_map<(d0, d1) -> (d0, d1)>]
func.func @explicit_is_the_same_op(%out: tensor<4x2x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<8xi32 on #wg>
  %g = cnm.gather %buf[affine_map<(d0, d1, d2) -> (d0, d1, d2)>] of %wg
      into %out : !cnm.buffer<8xi32 on #wg> into tensor<4x2x8xi32>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// A host dimension longer than the buffer dimension indexing it is not that
// dimension's block -- each leaf writes the first two of eight elements -- so
// there is no shorthand for this map and it prints in full.

// CHECK-LABEL: func.func @host_dimension_is_longer
// CHECK:       cnm.scatter %{{.*}}[affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
func.func @host_dimension_is_longer(%a: tensor<4x2x8xi32>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<2xi32 on #wg>
  cnm.scatter %a into %buf[affine_map<(d0, d1, d2) -> (d0, d1, d2)>] of %wg
      : tensor<4x2x8xi32> into !cnm.buffer<2xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}

// -----

#wg = #upmem.array<1x4x2, <type = v1A, dimensions = 32x128x1>>

// A block is one run in memory, so how far the shorthand goes depends on the
// host value's layout. Both scatters below have the same map and the same
// shapes; only the strides differ. The packed one collapses to a single
// 2x8 block per leaf, the strided one stops at the 8-element run its layout
// actually has.

// CHECK-LABEL: func.func @layout_bounds_the_block
// CHECK:       cnm.scatter %{{.*}}[affine_map<(d0, d1) -> (d0)>] {{.*}} : memref<4x2x8xi32>
// CHECK:       cnm.scatter %{{.*}}[affine_map<(d0, d1, d2) -> (d0, d2)>] {{.*}} : memref<4x2x8xi32, strided<[64, 16, 1]>>
func.func @layout_bounds_the_block(%packed: memref<4x2x8xi32>,
                                   %strided: memref<4x2x8xi32, strided<[64, 16, 1]>>) {
  %wg = cnm.workgroup : !cnm.workgroup<#wg>
  %buf = cnm.declare_buffer() for %wg : !cnm.buffer<2x8xi32 on #wg>
  cnm.scatter %packed into %buf[affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>] of %wg
      : memref<4x2x8xi32> into !cnm.buffer<2x8xi32 on #wg>
  cnm.scatter %strided into %buf[affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>] of %wg
      : memref<4x2x8xi32, strided<[64, 16, 1]>> into !cnm.buffer<2x8xi32 on #wg>
  cnm.free_workgroup %wg : !cnm.workgroup<#wg>
  return
}
