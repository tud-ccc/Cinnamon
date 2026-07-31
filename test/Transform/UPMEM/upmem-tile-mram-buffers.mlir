// RUN: cinm-opt %s --split-input-file --upmem-tile-mram-buffers --canonicalize | FileCheck %s

// A cnm.launch body may operate on buffers in a level the compute elements
// cannot address (MRAM). This pass tiles the body down to a leaf-sized tile and
// stages that tile through the leaf level (WRAM) with cnm.local_transfers.
//
// It keys off the memref's memory space, not the op, so it applies to any
// linalg op on buffers.

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// CHECK-LABEL: @gemv
func.func @gemv() {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %a = cnm.alloc() for %wg : !cnm.buffer<64x512xi32 on #acc, #upmem.mram>
  %x = cnm.alloc() for %wg : !cnm.buffer<512xi32 on #acc, #upmem.mram>
  %y = cnm.alloc() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK: %[[SA:.*]] = memref.subview %{{.*}}[%[[I]], %[[K]]] [16, 128] [1, 1] : memref<64x512xi32, #upmem.mram>
  // CHECK: %[[SX:.*]] = memref.subview %{{.*}}[%[[K]]] [128] [1] : memref<512xi32, #upmem.mram>
  // CHECK: %[[SY:.*]] = memref.subview %{{.*}}[%[[I]]] [16] [1] : memref<64xi32, #upmem.mram>

  // Statically shaped WRAM buffers, not the flat i8 buffer + memref.view the
  // default promotion allocator would produce.
  // CHECK: %[[WA:.*]] = memref.alloc() : memref<16x128xi32, #upmem.wram>
  // CHECK: %[[WX:.*]] = memref.alloc() : memref<128xi32, #upmem.wram>
  // CHECK: %[[WY:.*]] = memref.alloc() : memref<16xi32, #upmem.wram>

  // CHECK: cnm.local_transfer %[[SA]] into %[[WA]] : memref<16x128xi32, strided<[512, 1], offset: ?>, #upmem.mram> to memref<16x128xi32, #upmem.wram>
  // CHECK: cnm.local_transfer %[[SX]] into %[[WX]]
  // The output is read as well as written: the contract accumulates into it.
  // CHECK: cnm.local_transfer %[[SY]] into %[[WY]]
  // CHECK: linalg.contract {{.*}} ins(%[[WA]], %[[WX]] : memref<16x128xi32, #upmem.wram>, memref<128xi32, #upmem.wram>) outs(%[[WY]] : memref<16xi32, #upmem.wram>)
  // CHECK: cnm.local_transfer %[[WY]] into %[[SY]] : memref<16xi32, #upmem.wram> to memref<16xi32, strided<[1], offset: ?>, #upmem.mram>
  // CHECK: memref.dealloc %[[WA]]

  // The tile sizes have been consumed.
  // CHECK-NOT: upmem.leaf_tile_sizes
  cnm.launch %wg ins(%A = %a : <64x512xi32, #upmem.mram>, %X = %x : <512xi32, #upmem.mram>)
                 outs(%Y = %y : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    linalg.contract indexing_maps = [#m, #v, #r]
      {upmem.leaf_tile_sizes = array<i64: 16, 128>}
      ins(%A, %X : memref<64x512xi32, #upmem.mram>, memref<512xi32, #upmem.mram>)
      outs(%Y : memref<64xi32, #upmem.mram>)
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

// -----

#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// Not contract-specific: a linalg.reduce gets the same treatment.
// CHECK-LABEL: @reduce
func.func @reduce() {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %a = cnm.alloc() for %wg : !cnm.buffer<64x512xi32 on #acc, #upmem.mram>
  %o = cnm.alloc() for %wg : !cnm.buffer<64xi32 on #acc, #upmem.mram>
  // CHECK: memref.alloc() : memref<16x128xi32, #upmem.wram>
  // CHECK: memref.alloc() : memref<16xi32, #upmem.wram>
  // CHECK: cnm.local_transfer
  // CHECK: linalg.reduce ins(%{{.*}} : memref<16x128xi32, #upmem.wram>) outs(%{{.*}} : memref<16xi32, #upmem.wram>)
  cnm.launch %wg ins(%A = %a : <64x512xi32, #upmem.mram>)
                 outs(%O = %o : <64xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    linalg.reduce ins(%A : memref<64x512xi32, #upmem.mram>)
      outs(%O : memref<64xi32, #upmem.mram>) dimensions = [1]
      {upmem.leaf_tile_sizes = array<i64: 16, 128>}
      (%in: i32, %init: i32) {
        %s = arith.addi %in, %init : i32
        linalg.yield %s : i32
      }
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// With no tile sizes the buffer already fits the leaf level, so the op is
// staged whole and no loop nest appears. This is the shape a single-level
// tiling configuration produces.
// CHECK-LABEL: @no_tiling
func.func @no_tiling() {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %a = cnm.alloc() for %wg : !cnm.buffer<16x128xi32 on #acc, #upmem.mram>
  %x = cnm.alloc() for %wg : !cnm.buffer<128xi32 on #acc, #upmem.mram>
  %y = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #acc, #upmem.mram>
  // CHECK-NOT: scf.for
  // CHECK: memref.alloc() : memref<16x128xi32, #upmem.wram>
  // CHECK: cnm.local_transfer %{{.*}} : memref<16x128xi32, #upmem.mram> to memref<16x128xi32, #upmem.wram>
  // CHECK: linalg.contract {{.*}} : memref<16x128xi32, #upmem.wram>, memref<128xi32, #upmem.wram>) outs(%{{.*}} : memref<16xi32, #upmem.wram>)
  cnm.launch %wg ins(%A = %a : <16x128xi32, #upmem.mram>, %X = %x : <128xi32, #upmem.mram>)
                 outs(%Y = %y : <16xi32, #upmem.mram>) on !cnm.workgroup<#acc> {
    linalg.contract indexing_maps = [#m, #v, #r]
      ins(%A, %X : memref<16x128xi32, #upmem.mram>, memref<128xi32, #upmem.mram>)
      outs(%Y : memref<16xi32, #upmem.mram>)
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}

// -----

#m = affine_map<(m, k) -> (m, k)>
#v = affine_map<(m, k) -> (k)>
#r = affine_map<(m, k) -> (m)>
#pf = #upmem.platform<type=v1A, dimensions = 4x16>
#acc = #upmem.array<1x16x1, #pf>

// A body already entirely in the leaf level is left alone.
// CHECK-LABEL: @already_leaf
func.func @already_leaf() {
  %wg = cnm.workgroup : !cnm.workgroup<#acc>
  %a = cnm.alloc() for %wg : !cnm.buffer<16x128xi32 on #acc, #upmem.wram>
  %x = cnm.alloc() for %wg : !cnm.buffer<128xi32 on #acc, #upmem.wram>
  %y = cnm.alloc() for %wg : !cnm.buffer<16xi32 on #acc, #upmem.wram>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: cnm.local_transfer
  // CHECK: linalg.contract
  cnm.launch %wg ins(%A = %a : <16x128xi32, #upmem.wram>, %X = %x : <128xi32, #upmem.wram>)
                 outs(%Y = %y : <16xi32, #upmem.wram>) on !cnm.workgroup<#acc> {
    linalg.contract indexing_maps = [#m, #v, #r]
      ins(%A, %X : memref<16x128xi32, #upmem.wram>, memref<128xi32, #upmem.wram>)
      outs(%Y : memref<16xi32, #upmem.wram>)
  }
  cnm.free_workgroup %wg : !cnm.workgroup<#acc>
  return
}
