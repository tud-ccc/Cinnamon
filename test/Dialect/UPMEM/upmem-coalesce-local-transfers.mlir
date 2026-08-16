// RUN: cinm-opt %s --upmem-coalesce-local-transfers --split-input-file | FileCheck %s

// A leaf tile of one i32 is 4 bytes, half of the 8-byte DMA granule the level
// declares. The far-level buffer is indexed per tile, so the loop's transfers
// start 4 bytes apart and every other one is misaligned. Two tiles per
// transfer puts them all back on a boundary.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: @coalesces_a_fractional_tile
func.func @coalesces_a_fractional_tile(%wg: !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>,
                                       %buf: !cnm.buffer<8x1x1xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>) {
  // CHECK: cnm.launch
  cnm.launch %wg outs(%arg = %buf : <8x1x1xi32, #upmem.mram>) on !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>> {
    // The loop counts strips of two tiles, and stages both in one transfer.
    // The offset reads as `strip * 2`, so it is a whole number of granules in
    // the expression itself -- the emitter's alignment check is syntactic.
    // CHECK:      affine.for %[[STRIP:.*]] = 0 to 4 {
    // CHECK-NEXT:   %[[BIG:.*]] = memref.alloca() : memref<2x1x1xi32, #upmem.wram>
    // CHECK-NEXT:   %[[START:.*]] = affine.apply #{{.*}}(%[[STRIP]])
    // CHECK-NEXT:   %[[TILE:.*]] = memref.subview %{{.*}}[%[[START]], 0, 0] [2, 1, 1]
    // CHECK-NEXT:   cnm.local_transfer %[[TILE]] into %[[BIG]]
    // CHECK-NEXT:   affine.for %[[I:.*]] = #{{.*}}(%[[STRIP]]) to #{{.*}}(%[[STRIP]]) {
    // CHECK-NEXT:     %[[WITHIN:.*]] = affine.apply #{{.*}}(%[[I]], %[[STRIP]])
    // CHECK-NEXT:     memref.subview %[[BIG]][%[[WITHIN]], 0, 0] [1, 1, 1]
    // CHECK:        }
    // CHECK-NEXT:   cnm.local_transfer %[[BIG]] into %[[TILE]]
    // CHECK-NEXT: }
    affine.for %i = 0 to 8 {
      %tile = memref.subview %arg[%i, 0, 0] [1, 1, 1] [1, 1, 1] : memref<8x1x1xi32, #upmem.mram> to memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram>
      %staged = memref.alloca() : memref<1x1x1xi32, #upmem.wram>
      cnm.local_transfer %tile into %staged : memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram> to memref<1x1x1xi32, #upmem.wram>
      %c0 = arith.constant 0 : index
      %v = memref.load %staged[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
      %doubled = arith.addi %v, %v : i32
      memref.store %doubled, %staged[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
      cnm.local_transfer %staged into %tile : memref<1x1x1xi32, #upmem.wram> to memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram>
    }
  }
  return
}

// -----

// A tile that already fills whole granules is left alone: 64 i32 is 256 bytes,
// so every transfer already starts on a boundary and splitting the loop would
// only cost WRAM.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: @leaves_an_aligned_tile_alone
func.func @leaves_an_aligned_tile_alone(%wg: !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>,
                                        %buf: !cnm.buffer<8x64xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>) {
  cnm.launch %wg outs(%arg = %buf : <8x64xi32, #upmem.mram>) on !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>> {
    // CHECK:      affine.for %{{.*}} = 0 to 8 {
    // CHECK-NEXT:   memref.subview
    // CHECK-NEXT:   memref.alloca() : memref<1x64xi32, #upmem.wram>
    // CHECK-NEXT:   cnm.local_transfer
    // CHECK-NOT:  step 2
    affine.for %i = 0 to 8 {
      %tile = memref.subview %arg[%i, 0] [1, 64] [1, 1] : memref<8x64xi32, #upmem.mram> to memref<1x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
      %staged = memref.alloca() : memref<1x64xi32, #upmem.wram>
      cnm.local_transfer %tile into %staged : memref<1x64xi32, strided<[64, 1], offset: ?>, #upmem.mram> to memref<1x64xi32, #upmem.wram>
      %c0 = arith.constant 0 : index
      %v = memref.load %staged[%c0, %c0] : memref<1x64xi32, #upmem.wram>
      %doubled = arith.addi %v, %v : i32
      memref.store %doubled, %staged[%c0, %c0] : memref<1x64xi32, #upmem.wram>
      cnm.local_transfer %staged into %tile : memref<1x64xi32, #upmem.wram> to memref<1x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
    }
  }
  return
}

// -----

// A tile that does not depend on the loop is hoisted out of it: the reduction
// updates the same bytes on every trip, so only the final value has to travel.
// This is the heuristic upmem-tile-mram-buffers used to apply while tiling.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: @hoists_an_invariant_tile
func.func @hoists_an_invariant_tile(%wg: !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>,
                                    %buf: !cnm.buffer<8x2xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>) {
  cnm.launch %wg outs(%arg = %buf : <8x2xi32, #upmem.mram>) on !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>> {
    %c0 = arith.constant 0 : index
    %tile = memref.subview %arg[0, 0] [1, 2] [1, 1] : memref<8x2xi32, #upmem.mram> to memref<1x2xi32, strided<[2, 1]>, #upmem.mram>
    // One read in, then the loop, then one write back.
    // CHECK:      cnm.local_transfer %{{.*}} into %[[STAGED:.*]] :
    // CHECK-NEXT: affine.for
    // CHECK-NOT:    cnm.local_transfer
    // CHECK:      }
    // CHECK-NEXT: cnm.local_transfer %[[STAGED]] into
    affine.for %i = 0 to 8 {
      %staged = memref.alloca() : memref<1x2xi32, #upmem.wram>
      cnm.local_transfer %tile into %staged : memref<1x2xi32, strided<[2, 1]>, #upmem.mram> to memref<1x2xi32, #upmem.wram>
      %v = memref.load %staged[%c0, %c0] : memref<1x2xi32, #upmem.wram>
      %doubled = arith.addi %v, %v : i32
      memref.store %doubled, %staged[%c0, %c0] : memref<1x2xi32, #upmem.wram>
      cnm.local_transfer %staged into %tile : memref<1x2xi32, #upmem.wram> to memref<1x2xi32, strided<[2, 1]>, #upmem.mram>
    }
  }
  return
}
