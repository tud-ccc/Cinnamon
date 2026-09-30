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
    // CHECK-NEXT:   %[[START:.*]] = affine.apply #{{.*}}(%[[STRIP]])
    // CHECK-NEXT:   %[[BIG:.*]] = memref.alloca() : memref<2x1x1xi32, #upmem.wram>
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

// Every fractional staging of a loop is widened by the same split. Splitting
// for one of them and leaving the other behind would strand it: the walk the
// split leaves has non-constant bounds, so a later sweep can no longer widen
// it, and its transfers would stay mid-granule.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: @coalesces_every_staging_of_a_loop
func.func @coalesces_every_staging_of_a_loop(%wg: !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>,
                                             %in: !cnm.buffer<8x1x1xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>,
                                             %out: !cnm.buffer<8x1x1xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>) {
  cnm.launch %wg ins(%argIn = %in : <8x1x1xi32, #upmem.mram>) outs(%argOut = %out : <8x1x1xi32, #upmem.mram>) on !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>> {
    // Both strips start at the same `strip * 2`, and both buffers hold two
    // tiles.
    // CHECK:      affine.for %[[STRIP:.*]] = 0 to 4 {
    // CHECK-NEXT:   %[[START:.*]] = affine.apply #{{.*}}(%[[STRIP]])
    // CHECK-NEXT:   memref.alloca() : memref<2x1x1xi32, #upmem.wram>
    // CHECK-NEXT:   memref.subview %{{.*}}[%[[START]], 0, 0] [2, 1, 1]
    // CHECK-NEXT:   cnm.local_transfer
    // CHECK-NEXT:   %[[BIGOUT:.*]] = memref.alloca() : memref<2x1x1xi32, #upmem.wram>
    // CHECK-NEXT:   %[[TILEOUT:.*]] = memref.subview %{{.*}}[%[[START]], 0, 0] [2, 1, 1]
    // CHECK-NEXT:   cnm.local_transfer
    // CHECK-NEXT:   affine.for
    // CHECK-NOT:      cnm.local_transfer
    // CHECK:        }
    // CHECK-NEXT:   cnm.local_transfer %[[BIGOUT]] into %[[TILEOUT]]
    affine.for %i = 0 to 8 {
      %inTile = memref.subview %argIn[%i, 0, 0] [1, 1, 1] [1, 1, 1] : memref<8x1x1xi32, #upmem.mram> to memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram>
      %stagedIn = memref.alloca() : memref<1x1x1xi32, #upmem.wram>
      cnm.local_transfer %inTile into %stagedIn : memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram> to memref<1x1x1xi32, #upmem.wram>
      %outTile = memref.subview %argOut[%i, 0, 0] [1, 1, 1] [1, 1, 1] : memref<8x1x1xi32, #upmem.mram> to memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram>
      %stagedOut = memref.alloca() : memref<1x1x1xi32, #upmem.wram>
      cnm.local_transfer %outTile into %stagedOut : memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram> to memref<1x1x1xi32, #upmem.wram>
      %c0 = arith.constant 0 : index
      %v = memref.load %stagedIn[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
      %acc = memref.load %stagedOut[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
      %sum = arith.addi %v, %acc : i32
      memref.store %sum, %stagedOut[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
      cnm.local_transfer %stagedOut into %outTile : memref<1x1x1xi32, #upmem.wram> to memref<1x1x1xi32, strided<[1, 1, 1], offset: ?>, #upmem.mram>
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

// -----

// A scalar operand staged in the innermost loop that uses it, as the tiling
// places a scalar fused into the kernel: the same bytes are read on every
// trip, so the staging leaves both loops.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: @hoists_an_invariant_read
func.func @hoists_an_invariant_read(%wg: !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>,
                                    %s: !cnm.buffer<i32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>,
                                    %out: !cnm.buffer<4x16xi32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>) {
  cnm.launch %wg ins(%argS = %s : <i32, #upmem.mram>) outs(%argOut = %out : <4x16xi32, #upmem.mram>) on !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>> {
    // CHECK:      %[[BUF:.*]] = memref.alloca() : memref<i32, #upmem.wram>
    // CHECK-NEXT: cnm.local_transfer %{{.*}} into %[[BUF]]
    // CHECK-NEXT: affine.for
    // CHECK-NOT:    cnm.local_transfer
    // CHECK:        memref.load %[[BUF]][]
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 16 {
        %staged = memref.alloca() : memref<i32, #upmem.wram>
        cnm.local_transfer %argS into %staged : memref<i32, #upmem.mram> to memref<i32, #upmem.wram>
        %v = memref.load %staged[] : memref<i32, #upmem.wram>
        memref.store %v, %argOut[%i, %j] : memref<4x16xi32, #upmem.mram>
      }
    }
  }
  return
}

// -----

// The loop also writes the staged source: each trip may read different bytes,
// so the read stays.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// CHECK-LABEL: @keeps_a_read_of_a_written_source
func.func @keeps_a_read_of_a_written_source(%wg: !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>>,
                                            %s: !cnm.buffer<i32 on #upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>, #upmem.mram>) {
  cnm.launch %wg outs(%argS = %s : <i32, #upmem.mram>) on !cnm.workgroup<#upmem.array<2048x8, <type = v1A, dpus = 2048, tasklets = 24>>> {
    // CHECK:      affine.for
    // CHECK-NEXT:   memref.alloca
    // CHECK-NEXT:   cnm.local_transfer
    affine.for %i = 0 to 4 {
      %staged = memref.alloca() : memref<i32, #upmem.wram>
      cnm.local_transfer %argS into %staged : memref<i32, #upmem.mram> to memref<i32, #upmem.wram>
      %v = memref.load %staged[] : memref<i32, #upmem.wram>
      %w = arith.addi %v, %v : i32
      memref.store %w, %argS[] : memref<i32, #upmem.mram>
    }
  }
  return
}
