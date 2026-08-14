// Input for the NESTED check in ../upmem-to-c-subview.mlir; lit excludes
// Inputs/ from the test suite, so this is never run on its own.
//
// The chain is what is under test, so the transfer itself has to be something
// a DMA could move: the buffer is chunk-major and the innermost view is one
// contiguous run. Otherwise the contiguity check fires first and the nested
// subview is never reached.

upmem.dpu_program @nested() tasklets(8) {
  %c0 = arith.constant 0 : index
  %a = upmem.static_alloc @a(mram) noinit : memref<8x2x8x64xi32, #upmem.mram>
  %t = upmem.tasklet_dim()
  %outer = memref.subview %a[%t, 0, 0, 0] [1, 2, 8, 64] [1, 1, 1, 1]
    : memref<8x2x8x64xi32, #upmem.mram>
    to memref<2x8x64xi32, strided<[512, 64, 1], offset: ?>, #upmem.mram>
  %inner = memref.subview %outer[%c0, 0, 0] [1, 8, 64] [1, 1, 1]
    : memref<2x8x64xi32, strided<[512, 64, 1], offset: ?>, #upmem.mram>
    to memref<8x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
  %w = memref.alloca() : memref<8x64xi32, #upmem.wram>
  upmem.local_transfer %inner into %w
    : memref<8x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
    to memref<8x64xi32, #upmem.wram>
  upmem.return
}
