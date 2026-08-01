// Input for the NESTED check in ../upmem-to-c-subview.mlir; lit excludes
// Inputs/ from the test suite, so this is never run on its own.

upmem.dpu_program @nested() tasklets(8) {
  %c0 = arith.constant 0 : index
  %a = upmem.static_alloc @a(mram) noinit : memref<8x8x1x128xi32, #upmem.mram>
  %t = upmem.tasklet_dim()
  %outer = memref.subview %a[%t, 0, 0, 0] [1, 8, 1, 128] [1, 1, 1, 1]
    : memref<8x8x1x128xi32, #upmem.mram>
    to memref<8x1x128xi32, strided<[128, 128, 1], offset: ?>, #upmem.mram>
  %inner = memref.subview %outer[0, 0, %c0] [8, 1, 64] [1, 1, 1]
    : memref<8x1x128xi32, strided<[128, 128, 1], offset: ?>, #upmem.mram>
    to memref<8x1x64xi32, strided<[128, 128, 1], offset: ?>, #upmem.mram>
  %w = memref.alloca() : memref<8x1x64xi32, #upmem.wram>
  upmem.local_transfer %inner into %w
    : memref<8x1x64xi32, strided<[128, 128, 1], offset: ?>, #upmem.mram>
    to memref<8x1x64xi32, #upmem.wram>
  upmem.return
}
