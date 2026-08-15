// Input for the OFFSET check in ../upmem-to-c-dma-alignment.mlir; lit excludes
// Inputs/ from the test suite, so this is never run on its own.
//
// The transfer moves 8 bytes, so the length rule is satisfied and the address
// rule is what is under test. The rows of the buffer are 3 i32 apart, so the
// slice of tasklet %t starts at `%t * 12` bytes -- a whole granule only for
// even %t.

upmem.dpu_program @unaligned_offset() tasklets(8) {
  %a = upmem.static_alloc @a(mram) noinit : memref<8x3xi32, #upmem.mram>
  %t = upmem.tasklet_dim()
  %s = memref.subview %a[%t, 0] [1, 2] [1, 1]
    : memref<8x3xi32, #upmem.mram>
    to memref<2xi32, strided<[1], offset: ?>, #upmem.mram>
  %w = memref.alloca() : memref<2xi32, #upmem.wram>
  upmem.local_transfer %s into %w
    : memref<2xi32, strided<[1], offset: ?>, #upmem.mram>
    to memref<2xi32, #upmem.wram>
  upmem.return
}
