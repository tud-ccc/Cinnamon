// Input for the LENGTH check in ../upmem-to-c-dma-alignment.mlir; lit excludes
// Inputs/ from the test suite, so this is never run on its own.
//
// This is the shape --convert-cnm-to-upmem produces for a launch whose output
// tile is a single i32 per tasklet: the MRAM buffer packs the tasklet slices
// four bytes apart, so neither the length nor the address is a whole granule.
// The length is reported first, as it is the one the tile size fixes.

upmem.dpu_program @short_length() tasklets(8) {
  %a = upmem.static_alloc @a(mram) noinit : memref<8x1xi32, #upmem.mram>
  %t = upmem.tasklet_dim()
  %s = memref.subview %a[%t, 0] [1, 1] [1, 1]
    : memref<8x1xi32, #upmem.mram>
    to memref<1xi32, strided<[1], offset: ?>, #upmem.mram>
  %w = memref.alloca() : memref<1xi32, #upmem.wram>
  upmem.local_transfer %w into %s
    : memref<1xi32, #upmem.wram>
    to memref<1xi32, strided<[1], offset: ?>, #upmem.mram>
  upmem.return
}
