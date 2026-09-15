// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s
// RUN: not cinm-translate --mlir-to-upmem-cpp %S/Inputs/dma-short-length.mlir 2>&1 | FileCheck %s --check-prefix=LENGTH
// RUN: not cinm-translate --mlir-to-upmem-cpp %S/Inputs/dma-unaligned-offset.mlir 2>&1 | FileCheck %s --check-prefix=OFFSET

// An MRAM DMA moves whole 8-byte granules between 8-byte aligned addresses.
// The translator used to round a short length up to the granule and emit the
// address verbatim, which produces a kernel that compiles, runs, and computes
// the wrong answer: the hardware drops the low bits of a misaligned MRAM
// address, so the tasklets of a DPU overwrite each other's output slices while
// the rounded-up length spills into the neighbouring one. Both are refused.

// A tile that is a whole number of granules still translates: the slice of
// tasklet %t starts at `%t * 16` bytes and is 16 bytes long.
// CHECK-LABEL: void aligned(
// CHECK: mram_read(&a[0 + ({{v[0-9]+}} * 16) + 0], &((char*) {{v[0-9]+}})[0 + 0], 16)
upmem.dpu_program @aligned() tasklets(8) {
  %a = upmem.static_alloc @a(mram) noinit : memref<8x4xi32, #upmem.mram>
  %t = upmem.tasklet_dim()
  %s = memref.subview %a[%t, 0] [1, 4] [1, 1]
    : memref<8x4xi32, #upmem.mram>
    to memref<4xi32, strided<[1], offset: ?>, #upmem.mram>
  %w = memref.alloca() : memref<4xi32, #upmem.wram>
  upmem.local_transfer %s into %w
    : memref<4xi32, strided<[1], offset: ?>, #upmem.mram>
    to memref<4xi32, #upmem.wram>
  upmem.return
}

// A short *read* is allowed to round up, and the LENGTH case above shows the
// matching write is not. Only the write does damage: the bytes it rounds up
// over belong to the next tile. The read fetches a few nobody looks at, into
// a destination declared padded to a whole granule. Refusing it would strand
// every broadcast scalar -- geva's coefficients are four bytes at offset 0.
// CHECK-LABEL: void short_read(
// CHECK: mram_read(&a[0 + 0], &((char*) {{v[0-9]+}})[0 + 0], 8)
upmem.dpu_program @short_read() tasklets(8) {
  %a = upmem.static_alloc @a(mram) noinit : memref<1xi32, #upmem.mram>
  %w = memref.alloca() : memref<1xi32, #upmem.wram>
  upmem.local_transfer %a into %w
    : memref<1xi32, #upmem.mram> to memref<1xi32, #upmem.wram>
  upmem.return
}

// LENGTH: cannot emit a DMA of 4 bytes
// OFFSET: source address is not 8-byte aligned
