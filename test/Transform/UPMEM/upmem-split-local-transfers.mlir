// RUN: cinm-opt %s --split-input-file --upmem-split-local-transfers | FileCheck %s
// RUN: cinm-opt %s --split-input-file --upmem-split-local-transfers | cinm-translate --allow-unregistered-dialect --mlir-to-upmem-cpp | FileCheck %s --check-prefix=C
// RUN: not cinm-opt %s --split-input-file --upmem-split-local-transfers=max-dma-bytes=12 2>&1 | FileCheck %s --check-prefix=OPTION

// OPTION: max-dma-bytes must be a multiple of 8 between 8 and 2048, not 12

// A tasklet's 8x64 i32 tile of A, 2 KB, read once per K step from the
// pre-tiled layout that keeps it contiguous: four 512-byte reads, each of the
// next 128 elements, at the tile's offset plus its own.

// CHECK-LABEL: upmem.dpu_program @read
//       CHECK:   scf.for %[[K:.*]] =
//       CHECK:     memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]] : memref<16x8x8x64xi32, #upmem.mram> into memref<65536xi32, #upmem.mram>
//       CHECK:     memref.collapse_shape %{{.*}} {{\[}}[0, 1]] : memref<8x64xi32, #upmem.wram> into memref<512xi32, #upmem.wram>
//  CHECK-COUNT-4:     upmem.local_transfer %{{.*}} into %{{.*}} : memref<128xi32, strided<[1], offset: ?>, #upmem.mram> to memref<128xi32, strided<[1]{{.*}}>, #upmem.wram>
//   CHECK-NOT:     upmem.local_transfer

// C-LABEL: void read(void)
//  C-COUNT-4:   mram_read({{.*}}, 512);
//   C-NOT:   mram_read
upmem.dpu_program @read() tasklets(16) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t = upmem.tasklet_dim()
  %buf = upmem.static_alloc @a(mram) noinit : memref<16x8x8x64xi32, #upmem.mram>
  %w = memref.alloca() : memref<8x64xi32, #upmem.wram>
  scf.for %k = %c0 to %c8 step %c1 {
    %v = memref.subview %buf[%t, %k, 0, 0] [1, 1, 8, 64] [1, 1, 1, 1] : memref<16x8x8x64xi32, #upmem.mram> to memref<8x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
    upmem.local_transfer %v into %w : memref<8x64xi32, strided<[64, 1], offset: ?>, #upmem.mram> to memref<8x64xi32, #upmem.wram>
  }
  upmem.return
}

// -----

// A whole 4 KB output buffer written back: eight 512-byte writes, the whole
// buffers addressed from their first element.

// CHECK-LABEL: upmem.dpu_program @write
//  CHECK-COUNT-8:   upmem.local_transfer %{{.*}} into %{{.*}} : memref<128xi32, strided<[1]{{.*}}>, #upmem.wram> to memref<128xi32, strided<[1]{{.*}}>, #upmem.mram>
//   CHECK-NOT:   upmem.local_transfer
upmem.dpu_program @write() tasklets(1) {
  %buf = upmem.static_alloc @c(mram) noinit : memref<1024xi32, #upmem.mram>
  %w = memref.alloca() : memref<1024xi32, #upmem.wram>
  upmem.local_transfer %w into %buf : memref<1024xi32, #upmem.wram> to memref<1024xi32, #upmem.mram>
  upmem.return
}

// -----

// At the cap: left as it is.

// CHECK-LABEL: upmem.dpu_program @small
//       CHECK:   upmem.local_transfer %{{.*}} into %{{.*}} : memref<128xi32, #upmem.mram> to memref<128xi32, #upmem.wram>
//   CHECK-NOT:   upmem.local_transfer
upmem.dpu_program @small() tasklets(1) {
  %buf = upmem.static_alloc @b(mram) noinit : memref<128xi32, #upmem.mram>
  %w = memref.alloca() : memref<128xi32, #upmem.wram>
  upmem.local_transfer %buf into %w : memref<128xi32, #upmem.mram> to memref<128xi32, #upmem.wram>
  upmem.return
}
