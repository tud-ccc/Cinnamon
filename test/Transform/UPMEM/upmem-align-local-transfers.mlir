// RUN: cinm-opt %s --split-input-file --upmem-align-local-transfers | FileCheck %s

// One i32 per trip, at a 4-byte stride: the read takes the granule holding
// it and copies the element out of it.

// CHECK-LABEL: upmem.dpu_program @read
//       CHECK:   %[[FLAT:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1]] : memref<4x2xi32, #upmem.mram> into memref<8xi32, #upmem.mram>
//       CHECK:   %[[Q:.*]] = arith.divui %{{.*}}, %{{.*}} : index
//       CHECK:   %[[G:.*]] = arith.muli %[[Q]], %{{.*}} : index
//       CHECK:   %[[IN:.*]] = arith.subi %{{.*}}, %[[G]] : index
//       CHECK:   %[[GV:.*]] = memref.subview %[[FLAT]][%[[G]]] [2] [1]
//       CHECK:   %[[B:.*]] = memref.alloca() : memref<2xi32, #upmem.wram>
//       CHECK:   %[[DV:.*]] = memref.subview %[[B]][%[[IN]]] [1] [1]
//       CHECK:   upmem.local_transfer %[[GV]] into %[[B]]
//       CHECK:   upmem.local_transfer %[[DV]] into %{{.*}} : memref<1xi32, {{.*}}#upmem.wram> to memref<1x1xi32, #upmem.wram>
//   CHECK-NOT:   upmem.local_transfer
upmem.dpu_program @read() tasklets(4) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t = upmem.tasklet_dim()
  %buf = upmem.static_alloc @in(mram) noinit : memref<4x2xi32, #upmem.mram>
  %w = memref.alloca() : memref<1x1xi32, #upmem.wram>
  scf.for %i = %c0 to %c2 step %c1 {
    %v = memref.subview %buf[%t, %i] [1, 1] [1, 1] : memref<4x2xi32, #upmem.mram> to memref<1x1xi32, strided<[2, 1], offset: ?>, #upmem.mram>
    upmem.local_transfer %v into %w : memref<1x1xi32, strided<[2, 1], offset: ?>, #upmem.mram> to memref<1x1xi32, #upmem.wram>
  }
  upmem.return
}

// -----

// One i8 into the calling tasklet's 8-byte slice: the write reads the
// granule, copies the byte in and writes the granule back.

// CHECK-LABEL: upmem.dpu_program @owned_write
//       CHECK:   %[[B:.*]] = memref.alloca() : memref<8xi8, #upmem.wram>
//       CHECK:   upmem.local_transfer %[[GV:.*]] into %[[B]]
//       CHECK:   upmem.local_transfer %{{.*}} into %[[DV:.*]] : memref<1x1xi8, #upmem.wram> to memref<1xi8, {{.*}}#upmem.wram>
//       CHECK:   upmem.local_transfer %[[B]] into %[[GV]]
upmem.dpu_program @owned_write() tasklets(4) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t = upmem.tasklet_dim()
  %buf = upmem.static_alloc @out(mram) noinit : memref<4x8xi8, #upmem.mram>
  %w = memref.alloca() : memref<1x1xi8, #upmem.wram>
  scf.for %i = %c0 to %c8 step %c1 {
    %v = memref.subview %buf[%t, %i] [1, 1] [1, 1] : memref<4x8xi8, #upmem.mram> to memref<1x1xi8, strided<[8, 1], offset: ?>, #upmem.mram>
    upmem.local_transfer %w into %v : memref<1x1xi8, #upmem.wram> to memref<1x1xi8, strided<[8, 1], offset: ?>, #upmem.mram>
  }
  upmem.return
}

// -----

// A tasklet's slice is 4 bytes, so the granule a write would rewrite holds
// another tasklet's bytes: left for the translator to reject.

// CHECK-LABEL: upmem.dpu_program @shared_write
//   CHECK-NOT:   memref.collapse_shape
//       CHECK:   upmem.local_transfer %{{.*}} into %{{.*}} : memref<1x1xi8, #upmem.wram> to memref<1x1xi8, strided<[4, 1], offset: ?>, #upmem.mram>
upmem.dpu_program @shared_write() tasklets(4) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %t = upmem.tasklet_dim()
  %buf = upmem.static_alloc @out(mram) noinit : memref<4x4xi8, #upmem.mram>
  %w = memref.alloca() : memref<1x1xi8, #upmem.wram>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.subview %buf[%t, %i] [1, 1] [1, 1] : memref<4x4xi8, #upmem.mram> to memref<1x1xi8, strided<[4, 1], offset: ?>, #upmem.mram>
    upmem.local_transfer %w into %v : memref<1x1xi8, #upmem.wram> to memref<1x1xi8, strided<[4, 1], offset: ?>, #upmem.mram>
  }
  upmem.return
}

// -----

// Whole granules at granule offsets need nothing.

// CHECK-LABEL: upmem.dpu_program @aligned
//   CHECK-NOT:   memref.collapse_shape
//       CHECK:   upmem.local_transfer %{{.*}} into %{{.*}} : memref<1x2xi32, strided<[2, 1], offset: ?>, #upmem.mram> to memref<1x2xi32, #upmem.wram>
upmem.dpu_program @aligned() tasklets(4) {
  %t = upmem.tasklet_dim()
  %buf = upmem.static_alloc @in(mram) noinit : memref<4x2xi32, #upmem.mram>
  %w = memref.alloca() : memref<1x2xi32, #upmem.wram>
  %v = memref.subview %buf[%t, 0] [1, 2] [1, 1] : memref<4x2xi32, #upmem.mram> to memref<1x2xi32, strided<[2, 1], offset: ?>, #upmem.mram>
  upmem.local_transfer %v into %w : memref<1x2xi32, strided<[2, 1], offset: ?>, #upmem.mram> to memref<1x2xi32, #upmem.wram>
  upmem.return
}
