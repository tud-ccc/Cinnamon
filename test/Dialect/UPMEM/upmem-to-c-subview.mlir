// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s
// RUN: not cinm-translate --mlir-to-upmem-cpp %S/Inputs/nested-subview.mlir 2>&1 | FileCheck %s --check-prefix=NESTED

// A DPU kernel addresses an MRAM buffer as `base + one linear offset`, and
// that offset is the dot product of the subview's offsets with the *source's
// strides*. Using the subview's own sizes instead agrees only when every
// dimension is either fully covered or offset zero, which is why the bug
// survived: the hand-written templates address their MRAM buffers directly and
// never take a subview. Only the generic path does, to slice a tasklet's share
// out of a shared allocation.

// The buffer is laid out chunk-major -- [tasklet][chunk][m][k] rather than
// [tasklet][m][k] -- because a transfer moves one run of memory: slicing a
// k-chunk out of an m-major tile would step over the rest of each row, which
// upmem.local_transfer rejects. That layout is what
// --convert-linalg-to-cnm's leaf-tile split produces.

// CHECK-LABEL: void k(
upmem.dpu_program @k() tasklets(8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %a = upmem.static_alloc @a(mram) noinit : memref<8x2x8x64xi32, #upmem.mram>
  %t = upmem.tasklet_dim()

  // memref<8x2x8x64xi32> has strides [1024, 512, 64, 1], so offsets
  // (%t, %c, 0, 0) give %t*1024 + %c*512 elements. The subscript is applied to
  // a char array, so it has to be those in bytes: %t*4096 + %c*2048.
  // CHECK: for (int32_t [[C:v[0-9]+]] = 0; [[C]] < 2; [[C]] += 1) {
  // CHECK: mram_read(&a[0 + ({{v[0-9]+}} * 4096) + ([[C]] * 2048) + 0]
  scf.for %c = %c0 to %c2 step %c1 {
    %w = memref.alloca() : memref<8x64xi32, #upmem.wram>
    %s = memref.subview %a[%t, %c, 0, 0] [1, 1, 8, 64] [1, 1, 1, 1]
      : memref<8x2x8x64xi32, #upmem.mram>
      to memref<8x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
    upmem.local_transfer %s into %w
      : memref<8x64xi32, strided<[64, 1], offset: ?>, #upmem.mram>
      to memref<8x64xi32, #upmem.wram>
  }
  upmem.return
}

// A chain of subviews cannot be expressed as one linear offset, so it is
// refused rather than silently resolved against the wrong base.
// --fold-memref-alias-ops composes chains before translation.
// NESTED: nested memref.subview is not supported
