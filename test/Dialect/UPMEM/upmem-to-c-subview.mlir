// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s
// RUN: not cinm-translate --mlir-to-upmem-cpp %S/Inputs/nested-subview.mlir 2>&1 | FileCheck %s --check-prefix=NESTED

// A DPU kernel addresses an MRAM buffer as `base + one linear offset`, and
// that offset is the dot product of the subview's offsets with the *source's
// strides*. Using the subview's own sizes instead agrees only when every
// dimension is either fully covered or offset zero, which is why the bug
// survived: the hand-written templates address their MRAM buffers directly and
// never take a subview. Only the generic path does, to slice a tasklet's share
// out of a shared allocation.

// CHECK-LABEL: void k(
upmem.dpu_program @k() tasklets(8) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %a = upmem.static_alloc @a(mram) noinit : memref<8x8x1x128xi32, #upmem.mram>
  %t = upmem.tasklet_dim()

  // memref<8x8x1x128xi32> has strides [1024, 128, 128, 1], so offsets
  // (%t, 0, 0, %k) give %t*1024 + %k. The old formula multiplied by the
  // subview's sizes in reverse instead and produced %k*64 + %t.
  // CHECK: for (int32_t [[K:v[0-9]+]] = 0; [[K]] < 128; [[K]] += 64) {
  // CHECK: mram_read(&a[0 + ({{v[0-9]+}} * 1024) + ([[K]] * 1) + 0]
  scf.for %k = %c0 to %c128 step %c64 {
    %w = memref.alloca() : memref<8x1x64xi32, #upmem.wram>
    %s = memref.subview %a[%t, 0, 0, %k] [1, 8, 1, 64] [1, 1, 1, 1]
      : memref<8x8x1x128xi32, #upmem.mram>
      to memref<8x1x64xi32, strided<[128, 128, 1], offset: ?>, #upmem.mram>
    upmem.local_transfer %s into %w
      : memref<8x1x64xi32, strided<[128, 128, 1], offset: ?>, #upmem.mram>
      to memref<8x1x64xi32, #upmem.wram>
  }
  upmem.return
}

// A chain of subviews cannot be expressed as one linear offset, so it is
// refused rather than silently resolved against the wrong base.
// --fold-memref-alias-ops composes chains before translation.
// NESTED: nested memref.subview is not supported
