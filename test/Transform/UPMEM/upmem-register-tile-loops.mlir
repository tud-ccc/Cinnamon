// RUN: cinm-opt %s --split-input-file --upmem-register-tile-loops | FileCheck %s
// RUN: cinm-opt %s --split-input-file --upmem-register-tile-loops=register-budget=5 | FileCheck %s --check-prefix=BUDGET5

// The programs sit at the top level, without a module around them: a pass
// anchored on `upmem.dpu_program` and given on the command line only visits
// the top-level module's direct children. The pipeline nests it explicitly.

// The gemv nest linalg lowers to: y[j] += A[j][k] * x[k]. Per copy of j the
// inner body loads A[j][k] and y[j]; x[k] is shared. With a budget of 12,
// f * 2 + 1 <= 12 gives f = 5, and the largest divisor of 16 below that is
// 4. The reduction loop of 512 is unrolled by 16.
//
// CHECK-LABEL: upmem.dpu_program @gemv
//       CHECK:   affine.for %[[J:.*]] = 0 to 16 step 4 {
//  CHECK-NEXT:     affine.for %[[K:.*]] = 0 to 512 step 16 {
// The body holds 4 copies of 16 elements each.
// CHECK-COUNT-64: arith.muli
//   CHECK-NOT:       arith.muli
//       CHECK:     }
//  CHECK-NEXT:   }
//
// A budget of 5 leaves f = 2.
// BUDGET5-LABEL: upmem.dpu_program @gemv
//       BUDGET5:   affine.for %{{.*}} = 0 to 16 step 2 {
//  BUDGET5-NEXT:     affine.for %{{.*}} = 0 to 512 step 16 {
// BUDGET5-COUNT-32: arith.muli
//   BUDGET5-NOT:       arith.muli
//       BUDGET5:     }
  upmem.dpu_program @gemv() tasklets(1) {
    %A = memref.alloca() : memref<16x512xi32, #upmem.wram>
    %x = memref.alloca() : memref<512xi32, #upmem.wram>
    %y = memref.alloca() : memref<16xi32, #upmem.wram>
    affine.for %j = 0 to 16 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%j, %k] : memref<16x512xi32, #upmem.wram>
        %xv = affine.load %x[%k] : memref<512xi32, #upmem.wram>
        %acc = affine.load %y[%j] : memref<16xi32, #upmem.wram>
        %p = arith.muli %a, %xv : i32
        %s = arith.addi %acc, %p : i32
        affine.store %s, %y[%j] : memref<16xi32, #upmem.wram>
      }
    }
    upmem.return
  }

// -----

// A trip count of 2 on the reuse loop takes the whole loop into the jam, and
// the promoted single iteration leaves the reduction loop directly under the
// row loop. 128 is unrolled by 16. The row loop, whose body stages tiles, is
// not jammed: its transfers must not be duplicated.
//
// CHECK-LABEL: upmem.dpu_program @two_rows
//       CHECK:   affine.for %{{.*}} = 0 to 25 {
//       CHECK:     upmem.local_transfer
//       CHECK:     upmem.local_transfer
//   CHECK-NOT:     affine.for %{{.*}} = 0 to 2
//       CHECK:     affine.for %{{.*}} = 0 to 128 step 16 {
// CHECK-COUNT-32: arith.muli
//   CHECK-NOT:       arith.muli
//       CHECK:     }
//   CHECK-NOT:     affine.for
//       CHECK:   }
  upmem.dpu_program @two_rows() tasklets(16) {
    %W = upmem.static_alloc @buf(mram) noinit : memref<25x2x128xi8, #upmem.mram>
    %X = upmem.static_alloc @buf_0(mram) noinit : memref<25x128xi8, #upmem.mram>
    %w = memref.alloca() : memref<2x128xi8, #upmem.wram>
    %x = memref.alloca() : memref<128xi8, #upmem.wram>
    %y = memref.alloca() : memref<2xi32, #upmem.wram>
    affine.for %r = 0 to 25 {
      %wt = memref.subview %W[%r, 0, 0] [1, 2, 128] [1, 1, 1] : memref<25x2x128xi8, #upmem.mram> to memref<2x128xi8, strided<[128, 1], offset: ?>, #upmem.mram>
      %xt = memref.subview %X[%r, 0] [1, 128] [1, 1] : memref<25x128xi8, #upmem.mram> to memref<128xi8, strided<[1], offset: ?>, #upmem.mram>
      upmem.local_transfer %wt into %w : memref<2x128xi8, strided<[128, 1], offset: ?>, #upmem.mram> to memref<2x128xi8, #upmem.wram>
      upmem.local_transfer %xt into %x : memref<128xi8, strided<[1], offset: ?>, #upmem.mram> to memref<128xi8, #upmem.wram>
      affine.for %j = 0 to 2 {
        affine.for %k = 0 to 128 {
          %a = affine.load %w[%j, %k] : memref<2x128xi8, #upmem.wram>
          %xv = affine.load %x[%k] : memref<128xi8, #upmem.wram>
          %acc = affine.load %y[%j] : memref<2xi32, #upmem.wram>
          %ae = arith.extsi %a : i8 to i32
          %xe = arith.extsi %xv : i8 to i32
          %p = arith.muli %ae, %xe : i32
          %s = arith.addi %acc, %p : i32
          affine.store %s, %y[%j] : memref<2xi32, #upmem.wram>
        }
      }
    }
    upmem.return
  }

// -----

// Nothing is shared between copies of j when every operand is indexed by j,
// so the reuse loop is left alone and only the inner loop is unrolled. A trip
// count of 8 is unrolled fully.
//
// CHECK-LABEL: upmem.dpu_program @no_reuse
//       CHECK:   affine.for %{{.*}} = 0 to 16 {
//   CHECK-NOT:     affine.for
// CHECK-COUNT-8:   arith.addi
//   CHECK-NOT:     arith.addi
//       CHECK:   }
  upmem.dpu_program @no_reuse() tasklets(1) {
    %A = memref.alloca() : memref<16x8xi32, #upmem.wram>
    %y = memref.alloca() : memref<16xi32, #upmem.wram>
    affine.for %j = 0 to 16 {
      affine.for %k = 0 to 8 {
        %a = affine.load %A[%j, %k] : memref<16x8xi32, #upmem.wram>
        %acc = affine.load %y[%j] : memref<16xi32, #upmem.wram>
        %s = arith.addi %acc, %a : i32
        affine.store %s, %y[%j] : memref<16xi32, #upmem.wram>
      }
    }
    upmem.return
  }

// -----

// A store whose address does not move with the reuse loop -- a reduction
// over j into z[k] -- makes the copies write the same location, so no jam.
//
// CHECK-LABEL: upmem.dpu_program @reduction_over_parent
//       CHECK:   affine.for %{{.*}} = 0 to 16 {
//  CHECK-NEXT:     affine.for %{{.*}} = 0 to 512 step 16 {
// CHECK-COUNT-16:   arith.muli
//   CHECK-NOT:       arith.muli
  upmem.dpu_program @reduction_over_parent() tasklets(1) {
    %A = memref.alloca() : memref<16x512xi32, #upmem.wram>
    %x = memref.alloca() : memref<16xi32, #upmem.wram>
    %z = memref.alloca() : memref<512xi32, #upmem.wram>
    affine.for %j = 0 to 16 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%j, %k] : memref<16x512xi32, #upmem.wram>
        %xv = affine.load %x[%j] : memref<16xi32, #upmem.wram>
        %acc = affine.load %z[%k] : memref<512xi32, #upmem.wram>
        %p = arith.muli %a, %xv : i32
        %s = arith.addi %acc, %p : i32
        affine.store %s, %z[%k] : memref<512xi32, #upmem.wram>
      }
    }
    upmem.return
  }
