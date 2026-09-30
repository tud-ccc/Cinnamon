// RUN: cinm-opt %s --split-input-file --upmem-register-tile-loops | FileCheck %s
// RUN: cinm-opt %s --split-input-file --upmem-register-tile-loops=register-budget=5 | FileCheck %s --check-prefix=BUDGET5

// The programs sit at the top level, without a module around them: a pass
// anchored on `upmem.dpu_program` and given on the command line only visits
// the top-level module's direct children. The pipeline nests it explicitly.

// The gemv nest linalg lowers to: y[j] += A[j][k] * x[k]. The operands are
// bytes widened to i32, which the DPU multiplies inline, so the whole register
// file is available. Per copy of j the inner body loads A[j][k] and y[j]; x[k]
// is shared. With a budget of 12,
// f * 2 + 1 <= 12 gives f = 5, and the largest divisor of 16 below that is
// 4. The reduction loop of 512 is unrolled by 16.
//
// CHECK-LABEL: upmem.dpu_program @gemv
//       CHECK:   affine.for %[[J:.*]] = 0 to 16 step 4 {
//  CHECK-NEXT:     affine.for %[[K:.*]] = 0 to 512 step 16 {
// The body holds 4 copies of 16 elements each.
// CHECK-COUNT-64: arith.muli
//   CHECK-NOT:       arith.muli
//       CHECK:     } {upmem.nounroll}
//  CHECK-NEXT:   } {upmem.nounroll}
//
// A budget of 5 leaves f = 2.
// BUDGET5-LABEL: upmem.dpu_program @gemv
//       BUDGET5:   affine.for %{{.*}} = 0 to 16 step 2 {
//  BUDGET5-NEXT:     affine.for %{{.*}} = 0 to 512 step 16 {
// BUDGET5-COUNT-32: arith.muli
//   BUDGET5-NOT:       arith.muli
//       BUDGET5:     }
  upmem.dpu_program @gemv() tasklets(1) {
    %A = memref.alloca() : memref<16x512xi8, #upmem.wram>
    %x = memref.alloca() : memref<512xi8, #upmem.wram>
    %y = memref.alloca() : memref<16xi32, #upmem.wram>
    affine.for %j = 0 to 16 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%j, %k] : memref<16x512xi8, #upmem.wram>
        %xv = affine.load %x[%k] : memref<512xi8, #upmem.wram>
        %acc = affine.load %y[%j] : memref<16xi32, #upmem.wram>
        %ae = arith.extsi %a : i8 to i32
        %xe = arith.extsi %xv : i8 to i32
        %p = arith.muli %ae, %xe : i32
        %s = arith.addi %acc, %p : i32
        affine.store %s, %y[%j] : memref<16xi32, #upmem.wram>
      }
    }
    upmem.return
  }

// -----

// The same gemv on i32 operands: the multiply is a call to `__mulsi3`, and
// what is live across it must sit in the 8 callee-saved registers. The jam
// gets that budget instead: two registers per copy (y[j] and A[j][k]) and the
// shared x[k] fit 3 copies, so 2. But jammed by 2, j keeps 8 trips, and a
// full unroll of k would hoist its 512 x[k] out of them; unrolled partially,
// k stays a loop, whose addresses take the registers the accumulators need.
// So j is not jammed, and k is unrolled by 16. Both loops left rolled are
// marked for the DPU compiler not to unroll them.
//
// CHECK-LABEL: upmem.dpu_program @gemv_i32
//       CHECK:   affine.for %{{.*}} = 0 to 16 {
//  CHECK-NEXT:     affine.for %{{.*}} = 0 to 512 step 16 {
// CHECK-COUNT-16: arith.muli
//   CHECK-NOT:       arith.muli
//       CHECK:     } {upmem.nounroll}
//  CHECK-NEXT:   } {upmem.nounroll}
  upmem.dpu_program @gemv_i32() tasklets(1) {
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

// The i32 gemv over 2 rows of 64: the jam by 2 takes every trip of j, so
// there is no loop left for x[k] to be hoisted out of, and k is unrolled
// fully after it.
//
// CHECK-LABEL: upmem.dpu_program @gemv_i32_two_rows
//   CHECK-NOT:   affine.for
// CHECK-COUNT-128: arith.muli
//   CHECK-NOT:   arith.muli
  upmem.dpu_program @gemv_i32_two_rows() tasklets(1) {
    %A = memref.alloca() : memref<2x64xi32, #upmem.wram>
    %x = memref.alloca() : memref<64xi32, #upmem.wram>
    %y = memref.alloca() : memref<2xi32, #upmem.wram>
    affine.for %j = 0 to 2 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%j, %k] : memref<2x64xi32, #upmem.wram>
        %xv = affine.load %x[%k] : memref<64xi32, #upmem.wram>
        %acc = affine.load %y[%j] : memref<2xi32, #upmem.wram>
        %p = arith.muli %a, %xv : i32
        %s = arith.addi %acc, %p : i32
        affine.store %s, %y[%j] : memref<2xi32, #upmem.wram>
      }
    }
    upmem.return
  }

// -----

// The same gemv with the accumulator promoted to the reduction loop's
// iteration arguments, as scalar replacement leaves it: y[j] is loaded before
// the inner loop and stored after it. Each copy's load and store are at its
// own y[j], so the jam applies as above -- the accumulator is an iteration
// argument per copy now, still one register each.
//
// CHECK-LABEL: upmem.dpu_program @gemv_promoted
//       CHECK:   affine.for %{{.*}} = 0 to 16 step 4 {
// CHECK-COUNT-4:   affine.load %{{.*}} : memref<16xi32, #upmem.wram>
//       CHECK:     affine.for %{{.*}} = 0 to 512 step 16 iter_args(
// CHECK-COUNT-64: arith.muli
//   CHECK-NOT:       arith.muli
//       CHECK:     }
// CHECK-COUNT-4:   affine.store %{{.*}} : memref<16xi32, #upmem.wram>
  upmem.dpu_program @gemv_promoted() tasklets(1) {
    %A = memref.alloca() : memref<16x512xi8, #upmem.wram>
    %x = memref.alloca() : memref<512xi8, #upmem.wram>
    %y = memref.alloca() : memref<16xi32, #upmem.wram>
    affine.for %j = 0 to 16 {
      %init = affine.load %y[%j] : memref<16xi32, #upmem.wram>
      %r = affine.for %k = 0 to 512 iter_args(%acc = %init) -> (i32) {
        %a = affine.load %A[%j, %k] : memref<16x512xi8, #upmem.wram>
        %xv = affine.load %x[%k] : memref<512xi8, #upmem.wram>
        %ae = arith.extsi %a : i8 to i32
        %xe = arith.extsi %xv : i8 to i32
        %p = arith.muli %ae, %xe : i32
        %s = arith.addi %acc, %p : i32
        affine.yield %s : i32
      }
      affine.store %r, %y[%j] : memref<16xi32, #upmem.wram>
    }
    upmem.return
  }

// -----

// Eight rows over a reduction of 32: the jam takes 4 rows, leaving the row
// loop with two trips. A full unroll of the reduction fits max-body-ops, but it
// would make all 32 x[k] invariant in the row loop, which loop-invariant code
// motion then hoists -- 32 values, far past the register budget. The
// reduction is unrolled by 16 instead, keeping two trips over which x[k]
// moves.
//
// CHECK-LABEL: upmem.dpu_program @jammed_short_reduction
//       CHECK:   affine.for %{{.*}} = 0 to 8 step 4 {
//       CHECK:     affine.for %{{.*}} = 0 to 32 step 16 iter_args(
// CHECK-COUNT-64: arith.muli
//   CHECK-NOT:       arith.muli
//       CHECK:     }
  upmem.dpu_program @jammed_short_reduction() tasklets(1) {
    %A = memref.alloca() : memref<8x32xi8, #upmem.wram>
    %x = memref.alloca() : memref<32xi8, #upmem.wram>
    %y = memref.alloca() : memref<8xi32, #upmem.wram>
    affine.for %j = 0 to 8 {
      %init = affine.load %y[%j] : memref<8xi32, #upmem.wram>
      %r = affine.for %k = 0 to 32 iter_args(%acc = %init) -> (i32) {
        %a = affine.load %A[%j, %k] : memref<8x32xi8, #upmem.wram>
        %xv = affine.load %x[%k] : memref<32xi8, #upmem.wram>
        %ae = arith.extsi %a : i8 to i32
        %xe = arith.extsi %xv : i8 to i32
        %p = arith.muli %ae, %xe : i32
        %s = arith.addi %acc, %p : i32
        affine.yield %s : i32
      }
      affine.store %r, %y[%j] : memref<8xi32, #upmem.wram>
    }
    upmem.return
  }

// -----

// A store around the inner loop that every copy would make to the same
// address: the copies do not touch locations of their own, so no jam.
//
// CHECK-LABEL: upmem.dpu_program @shared_outer_store
//       CHECK:   affine.for %{{.*}} = 0 to 16 {
//       CHECK:     affine.for %{{.*}} = 0 to 512 step 16 iter_args(
  upmem.dpu_program @shared_outer_store() tasklets(1) {
    %A = memref.alloca() : memref<16x512xi32, #upmem.wram>
    %x = memref.alloca() : memref<512xi32, #upmem.wram>
    %y = memref.alloca() : memref<16xi32, #upmem.wram>
    %c0_i32 = arith.constant 0 : i32
    affine.for %j = 0 to 16 {
      %r = affine.for %k = 0 to 512 iter_args(%acc = %c0_i32) -> (i32) {
        %a = affine.load %A[%j, %k] : memref<16x512xi32, #upmem.wram>
        %xv = affine.load %x[%k] : memref<512xi32, #upmem.wram>
        %p = arith.muli %a, %xv : i32
        %s = arith.addi %acc, %p : i32
        affine.yield %s : i32
      }
      affine.store %r, %y[0] : memref<16xi32, #upmem.wram>
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
// so the reuse loop is not jammed. A trip count of 8 is unrolled fully, and
// then j too: the whole nest, 16 x 8 elements, fits max-body-ops.
//
// CHECK-LABEL: upmem.dpu_program @no_reuse
//   CHECK-NOT:   affine.for
// CHECK-COUNT-128: arith.addi
//   CHECK-NOT:   arith.addi
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
