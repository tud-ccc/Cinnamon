// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s

// The DPU compiler unrolls nothing by itself, so a short innermost loop pays
// its counter increment, its branch and an address computation per operand on
// every iteration -- around half the instructions of a multiply-accumulate
// body. Asking for full unrolling drops the loop control and turns the
// addresses into constant offsets.

// CHECK-LABEL: void k(
upmem.dpu_program @k() tasklets(1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c65 = arith.constant 65 : index
  %cst = arith.constant 0 : i32
  %w = memref.alloca() : memref<128xi32, #upmem.wram>

  // Innermost and short, so it is unrolled. 64 is the largest trip count that
  // still is -- the bound is the DPU's instruction memory.
  // CHECK:      #pragma clang loop unroll(full)
  // CHECK-NEXT: for (int32_t [[I:v[0-9]+]] = 0; [[I]] < 64; [[I]] += 1) {
  scf.for %i = %c0 to %c64 step %c1 {
    memref.store %cst, %w[%i] : memref<128xi32, #upmem.wram>
  }

  // One past it, so it is not.
  // CHECK-NOT:  #pragma
  // CHECK:      for (int32_t {{v[0-9]+}} = 0; {{v[0-9]+}} < 65; {{v[0-9]+}} += 1) {
  scf.for %i = %c0 to %c65 step %c1 {
    memref.store %cst, %w[%i] : memref<128xi32, #upmem.wram>
  }

  upmem.return
}


// A loop holding another is not unrolled however short it is: that duplicates
// the whole nest, so its own trip count bounds nothing. The inner one still is.

// CHECK-LABEL: void nest(
upmem.dpu_program @nest() tasklets(1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0 : i32
  %w = memref.alloca() : memref<4x8xi32, #upmem.wram>

  // CHECK-NOT:  #pragma
  // CHECK:      for (int32_t [[O:v[0-9]+]] = 0; [[O]] < 4; [[O]] += 1) {
  // CHECK:        #pragma clang loop unroll(full)
  // CHECK-NEXT:   for (int32_t [[N:v[0-9]+]] = 0; [[N]] < 8; [[N]] += 1) {
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      memref.store %cst, %w[%i, %j] : memref<4x8xi32, #upmem.wram>
    }
  }

  upmem.return
}


// A trip count that is not known at translation time cannot be checked
// against the bound, so it is left alone.

// CHECK-LABEL: void dyn(
upmem.dpu_program @dyn() tasklets(1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0 : i32
  %w = memref.alloca() : memref<128xi32, #upmem.wram>
  %n = upmem.tasklet_dim()

  // CHECK-NOT:  #pragma
  // CHECK:      for (int32_t {{v[0-9]+}} = 0;
  scf.for %i = %c0 to %n step %c1 {
    memref.store %cst, %w[%i] : memref<128xi32, #upmem.wram>
  }

  upmem.return
}
