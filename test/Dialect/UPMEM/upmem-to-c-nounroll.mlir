// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s

// A loop marked upmem.nounroll is emitted behind a pragma that keeps the DPU
// compiler from unrolling it; an unmarked loop is left to the compiler.

// CHECK:      #pragma clang loop unroll(disable)
// CHECK-NEXT: for (
// CHECK-NOT:  #pragma
// CHECK:      for (
upmem.dpu_program @k() tasklets(1) {
  %buf = memref.alloca() : memref<8xi32, #upmem.wram>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %v = arith.constant 1 : i32
  scf.for %i = %c0 to %c8 step %c1 {
    memref.store %v, %buf[%i] : memref<8xi32, #upmem.wram>
  } {upmem.nounroll}
  scf.for %i = %c0 to %c8 step %c1 {
    memref.store %v, %buf[%i] : memref<8xi32, #upmem.wram>
  }
  upmem.return
}
