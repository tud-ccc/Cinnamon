// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s

// The copies of an unrolled body index `iv + k`. The constant goes outside the
// subscript, as an offset from the address of the rest, so that every copy
// shares one base address and loads at an immediate offset from it.

// CHECK-LABEL: void k(void)
// CHECK: for (int32_t [[ROW:v[0-9]+]] = 0;
// CHECK: for (int32_t [[IV:v[0-9]+]] = 0; [[IV]] < 32; [[IV]] += 16)
// CHECK: = [[A:v[0-9]+]][([[ROW]] * 32) + [[IV]]];
// CHECK: = [[B:v[0-9]+]][[[IV]]];
// CHECK: = (&[[A]][([[ROW]] * 32) + [[IV]]])[1];
// CHECK: = (&[[B]][[[IV]]])[1];
// CHECK: (&[[A]][([[ROW]] * 32) + [[IV]]])[2] =
// A subscript with no constant to peel is unchanged.
// CHECK: [[B]][[[ROW]]] =
upmem.dpu_program @k() tasklets(16) {
  %a = memref.alloca() : memref<8x32xi32, #upmem.wram>
  %b = memref.alloca() : memref<32xi32, #upmem.wram>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  scf.for %row = %c0 to %c8 step %c1 {
  scf.for %iv = %c0 to %c32 step %c16 {
    %x0 = memref.load %a[%row, %iv] : memref<8x32xi32, #upmem.wram>
    %y0 = memref.load %b[%iv] : memref<32xi32, #upmem.wram>
    %i1 = arith.addi %iv, %c1 : index
    %x1 = memref.load %a[%row, %i1] : memref<8x32xi32, #upmem.wram>
    %y1 = memref.load %b[%i1] : memref<32xi32, #upmem.wram>
    %i2 = arith.addi %c2, %iv : index
    %s0 = arith.addi %x0, %y0 : i32
    %s1 = arith.addi %x1, %y1 : i32
    %s = arith.addi %s0, %s1 : i32
    memref.store %s, %a[%row, %i2] : memref<8x32xi32, #upmem.wram>
    memref.store %s, %b[%row] : memref<32xi32, #upmem.wram>
  }
  }
  upmem.return
}
