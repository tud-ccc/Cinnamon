// RUN: cinm-translate --mlir-to-upmem-cpp %s | FileCheck %s

// A named WRAM buffer is written by the host through its symbol and only read
// by the kernel, so it is __host: without it the DPU compiler folds the loads
// of a global nothing stores to and drops the symbol the host copies into.

// CHECK: int32_t __host __dma_aligned slot[2];

// Constants are inlined rather than declared, so an operand that is one is
// printed as its value wherever an expression names an operand -- a clamp's
// bound included.

// CHECK: int32_t [[X:v[0-9]+]] = slot[0];
// CHECK: int32_t [[M:v[0-9]+]] = ([[X]] >= (-330)) ? [[X]] : (-330);
// CHECK: int32_t [[S:v[0-9]+]] = {{v[0-9]+}} ? [[M]] : (7);

// A narrowing is a cast.

// CHECK: int8_t {{v[0-9]+}} = (int8_t)[[S]];
upmem.dpu_program @k() tasklets(1) {
  %c0 = arith.constant 0 : index
  %lo = arith.constant -330 : i32
  %zero = arith.constant 0 : i32
  %seven = arith.constant 7 : i32
  %slot = upmem.static_alloc @slot(wram) noinit : memref<2xi32, #upmem.wram>
  %out = memref.alloca() : memref<8xi8, #upmem.wram>
  %x = memref.load %slot[%c0] : memref<2xi32, #upmem.wram>
  %m = arith.maxsi %x, %lo : i32
  %neg = arith.cmpi slt, %m, %zero : i32
  %s = arith.select %neg, %m, %seven : i32
  %t = arith.trunci %s : i32 to i8
  memref.store %t, %out[%c0] : memref<8xi8, #upmem.wram>
  upmem.return
}
