// RUN: cinm-translate --allow-unregistered-dialect --mlir-to-upmem-cpp %s | FileCheck %s
upmem.dpu_program @gemv() tasklets(1) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %mram_buf = upmem.static_alloc @bufa(mram) : memref<32x1024xi32, #upmem.mram>
  %mram_buf_0 = upmem.static_alloc @bufx(mram) : memref<1024xi32, #upmem.mram>
  %mram_buf_1 = upmem.static_alloc @bufy(mram) : memref<32xi32, #upmem.mram>
  %wram_buf = upmem.static_alloc(wram) : memref<1x1x1x1xi32, #upmem.wram>
  %wram_buf_2 = upmem.static_alloc(wram) : memref<1xi32, #upmem.wram>
  %wram_buf_3 = upmem.static_alloc(wram) : memref<1x1x1xi32, #upmem.wram>
  %wram_buf_4 = upmem.static_alloc(wram) : memref<1x1xi32, #upmem.wram>
  %subview = memref.subview %wram_buf_4[0, 0] [1, 1] [1, 1] : memref<1x1xi32, #upmem.wram> to memref<1xi32, strided<[1]>, #upmem.wram>
  %0 = memref.load %wram_buf[%c0, %c0, %c0, %c0] : memref<1x1x1x1xi32, #upmem.wram>
  %1 = memref.load %wram_buf_4[%c0, %c0] : memref<1x1xi32, #upmem.wram>
  scf.for %arg0 = %c0 to %c32 step %c1 {
    memref.store %c0_i32, %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
    %2 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
      upmem.barrier()
      %subview_6 = memref.subview %mram_buf_0[%arg1] [1] [1] : memref<1024xi32, #upmem.mram> to memref<1xi32, strided<[1], offset: ?>, #upmem.mram>
      upmem.local_transfer %subview_6 into %wram_buf_2 : memref<1xi32, strided<[1], offset: ?>, #upmem.mram> to memref<1xi32, #upmem.wram>
      %subview_7 = memref.subview %mram_buf[%arg0, %arg1] [1, 1] [1, 1] : memref<32x1024xi32, #upmem.mram> to memref<1x1xi32, strided<[1024, 1], offset: ?>, #upmem.mram>
      %reinterpret_cast_8 = memref.reinterpret_cast %wram_buf to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<1x1x1x1xi32, #upmem.wram> to memref<1x1xi32, #upmem.wram>
      upmem.local_transfer %subview_7 into %reinterpret_cast_8 : memref<1x1xi32, strided<[1024, 1], offset: ?>, #upmem.mram> to memref<1x1xi32, #upmem.wram>
      upmem.barrier()
      %5 = memref.load %wram_buf_2[%c0] : memref<1xi32, #upmem.wram>
      %6 = arith.muli %0, %5 : i32
      %7 = arith.addi %arg2, %6 : i32
      scf.yield %7 : i32
    }
    memref.store %2, %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
    upmem.barrier()
    %subview_5 = memref.subview %mram_buf_1[%arg0] [1] [1] : memref<32xi32, #upmem.mram> to memref<1xi32, strided<[1], offset: ?>, #upmem.mram>
    upmem.local_transfer %subview_5 into %subview : memref<1xi32, strided<[1], offset: ?>, #upmem.mram> to memref<1xi32, strided<[1]>, #upmem.wram>
    %3 = memref.load %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
    %4 = arith.addi %1, %3 : i32
    memref.store %4, %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi32, #upmem.wram>
    upmem.barrier()
    %reinterpret_cast = memref.reinterpret_cast %wram_buf_3 to offset: [0], sizes: [1], strides: [1] : memref<1x1x1xi32, #upmem.wram> to memref<1xi32, #upmem.wram>
    upmem.local_transfer %reinterpret_cast into %subview_5 : memref<1xi32, #upmem.wram> to memref<1xi32, strided<[1], offset: ?>, #upmem.mram>
  }
  upmem.return
}