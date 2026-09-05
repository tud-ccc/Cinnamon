// RUN: cinm-translate --allow-unregistered-dialect --mlir-to-upmem-cpp %s | FileCheck %s

// The structural contract of the emitter: the COMPILE header names the
// program and its tasklet count, MRAM symbols become sized __mram arrays, and
// the transfers/synchronization of the kernel survive as mram_read/mram_write
// and barrier_wait with the right addresses and byte counts.

// CHECK: UPMEM-TRANSLATE: COMPILE_gemv:1:1024:gemv;
// CHECK: BARRIER_INIT(my_barrier, NR_TASKLETS)
// CHECK: char __mram __dma_aligned bufa[262144]; // int64_t[32][1024]
// CHECK: char __mram __dma_aligned bufx[8192]; // int64_t[1024]
// CHECK: char __mram __dma_aligned bufy[256]; // int64_t[32]
// WRAM buffers keep their element type: they are dereferenced element by
// element, and as byte arrays those accesses would read one byte at an
// offset scaled wrong by the element width. MRAM buffers stay byte arrays --
// their bytes only move through mram_read/mram_write, which index in bytes.
// CHECK: int64_t __dma_aligned [[V1:v[0-9]+]][1]; // int64_t[1][1][1][1]
// CHECK: void gemv(void) {
// CHECK: int64_t {{v[0-9]+}} = [[V1]][0];
// CHECK: for (int32_t [[I:v[0-9]+]] = 0; [[I]] < 32; [[I]] += 1) {
// CHECK: for (int32_t [[K:v[0-9]+]] = 0; [[K]] < 1024; [[K]] += 1) {
// CHECK: barrier_wait(&my_barrier);
// CHECK: mram_read(&bufx[0 + ([[K]] * 8) + 0], &((char*) {{v[0-9]+}})[0 + 0], 8);
// CHECK: mram_read(&bufa[0 + ([[I]] * 8192) + ([[K]] * 8) + 0], &((char*) {{v[0-9]+}})[0 + 0], 8);
// CHECK: mram_read(&bufy[0 + ([[I]] * 8) + 0], &((char*) {{v[0-9]+}})[0 + 0], 8);
// CHECK: mram_write(&((const char*) {{v[0-9]+}})[0 + 0], &bufy[0 + ([[I]] * 8) + 0], 8);
// CHECK: int main(void) {
upmem.dpu_program @gemv() tasklets(1) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %mram_buf = upmem.static_alloc @bufa(mram) : memref<32x1024xi64, #upmem.mram>
  %mram_buf_0 = upmem.static_alloc @bufx(mram) : memref<1024xi64, #upmem.mram>
  %mram_buf_1 = upmem.static_alloc @bufy(mram) : memref<32xi64, #upmem.mram>
  %wram_buf = upmem.static_alloc(wram) : memref<1x1x1x1xi64, #upmem.wram>
  %wram_buf_2 = upmem.static_alloc(wram) : memref<1xi64, #upmem.wram>
  %wram_buf_3 = upmem.static_alloc(wram) : memref<1x1x1xi64, #upmem.wram>
  %wram_buf_4 = upmem.static_alloc(wram) : memref<1x1xi64, #upmem.wram>
  %subview = memref.subview %wram_buf_4[0, 0] [1, 1] [1, 1] : memref<1x1xi64, #upmem.wram> to memref<1xi64, strided<[1]>, #upmem.wram>
  %0 = memref.load %wram_buf[%c0, %c0, %c0, %c0] : memref<1x1x1x1xi64, #upmem.wram>
  %1 = memref.load %wram_buf_4[%c0, %c0] : memref<1x1xi64, #upmem.wram>
  scf.for %arg0 = %c0 to %c32 step %c1 {
    memref.store %c0_i64, %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi64, #upmem.wram>
    %2 = scf.for %arg1 = %c0 to %c1024 step %c1 iter_args(%arg2 = %c0_i64) -> (i64) {
      upmem.barrier()
      %subview_6 = memref.subview %mram_buf_0[%arg1] [1] [1] : memref<1024xi64, #upmem.mram> to memref<1xi64, strided<[1], offset: ?>, #upmem.mram>
      upmem.local_transfer %subview_6 into %wram_buf_2 : memref<1xi64, strided<[1], offset: ?>, #upmem.mram> to memref<1xi64, #upmem.wram>
      %subview_7 = memref.subview %mram_buf[%arg0, %arg1] [1, 1] [1, 1] : memref<32x1024xi64, #upmem.mram> to memref<1x1xi64, strided<[1024, 1], offset: ?>, #upmem.mram>
      %reinterpret_cast_8 = memref.reinterpret_cast %wram_buf to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<1x1x1x1xi64, #upmem.wram> to memref<1x1xi64, #upmem.wram>
      upmem.local_transfer %subview_7 into %reinterpret_cast_8 : memref<1x1xi64, strided<[1024, 1], offset: ?>, #upmem.mram> to memref<1x1xi64, #upmem.wram>
      upmem.barrier()
      %5 = memref.load %wram_buf_2[%c0] : memref<1xi64, #upmem.wram>
      %6 = arith.muli %0, %5 : i64
      %7 = arith.addi %arg2, %6 : i64
      scf.yield %7 : i64
    }
    memref.store %2, %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi64, #upmem.wram>
    upmem.barrier()
    %subview_5 = memref.subview %mram_buf_1[%arg0] [1] [1] : memref<32xi64, #upmem.mram> to memref<1xi64, strided<[1], offset: ?>, #upmem.mram>
    upmem.local_transfer %subview_5 into %subview : memref<1xi64, strided<[1], offset: ?>, #upmem.mram> to memref<1xi64, strided<[1]>, #upmem.wram>
    %3 = memref.load %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi64, #upmem.wram>
    %4 = arith.addi %1, %3 : i64
    memref.store %4, %wram_buf_3[%c0, %c0, %c0] : memref<1x1x1xi64, #upmem.wram>
    upmem.barrier()
    %reinterpret_cast = memref.reinterpret_cast %wram_buf_3 to offset: [0], sizes: [1], strides: [1] : memref<1x1x1xi64, #upmem.wram> to memref<1xi64, #upmem.wram>
    upmem.local_transfer %reinterpret_cast into %subview_5 : memref<1xi64, #upmem.wram> to memref<1xi64, strided<[1], offset: ?>, #upmem.mram>
  }
  upmem.return
}
