#map = affine_map<(d0, d1, d2) -> (d0 * 128 + d1 + d2)>
module {
  memref.global "private" constant @__constant_1xi64_1 : memref<1xi64> = dense<32768>
  memref.global "private" constant @__constant_2048x16xi32 : memref<2048x16xi32> = dense<0>
  memref.global "private" constant @__constant_2xi64_2 : memref<2xi64> = dense<[2048, 16]>
  memref.global "private" constant @__constant_32768xi32 : memref<32768xi32> = dense<0>
  memref.global "private" constant @__constant_2xi64_1 : memref<2xi64> = dense<[16, 1048576]>
  memref.global "private" constant @__constant_1xi64_0 : memref<1xi64> = dense<16384>
  memref.global "private" constant @__constant_1024x16xi32 : memref<1024x16xi32> = dense<0>
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[1024, 16]>
  memref.global "private" constant @__constant_16384xi32 : memref<16384xi32> = dense<0>
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<16777216>
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[8, 2097152]>
  func.func @va_8(%arg0: memref<8x2097152xi32>, %arg1: memref<8x2097152xi32>) {
    %c16384 = arith.constant 16384 : index
    %c16777216 = arith.constant 16777216 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape = memref.reshape %arg0(%0) : (memref<8x2097152xi32>, memref<1xi64>) -> memref<16777216xi32>
    %reshape_0 = memref.reshape %arg1(%0) : (memref<8x2097152xi32>, memref<1xi64>) -> memref<16777216xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
    %1 = scf.for %arg2 = %c0 to %c16777216 step %c16384 iter_args(%arg3 = %alloc) -> (memref<16777216xi32>) {
      %subview = memref.subview %reshape[%arg2] [16384] [1] : memref<16777216xi32> to memref<16384xi32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %reshape_0[%arg2] [16384] [1] : memref<16777216xi32> to memref<16384xi32, strided<[1], offset: ?>>
      %2 = upmem.alloc_dpus : !upmem.hierarchy<8x128x1>
      %3 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16384xi32>
      memref.copy %subview, %alloc_2 : memref<16384xi32, strided<[1], offset: ?>> to memref<16384xi32>
      %reshape_3 = memref.reshape %alloc_2(%3) : (memref<16384xi32>, memref<2xi64>) -> memref<1024x16xi32>
      upmem.scatter %reshape_3[0, 16, #map] onto %2 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16384xi32>
      memref.copy %subview_1, %alloc_4 : memref<16384xi32, strided<[1], offset: ?>> to memref<16384xi32>
      %reshape_5 = memref.reshape %alloc_4(%3) : (memref<16384xi32>, memref<2xi64>) -> memref<1024x16xi32>
      upmem.scatter %reshape_5[64, 16, #map] onto %2 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %4 = memref.get_global @__constant_1024x16xi32 : memref<1024x16xi32>
      upmem.scatter %4[128, 16, #map] onto %2 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      upmem.launch_func  @va_8_dpu::@main %2 : !upmem.hierarchy<8x128x1> 
      %alloc_6 = memref.alloc() : memref<1024x16xi32>
      upmem.gather %alloc_6[128, 16, #map] from %2 : memref<1024x16xi32> from !upmem.hierarchy<8x128x1>
      %5 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
      %reshape_7 = memref.reshape %alloc_6(%5) : (memref<1024x16xi32>, memref<1xi64>) -> memref<16384xi32>
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
      memref.copy %arg3, %alloc_8 : memref<16777216xi32> to memref<16777216xi32>
      %subview_9 = memref.subview %alloc_8[%arg2] [16384] [1] : memref<16777216xi32> to memref<16384xi32, strided<[1], offset: ?>>
      memref.copy %reshape_7, %subview_9 : memref<16384xi32> to memref<16384xi32, strided<[1], offset: ?>>
      scf.yield %alloc_8 : memref<16777216xi32>
    }
    return
  }
  upmem.module @va_8_dpu{
    upmem.func @main() {
      %0 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %1 = upmem.dpu_heap_base_addr : index
      %2 = upmem.pwram_alloc : memref<16xi32>
      %3 = arith.addi %1, %c16 : index
      %4 = upmem.pwram_alloc : memref<16xi32>
      %5 = arith.addi %3, %c16 : index
      %6 = upmem.pwram_alloc : memref<16xi32>
      upmem.memcpy  mram_to_wram %2, %c16, %1 : memref<16xi32>, index, index
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %7 = memref.load %2[%arg0] : memref<16xi32>
        %8 = memref.load %4[%arg0] : memref<16xi32>
        %9 = arith.addi %7, %8 : i32
        memref.store %9, %6[%arg0] : memref<16xi32>
      }
      upmem.memcpy  wram_to_mram %6, %c16, %5 : memref<16xi32>, index, index
      upmem.return
    }
  }
  func.func @va_16(%arg0: memref<16x1048576xi32>, %arg1: memref<16x1048576xi32>) {
    %c32768 = arith.constant 32768 : index
    %c16777216 = arith.constant 16777216 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape = memref.reshape %arg0(%0) : (memref<16x1048576xi32>, memref<1xi64>) -> memref<16777216xi32>
    %reshape_0 = memref.reshape %arg1(%0) : (memref<16x1048576xi32>, memref<1xi64>) -> memref<16777216xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
    %1 = scf.for %arg2 = %c0 to %c16777216 step %c32768 iter_args(%arg3 = %alloc) -> (memref<16777216xi32>) {
      %subview = memref.subview %reshape[%arg2] [32768] [1] : memref<16777216xi32> to memref<32768xi32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %reshape_0[%arg2] [32768] [1] : memref<16777216xi32> to memref<32768xi32, strided<[1], offset: ?>>
      %2 = upmem.alloc_dpus : !upmem.hierarchy<16x128x1>
      %3 = memref.get_global @__constant_2xi64_2 : memref<2xi64>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32768xi32>
      memref.copy %subview, %alloc_2 : memref<32768xi32, strided<[1], offset: ?>> to memref<32768xi32>
      %reshape_3 = memref.reshape %alloc_2(%3) : (memref<32768xi32>, memref<2xi64>) -> memref<2048x16xi32>
      upmem.scatter %reshape_3[0, 16, #map] onto %2 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32768xi32>
      memref.copy %subview_1, %alloc_4 : memref<32768xi32, strided<[1], offset: ?>> to memref<32768xi32>
      %reshape_5 = memref.reshape %alloc_4(%3) : (memref<32768xi32>, memref<2xi64>) -> memref<2048x16xi32>
      upmem.scatter %reshape_5[64, 16, #map] onto %2 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %4 = memref.get_global @__constant_2048x16xi32 : memref<2048x16xi32>
      upmem.scatter %4[128, 16, #map] onto %2 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      upmem.launch_func  @va_16_dpu::@main %2 : !upmem.hierarchy<16x128x1> 
      %alloc_6 = memref.alloc() : memref<2048x16xi32>
      upmem.gather %alloc_6[128, 16, #map] from %2 : memref<2048x16xi32> from !upmem.hierarchy<16x128x1>
      %5 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
      %reshape_7 = memref.reshape %alloc_6(%5) : (memref<2048x16xi32>, memref<1xi64>) -> memref<32768xi32>
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<16777216xi32>
      memref.copy %arg3, %alloc_8 : memref<16777216xi32> to memref<16777216xi32>
      %subview_9 = memref.subview %alloc_8[%arg2] [32768] [1] : memref<16777216xi32> to memref<32768xi32, strided<[1], offset: ?>>
      memref.copy %reshape_7, %subview_9 : memref<32768xi32> to memref<32768xi32, strided<[1], offset: ?>>
      scf.yield %alloc_8 : memref<16777216xi32>
    }
    return
  }
  upmem.module @va_16_dpu{
    upmem.func @main() {
      %0 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %1 = upmem.dpu_heap_base_addr : index
      %2 = upmem.pwram_alloc : memref<16xi32>
      %3 = arith.addi %1, %c16 : index
      %4 = upmem.pwram_alloc : memref<16xi32>
      %5 = arith.addi %3, %c16 : index
      %6 = upmem.pwram_alloc : memref<16xi32>
      upmem.memcpy  mram_to_wram %2, %c16, %1 : memref<16xi32>, index, index
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %7 = memref.load %2[%arg0] : memref<16xi32>
        %8 = memref.load %4[%arg0] : memref<16xi32>
        %9 = arith.addi %7, %8 : i32
        memref.store %9, %6[%arg0] : memref<16xi32>
      }
      upmem.memcpy  wram_to_mram %6, %c16, %5 : memref<16xi32>, index, index
      upmem.return
    }
  }
}

