#map = affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 16 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0 * 512 + d1 * 8 + d2)>
module {
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<16777216> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[16384, 1024]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16384x1024xi32 : memref<16384x1024xi32> = dense<0> {alignment = 64 : i64}
  func.func @va_8(%arg0: memref<8x2097152xi32>, %arg1: memref<8x2097152xi32>) {
    %0 = memref.get_global @__constant_16384x1024xi32 : memref<16384x1024xi32>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %2 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape = memref.reshape %arg0(%2) : (memref<8x2097152xi32>, memref<1xi64>) -> memref<16777216xi32>
    %reshape_0 = memref.reshape %arg1(%2) : (memref<8x2097152xi32>, memref<1xi64>) -> memref<16777216xi32>
    %3 = upmem.alloc_dpus : !upmem.hierarchy<16x64x16>
    %reshape_1 = memref.reshape %reshape(%1) : (memref<16777216xi32>, memref<2xi64>) -> memref<16384x1024xi32>
    upmem.scatter %reshape_1[8192, 1024, #map] onto %3 : memref<16384x1024xi32> onto !upmem.hierarchy<16x64x16>
    %reshape_2 = memref.reshape %reshape_0(%1) : (memref<16777216xi32>, memref<2xi64>) -> memref<16384x1024xi32>
    upmem.scatter %reshape_2[4096, 1024, #map] onto %3 : memref<16384x1024xi32> onto !upmem.hierarchy<16x64x16>
    upmem.scatter %0[0, 1024, #map] onto %3 : memref<16384x1024xi32> onto !upmem.hierarchy<16x64x16>
    upmem.launch_func  @dpu_kernels::@va_8 %3 : !upmem.hierarchy<16x64x16> 
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16384x1024xi32>
    upmem.gather %alloc[0, 1024, #map] from %3 : memref<16384x1024xi32> from !upmem.hierarchy<16x64x16>
    upmem.free_dpus %3 : !upmem.hierarchy<16x64x16>
    return
  }
  func.func @va_16(%arg0: memref<16x1048576xi32>, %arg1: memref<16x1048576xi32>) {
    %0 = memref.get_global @__constant_16384x1024xi32 : memref<16384x1024xi32>
    %1 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %2 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape = memref.reshape %arg0(%2) : (memref<16x1048576xi32>, memref<1xi64>) -> memref<16777216xi32>
    %reshape_0 = memref.reshape %arg1(%2) : (memref<16x1048576xi32>, memref<1xi64>) -> memref<16777216xi32>
    %3 = upmem.alloc_dpus : !upmem.hierarchy<32x64x8>
    %reshape_1 = memref.reshape %reshape(%1) : (memref<16777216xi32>, memref<2xi64>) -> memref<16384x1024xi32>
    upmem.scatter %reshape_1[8192, 1024, #map1] onto %3 : memref<16384x1024xi32> onto !upmem.hierarchy<32x64x8>
    %reshape_2 = memref.reshape %reshape_0(%1) : (memref<16777216xi32>, memref<2xi64>) -> memref<16384x1024xi32>
    upmem.scatter %reshape_2[4096, 1024, #map1] onto %3 : memref<16384x1024xi32> onto !upmem.hierarchy<32x64x8>
    upmem.scatter %0[0, 1024, #map1] onto %3 : memref<16384x1024xi32> onto !upmem.hierarchy<32x64x8>
    upmem.launch_func  @dpu_kernels::@va_16 %3 : !upmem.hierarchy<32x64x8> 
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16384x1024xi32>
    upmem.gather %alloc[0, 1024, #map1] from %3 : memref<16384x1024xi32> from !upmem.hierarchy<32x64x8>
    upmem.free_dpus %3 : !upmem.hierarchy<32x64x8>
    return
  }
  upmem.module @dpu_kernels {
    upmem.func @va_64_fast() attributes {num_tasklets = 1 : i64} {
      %cTOP = arith.constant 262144 : index
      %cTile = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c16384 = arith.constant 16384 : index

      %0 = upmem.dpu_heap_base_addr : index
      %2 = arith.addi %0, %c16384 : index
      %4 = arith.addi %2, %c16384 : index

      %Abuf = upmem.pwram_alloc : memref<4096xi32>
      %Bbuf = upmem.pwram_alloc : memref<4096xi32>
      %Cbuf = upmem.pwram_alloc : memref<4096xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      scf.for %off = %c0 to %cTOP step %cTile {
        %aoff = arith.addi %0, %off : index
        %boff = arith.addi %2, %off : index
        upmem.memcpy mram_to_wram %Abuf, %cTile, %aoff : memref<4096xi32>, index, index
        upmem.memcpy mram_to_wram %Bbuf, %cTile, %boff : memref<4096xi32>, index, index

        scf.for %arg0 = %c0 to %cTile step %c1 {
          %6 = memref.load %Abuf[%arg0] : memref<4096xi32>
          %7 = memref.load %Bbuf[%arg0] : memref<4096xi32>
          %8 = arith.addi %6, %7 : i32
          memref.store %8, %Cbuf[%arg0] : memref<4096xi32>
        }
        %coff = arith.addi %4, %off : index
        upmem.memcpy wram_to_mram %Cbuf, %cTile, %coff : memref<4096xi32>, index, index
      }
      upmem.return
    }
    upmem.func @va_8() attributes {num_tasklets = 16 : i64} {
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %c4 = arith.constant 4 : index
      %c16384 = arith.constant 16384 : index

      %0 = upmem.dpu_heap_base_addr : index
      %2 = arith.addi %0, %c16384 : index
      %4 = arith.addi %2, %c16384 : index

      %Abuf = upmem.pwram_alloc : memref<256xi32>
      %Bbuf = upmem.pwram_alloc : memref<256xi32>
      %Cbuf = upmem.pwram_alloc : memref<256xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      scf.for %off = %c0 to %c1024 step %c256 {
        %aoff = arith.addi %0, %off : index
        %boff = arith.addi %2, %off : index
        upmem.memcpy mram_to_wram %Abuf, %c256, %aoff : memref<256xi32>, index, index
        upmem.memcpy mram_to_wram %Bbuf, %c256, %boff : memref<256xi32>, index, index

        scf.for %arg0 = %c0 to %c256 step %c1 {
          %6 = memref.load %Abuf[%arg0] : memref<256xi32>
          %7 = memref.load %Bbuf[%arg0] : memref<256xi32>
          %8 = arith.addi %6, %7 : i32
          memref.store %8, %Cbuf[%arg0] : memref<256xi32>
        }
        %coff = arith.addi %4, %off : index
        upmem.memcpy wram_to_mram %Cbuf, %c256, %coff : memref<256xi32>, index, index
      }
      upmem.return
    }
    upmem.func @va_16() attributes {num_tasklets = 8 : i64} {
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c16384 = arith.constant 16384 : index

      %0 = upmem.dpu_heap_base_addr : index
      %2 = arith.addi %0, %c16384 : index
      %4 = arith.addi %2, %c16384 : index

      %Abuf = upmem.pwram_alloc : memref<128xi32>
      %Bbuf = upmem.pwram_alloc : memref<128xi32>
      %Cbuf = upmem.pwram_alloc : memref<128xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      scf.for %off = %c0 to %c1024 step %c128 {
        %aoff = arith.addi %0, %off : index
        %boff = arith.addi %2, %off : index
        upmem.memcpy mram_to_wram %Abuf, %c128, %aoff : memref<128xi32>, index, index
        upmem.memcpy mram_to_wram %Bbuf, %c128, %boff : memref<128xi32>, index, index

        scf.for %arg0 = %c0 to %c128 step %c1 {
          %6 = memref.load %Abuf[%arg0] : memref<128xi32>
          %7 = memref.load %Bbuf[%arg0] : memref<128xi32>
          %8 = arith.addi %6, %7 : i32
          memref.store %8, %Cbuf[%arg0] : memref<128xi32>
        }
        %coff = arith.addi %4, %off : index
        upmem.memcpy wram_to_mram %Cbuf, %c128, %coff : memref<128xi32>, index, index
      }
      upmem.return
    }
  }
}

