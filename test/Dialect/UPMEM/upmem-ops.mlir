// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s

#map = affine_map<(d0, d1) -> (d1 mod 4, 0)>
#map1 = affine_map<(d0, d1) -> (d1, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1 mod 8, 0)>
module {
  memref.global "private" constant @__constant_8x128xi32 : memref<8x128xi32> = dense<0> {alignment = 64 : i64}

  // CHECK-LABEL: @mm_dimm8_nopt
  func.func @mm_dimm8_nopt(%arg0: memref<8x1024xi32>, %arg1: memref<1024x128xi32>) -> memref<8x128xi32> {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_8x128xi32 : memref<8x128xi32>
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<8x128x1>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x1024xi32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c1024 step %c1 {
        %2 = memref.load %arg1[%arg3, %arg2] : memref<1024x128xi32>
        memref.store %2, %alloc[%arg2, %arg3] : memref<128x1024xi32>
      }
    }
    upmem.scatter %arg0[1024, #map3] onto @buf_1 of %1 : memref<8x1024xi32> onto !upmem.hierarchy<8x128x1>
    upmem.scatter %alloc[1024, #map1] onto @buf_0 of %1 : memref<128x1024xi32> onto !upmem.hierarchy<8x128x1>
    upmem.scatter %0[1, #map2] onto @buf of %1 : memref<8x128xi32> onto !upmem.hierarchy<8x128x1>
    upmem.wait_for %1 : !upmem.hierarchy<8x128x1>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x128xi32>
    upmem.gather %alloc_0[1, #map2] from @buf of %1 : memref<8x128xi32> from !upmem.hierarchy<8x128x1>
    upmem.free_dpus %1 : !upmem.hierarchy<8x128x1>
    return %alloc_0 : memref<8x128xi32>
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %pwram_buf = pwram_alloc() : memref<i32, "wram">
      %mram_buf = static_alloc @buf(mram) : memref<1xi32, "mram">
      %pwram_buf_0 = pwram_alloc() : memref<1024xi32, "wram">
      %mram_buf_1 = static_alloc @buf_0(mram) : memref<1x1024xi32, "mram">
      %pwram_buf_2 = pwram_alloc() : memref<1024xi32, "wram">
      %mram_buf_3 = static_alloc @buf_1(mram) : memref<1x1024xi32, "mram">
      %0 = tasklet_dim()
      %subview = memref.subview %mram_buf[%0] [1] [1] : memref<1xi32, "mram"> to memref<i32, strided<[], offset: ?>, "mram">
      local_transfer %subview into %pwram_buf : memref<i32, strided<[], offset: ?>, "mram"> to memref<i32, "wram">
      %subview_4 = memref.subview %mram_buf_1[%0, 0] [1, 1024] [1, 1] : memref<1x1024xi32, "mram"> to memref<1024xi32, strided<[1], offset: ?>, "mram">
      local_transfer %subview_4 into %pwram_buf_0 : memref<1024xi32, strided<[1], offset: ?>, "mram"> to memref<1024xi32, "wram">
      %subview_5 = memref.subview %mram_buf_3[%0, 0] [1, 1024] [1, 1] : memref<1x1024xi32, "mram"> to memref<1024xi32, strided<[1], offset: ?>, "mram">
      local_transfer %subview_5 into %pwram_buf_2 : memref<1024xi32, strided<[1], offset: ?>, "mram"> to memref<1024xi32, "wram">
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1024 step %c1 {
        %1 = memref.load %pwram_buf_2[%arg0] : memref<1024xi32, "wram">
        %2 = memref.load %pwram_buf_0[%arg0] : memref<1024xi32, "wram">
        %3 = memref.load %pwram_buf[] : memref<i32, "wram">
        %4 = arith.muli %1, %2 : i32
        %5 = arith.addi %3, %4 : i32
        memref.store %5, %pwram_buf[] : memref<i32, "wram">
      }
      local_transfer %pwram_buf into %subview : memref<i32, "wram"> to memref<i32, strided<[], offset: ?>, "mram">
      return
    }
  }
}


