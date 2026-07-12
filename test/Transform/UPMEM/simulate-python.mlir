// RUN: cinm-opt --upmem-annotate-costs=simulator=cycle-accurate %s | FileCheck %s

// CHECK: upmem.wait_for %{{.*}} {upmem.sim_cost = 

#map = affine_map<(d0, d1) -> (d0 * 16 + d1 * 16, 0)>
#map1 = affine_map<(d0, d1) -> (0)>
#map2 = affine_map<(d0, d1) -> (d0 * 16 + d1 * 16)>
#upmem = #upmem.platform<type = v1A, dimensions = 16x32x24>
module {
  func.func @gemv_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> attributes {cinm.available_platforms = [#upmem]} {
    %0 = cinm.compute_block on accelerator #upmem.array<1x1x16, <type = v1A, dimensions = 16x32x24>> (%arg2 = %arg0 : tensor<?x?xf32>, %arg3 = %arg1 : tensor<?xf32>) -> tensor<?xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1 = bufferization.to_buffer %arg3 : tensor<?xf32> to memref<?xf32>
      %2 = bufferization.to_buffer %arg2 : tensor<?x?xf32> to memref<?x?xf32>
      %c16 = arith.constant  16 : index
      %c0 = arith.constant  0 : index
      %c1 = arith.constant  1 : index
      %3 = memref.get_global @__constant_16xf32_0 : memref<16xf32> 
      %dim = memref.dim  %2, %c0 : memref<?x?xf32>
      %dim_0 = memref.dim  %2, %c1 : memref<?x?xf32>
      %alloc = memref.alloc(%dim) {alignment = 64 : i64, upmem.sim_cost = 1.000000e+00 : f64} : memref<?xf32>
      %4 = bufferization.to_tensor %alloc : memref<?xf32> to tensor<?xf32>
      %5 = upmem.alloc_dpus with program @dpu_kernels_0::@program  : !upmem.hierarchy<1x1x16>
      scf.for %arg4 = %c0 to %dim step %c16 {
        %subview = memref.subview %alloc[%arg4] [16] [1]  : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
        memref.copy %3, %subview  : memref<16xf32> to memref<16xf32, strided<[1], offset: ?>>
        scf.for %arg5 = %c0 to %dim_0 step %c1 {
          %subview_1 = memref.subview %2[%arg4, %arg5] [16, 1] [1, 1]  : memref<?x?xf32> to memref<16x1xf32, strided<[?, 1], offset: ?>>
          %subview_2 = memref.subview %1[%arg5] [1] [1]  : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          upmem.scatter %subview_1[16, #map] onto @buf of %5  : memref<16x1xf32, strided<[?, 1], offset: ?>> onto !upmem.hierarchy<1x1x16>
          upmem.scatter %subview_2[1, #map1] onto @buf_1 of %5  : memref<1xf32, strided<[1], offset: ?>> onto !upmem.hierarchy<1x1x16>
          upmem.scatter %subview[16, #map2] onto @buf_2 of %5  : memref<16xf32, strided<[1], offset: ?>> onto !upmem.hierarchy<1x1x16>
          upmem.wait_for %5  : !upmem.hierarchy<1x1x16>
          upmem.gather %subview[16, #map2] from @buf_2 of %5  : memref<16xf32, strided<[1], offset: ?>> from !upmem.hierarchy<1x1x16>
        } 
      } 
      upmem.free_dpus %5  : !upmem.hierarchy<1x1x16>
      cinm.yield  %4 : tensor<?xf32>
    }
    return %0 : tensor<?xf32>
  }
  memref.global "private" constant @__constant_16xf32_0 : memref<16xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  module @dpu_kernels_0 {
    upmem.dpu_program @program() tasklets(16) {
      %pwram_buf = upmem.pwram_alloc()  : memref<1xf32, #upmem.wram>
      %mram_buf = upmem.static_alloc @buf(mram)  : memref<16x1xf32, #upmem.mram>
      %wram_buf = upmem.static_alloc @buf_0(wram) noinit  : memref<1xf32, #upmem.wram>
      %mram_buf_0 = upmem.static_alloc @buf_1(mram)  : memref<1xf32, #upmem.mram>
      %pwram_buf_1 = upmem.pwram_alloc()  : memref<f32, #upmem.wram>
      %mram_buf_2 = upmem.static_alloc @buf_2(mram)  : memref<16xf32, #upmem.mram>
      %0 = upmem.tasklet_dim() 
      %subview = memref.subview %mram_buf[%0, 0] [1, 1] [1, 1]  : memref<16x1xf32, #upmem.mram> to memref<1xf32, strided<[1], offset: ?>, #upmem.mram>
      upmem.local_transfer %subview into %pwram_buf  : memref<1xf32, strided<[1], offset: ?>, #upmem.mram> to memref<1xf32, #upmem.wram>
      %c0 = arith.constant  0 : index
      %1 = arith.cmpi eq, %0, %c0  : index
      scf.if %1 {
        upmem.local_transfer %mram_buf_0 into %wram_buf  : memref<1xf32, #upmem.mram> to memref<1xf32, #upmem.wram>
      } 
      upmem.barrier() 
      %subview_3 = memref.subview %mram_buf_2[%0] [1] [1]  : memref<16xf32, #upmem.mram> to memref<f32, strided<[], offset: ?>, #upmem.mram>
      upmem.local_transfer %subview_3 into %pwram_buf_1  : memref<f32, strided<[], offset: ?>, #upmem.mram> to memref<f32, #upmem.wram>
      %2 = memref.load %pwram_buf[%c0]  : memref<1xf32, #upmem.wram>
      %3 = memref.load %wram_buf[%c0]  : memref<1xf32, #upmem.wram>
      %4 = memref.load %pwram_buf_1[]  : memref<f32, #upmem.wram>
      %5 = arith.mulf %2, %3  : f32
      %6 = arith.addf %4, %5  : f32
      memref.store %6, %pwram_buf_1[]  : memref<f32, #upmem.wram>
      upmem.local_transfer %pwram_buf_1 into %subview_3  : memref<f32, #upmem.wram> to memref<f32, strided<[], offset: ?>, #upmem.mram>
      upmem.return 
    }
  }
}
