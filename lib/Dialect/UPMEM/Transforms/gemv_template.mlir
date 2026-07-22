#map = affine_map<(d0, d1) -> (d0 * 128 + d1, 0, 0)>
#map1 = affine_map<(d0, d1) -> (0)>
#map2 = affine_map<(d0, d1) -> (d0 * 128 + d1, 0)>
module {
  memref.global "private" constant @__constant_1024xi32 : memref<1024xi32> = dense<0> {alignment = 64 : i64}
  func.func @mv(%arg0: memref<4096x2048xi32>, %arg1: memref<2048xi32>) -> memref<4096xi32> {
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xi32>
    %0 = memref.get_global @__constant_1024xi32 : memref<1024xi32>
    %1 = upmem.alloc_dpus with program @dpu_kernels::@program : !upmem.hierarchy<4x128x1>
    cinm.compute on accelerator #upmem.array<4x128x1, <type = v1A, dimensions = 32x128x1>> {
      %alloc_0 = memref.alloc() : memref<512x2x512xi32>
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1024xi32>
      %alloca = memref.alloca() {alignment = 64 : i64} : memref<2xindex>
      scf.for %arg2 = %c0 to %c4096 step %c1024 {
        %subview = memref.subview %alloc[%arg2] [1024] [1] : memref<4096xi32> to memref<1024xi32, strided<[1], offset: ?>>
        memref.copy %0, %subview : memref<1024xi32> to memref<1024xi32, strided<[1], offset: ?>>
        %expand_shape = memref.expand_shape %subview [[0, 1]] output_shape [512, 2] : memref<1024xi32, strided<[1], offset: ?>> into memref<512x2xi32, strided<[2, 1], offset: ?>>
        scf.for %arg3 = %c0 to %c2048 step %c512 {
          %subview_2 = memref.subview %arg0[%arg2, %arg3] [1024, 512] [1, 1] : memref<4096x2048xi32> to memref<1024x512xi32, strided<[2048, 1], offset: ?>>
          %subview_3 = memref.subview %arg1[%arg3] [512] [1] : memref<2048xi32> to memref<512xi32, strided<[1], offset: ?>>
          memref.store %c512, %alloca[%c0] : memref<2xindex>
          memref.store %c2, %alloca[%c1] : memref<2xindex>
          memref.copy %subview, %alloc_1 : memref<1024xi32, strided<[1], offset: ?>> to memref<1024xi32>
          %reshape = memref.reshape %alloc_1(%alloca) : (memref<1024xi32>, memref<2xindex>) -> memref<512x2xi32>
          %expand_shape_4 = memref.expand_shape %subview_2 [[0, 1], [2]] output_shape [512, 2, 512] : memref<1024x512xi32, strided<[2048, 1], offset: ?>> into memref<512x2x512xi32, strided<[4096, 2048, 1], offset: ?>>
          memref.copy %expand_shape_4, %alloc_0 : memref<512x2x512xi32, strided<[4096, 2048, 1], offset: ?>> to memref<512x2x512xi32>
          upmem.scatter %alloc_0[1024 elts, #map] onto @buf of %1 : memref<512x2x512xi32> onto !upmem.hierarchy<4x128x1>
          upmem.scatter %subview_3[512 elts, #map1] onto @buf_1 of %1 : memref<512xi32, strided<[1], offset: ?>> onto !upmem.hierarchy<4x128x1>
          upmem.scatter %expand_shape[2 elts, #map2] onto @buf_2 of %1 : memref<512x2xi32, strided<[2, 1], offset: ?>> onto !upmem.hierarchy<4x128x1>
          upmem.wait_for %1 : !upmem.hierarchy<4x128x1>
          upmem.gather %reshape[2 elts, #map2] from @buf_2 of %1 : memref<512x2xi32> from !upmem.hierarchy<4x128x1>
          %collapse_shape = memref.collapse_shape %reshape [[0, 1]] : memref<512x2xi32> into memref<1024xi32>
          memref.copy %collapse_shape, %subview : memref<1024xi32> to memref<1024xi32, strided<[1], offset: ?>>
        }
      }
      cinm.yield
    }
    upmem.free_dpus %1 : !upmem.hierarchy<4x128x1>
    return %alloc : memref<4096xi32>
  }
  module @dpu_kernels {
    upmem.dpu_program @program() tasklets(1) {
      %pwram_buf = pwram_alloc() : memref<2x512xi32, #upmem.wram>
      %mram_buf = static_alloc @buf(mram) : memref<1x4x1024xi32, #upmem.mram>
      %wram_buf = static_alloc @buf_0(wram) noinit : memref<512xi32, #upmem.wram>
      %mram_buf_0 = static_alloc @buf_1(mram) : memref<1024xi32, #upmem.mram>
      %pwram_buf_1 = pwram_alloc() : memref<4xi32, #upmem.wram>
      %mram_buf_2 = static_alloc @buf_2(mram) : memref<1x4xi32, #upmem.mram>
      %0 = tasklet_dim()
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %is_t0 = arith.cmpi eq, %0, %c0 : index
      %as4 = memref.collapse_shape %mram_buf_2 [[0, 1]] : memref<1x4xi32, #upmem.mram> into memref<4xi32, #upmem.mram>
      upmem.local_transfer %as4 into %pwram_buf_1 : memref<4xi32, #upmem.mram> to memref<4xi32, #upmem.wram>
      scf.for %row_tile = %c0 to %c2 step %c1 { // mram row tile
        %row_off = arith.muli %row_tile, %c2 : index
        scf.for %tile = %c0 to %c2 step %c1 { // mram col tile
          %tile_off = arith.muli %tile, %c512 : index
          %subview = memref.subview %mram_buf[%0, %row_off, %tile_off] [1, 2, 512] [1, 1, 1] : memref<1x4x1024xi32, #upmem.mram> to memref<2x512xi32, strided<[1024, 1], offset: ?>, #upmem.mram>
          upmem.local_transfer %subview into %pwram_buf : memref<2x512xi32, strided<[1024, 1], offset: ?>, #upmem.mram> to memref<2x512xi32, #upmem.wram>
          scf.if %is_t0 {
            %subview_x = memref.subview %mram_buf_0[%tile_off] [512] [1] : memref<1024xi32, #upmem.mram> to memref<512xi32, strided<[1], offset: ?>, #upmem.mram>
            upmem.local_transfer %subview_x into %wram_buf : memref<512xi32, strided<[1], offset: ?>, #upmem.mram> to memref<512xi32, #upmem.wram>
          }
          upmem.barrier()
          scf.for %arg0 = %c0 to %c2 step %c1 { // wram reduction
            %2 = memref.load %pwram_buf_1[%row_off] : memref<4xi32, #upmem.wram>
            %3 = scf.for %arg1 = %c0 to %c512 step %c1 iter_args(%arg2 = %2) -> (i32) {
              %4 = memref.load %pwram_buf[%arg0, %arg1] : memref<2x512xi32, #upmem.wram>
              %5 = memref.load %wram_buf[%arg1] : memref<512xi32, #upmem.wram>
              %6 = arith.muli %4, %5 : i32
              %7 = arith.addi %arg2, %6 : i32
              scf.yield %7 : i32
            }
            memref.store %3, %pwram_buf_1[%row_off] : memref<4xi32, #upmem.wram>
          }
        }
      }
      upmem.local_transfer %pwram_buf_1 into %as4 : memref<4xi32, #upmem.wram> to memref<4xi32, #upmem.mram>
      return
    }
  }
}

