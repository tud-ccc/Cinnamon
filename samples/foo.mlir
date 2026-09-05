      #map = affine_map<(d0, d1, d2) -> (d0 * 16 + d1 * 4 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
#upmem_24_4_4 = #upmem.array<96x4, #upmem>
module {
  memref.global "private" constant @__constant_768xf32 : memref<768xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @host(%arg0: tensor<768x768xf32>, %arg1: tensor<768xf32>) -> tensor<768
xf32> {
    %0 = bufferization.to_buffer %arg1 : tensor<768xf32> to memref<768xf32>
    %1 = bufferization.to_buffer %arg0 : tensor<768x768xf32> to memref<768x768xf32>
    %2 = cinm.compute_block on accelerator #upmem.array<96x4, <type = v1A, dpus = 2560, tasklets = 24>> (%arg2 = %1 : memref<768x768xf32>, %arg3 = %0 : memref<768xf32>)
-> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %4 = memref.get_global @__constant_768xf32 : memref<768xf32>
      %5 = cnm.workgroup : !cnm.workgroup<#upmem_24_4_4>
      %cnm_buf = cnm.declare_buffer() for %5 : !cnm.buffer<2xf32 on #upmem_24_4_4>
      %cnm_buf_0 = cnm.declare_buffer() for %5 : !cnm.buffer<96xf32 on #upmem_24_4_4>
      %cnm_buf_1 = cnm.declare_buffer() for %5 : !cnm.buffer<2x96xf32 on #upmem_24_4_4>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      memref.copy %4, %alloc : memref<768xf32> to memref<768xf32>
      %6 = affine.for %i = 0 to 768 step 96 iter_args(%acc = %alloc) -> (memref<768xf32>) {
        %expand_shape = memref.expand_shape %acc [[0, 1]] output_shape [384, 2] : memref<768xf32> into memref<384x2xf32>
        %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<384x2xf32>
        %subview = memref.subview %arg2[0, %i] [768, 96] [1, 1] : memref<768x768xf32> to memref<768x96xf32, strided<[768, 1], offset: ?>>
        %subview_3 = memref.subview %arg3[%i] [96] [1] : memref<768xf32> to memref<96xf32, strided<[1], offset: ?>>
        %expand_shape_4 = memref.expand_shape %subview [[0, 1], [2]] output_shape [384, 2, 96] : memref<768x96xf32, strided<[768, 1], offset: ?>> into memref<384x2x96xf32, strided<[1536, 768, 1], offset: ?>>
        cnm.scatter %expand_shape_4 into %cnm_buf_1[#map] of %5 : memref<384x2x96xf32, strided<[1536, 768, 1], offset: ?>> into !cnm.buffer<2x96xf32 on #upmem_24_4_4>
        cnm.scatter %subview_3 into %cnm_buf_0[#map1] of %5 : memref<96xf32, strided<[1], offset: ?>> into !cnm.buffer<96xf32 on #upmem_24_4_4>
        %expand_shape_5 = memref.expand_shape %acc [[0, 1]] output_shape [384, 2] : memref<768xf32> into memref<384x2xf32>
        cnm.scatter %expand_shape_5 into %cnm_buf[#map] of %5 : memref<384x2xf32> into !cnm.buffer<2xf32 on #upmem_24_4_4>
        cnm.launch %5 ins(%arg4 = %cnm_buf_1 : <2x96xf32>, %arg5 = %cnm_buf_0 : <96xf32>) outs(%arg6 = %cnm_buf : <2xf32>) on !cnm.workgroup<#upmem_24_4_4> {
          linalg.contract indexing_maps = [#map2, #map3, #map4] ins(%arg4, %arg5 : memref<2x96xf32>, memref<96xf32>) outs(%arg6 : memref<2xf32>)
        }
        cnm.gather %cnm_buf[#map] of %5 into %alloc_2 : !cnm.buffer<2xf32 on #upmem_24_4_4> into memref<384x2xf32>
        %collapse_shape = memref.collapse_shape %alloc_2 [[0, 1]] : memref<384x2xf32> into memref<768xf32>
        memref.copy %collapse_shape, %acc : memref<768xf32> to memref<768xf32>
        affine.yield %acc : memref<768xf32>
      }
      cnm.free_workgroup %5 : !cnm.workgroup<#upmem_24_4_4>
      cinm.yield %6 : memref<768xf32>
    }
    %3 = bufferization.to_tensor %2 : memref<768xf32> to tensor<768xf32>
    return %3 : tensor<768xf32>
  }
}
