#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> ()>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> ()>
#map8 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
module {
  memref.global "private" constant @__constant_xf32_2 : memref<f32> = dense<1.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32_1 : memref<f32> = dense<6.92820311> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32_0 : memref<f32> = dense<0xFFC00000> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @forward(%arg0: index, %arg1: index, %arg2: memref<6x1024x768xf32>, %arg3: memref<6x1024x768xf32>, %arg4: memref<32000x768xf32> {cinm.static}, %arg5: memref<6x768xf32> {cinm.static}, %arg6: memref<6x768x768xf32> {cinm.static}, %arg7: memref<6x768x768xf32> {cinm.static}, %arg8: memref<6x768x768xf32> {cinm.static}, %arg9: memref<6x768x768xf32> {cinm.static}, %arg10: memref<6x2048x768xf32> {cinm.static}, %arg11: memref<6x768x2048xf32> {cinm.static}, %arg12: memref<6x2048x768xf32> {cinm.static}, %arg13: memref<6x768xf32> {cinm.static}, %arg14: memref<768xf32> {cinm.static}, %arg15: memref<32000x768xf32> {cinm.static}) -> memref<32000xf32> attributes {cinm.available_platforms = [#upmem]} {
    %0 = cinm.compute_block (%arg16 = %arg4 : memref<32000x768xf32>, %arg17 = %arg0 : index) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %subview = memref.subview %arg16[%arg17, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%subview : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %1 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %0 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    cinm.compute_block (%arg16 = %arg4 : memref<32000x768xf32>, %arg17 = %arg0 : index, %arg18 = %arg5 : memref<6x768xf32>, %arg19 = %1 : f32, %arg20 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[%arg17, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      %subview_100 = memref.subview %arg18[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%subview, %arg19, %subview_100 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1]>>) outs(%arg20 : memref<768xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %319 = arith.mulf %in, %in_101 : f32
        %320 = arith.mulf %319, %in_102 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg6 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg17 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %2 = cinm.compute_block (%arg16 = %arg7 : memref<6x768x768xf32>, %arg17 = %alloc_0 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %arg17[0, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    cinm.compute_block (%arg16 = %arg8 : memref<6x768x768xf32>, %arg17 = %arg3 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %arg17[0, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %3:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %arg1 : index, %arg17 = %alloc : memref<768xf32>, %arg18 = %2 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %arg2 : memref<6x1024x768xf32>) -> f32, index attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_100 = arith.constant 1.000000e+04 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %319 = arith.index_cast %arg16 : index to i64
      %320 = arith.uitofp %319 : i64 to f32
      scf.for %arg20 = %c0 to %c768 step %c2 {
        %322 = arith.remui %arg20, %c48 : index
        %323 = arith.index_cast %322 : index to i64
        %324 = arith.uitofp %323 : i64 to f32
        %325 = arith.divf %324, %cst : f32
        %326 = math.powf %cst_100, %325 : f32
        %327 = arith.divf %cst_101, %326 : f32
        %328 = arith.mulf %320, %327 : f32
        %329 = math.cos %328 : f32
        %330 = math.sin %328 : f32
        %331 = arith.addi %arg20, %c1 : index
        %332 = memref.load %arg17[%arg20] : memref<768xf32>
        %333 = memref.load %arg17[%331] : memref<768xf32>
        %334 = arith.mulf %332, %329 : f32
        %335 = arith.mulf %333, %330 : f32
        %336 = arith.subf %334, %335 : f32
        memref.store %336, %arg17[%arg20] : memref<768xf32>
        %337 = arith.mulf %332, %330 : f32
        %338 = arith.mulf %333, %329 : f32
        %339 = arith.addf %337, %338 : f32
        memref.store %339, %arg17[%331] : memref<768xf32>
        %340 = arith.cmpi ult, %arg20, %c768 : index
        scf.if %340 {
          %341 = memref.load %arg18[%arg20] : memref<768xf32, strided<[1], offset: ?>>
          %342 = memref.load %arg18[%331] : memref<768xf32, strided<[1], offset: ?>>
          %343 = arith.mulf %341, %329 : f32
          %344 = arith.mulf %342, %330 : f32
          %345 = arith.subf %343, %344 : f32
          memref.store %345, %arg18[%arg20] : memref<768xf32, strided<[1], offset: ?>>
          %346 = arith.mulf %341, %330 : f32
          %347 = arith.mulf %342, %329 : f32
          %348 = arith.addf %346, %347 : f32
          memref.store %348, %arg18[%331] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview = memref.subview %arg19[0, %arg16, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %arg18, %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      %321 = arith.addi %arg16, %c1 : index
      cinm.yield %320, %321 : f32, index
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1]>>
      %subview_101 = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1]>>, memref<48xf32, strided<[1]>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %4 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %4 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %5 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %6:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1]>> into memref<1x48xf32>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %6#1 : memref<1x48xf32>, %arg18 = %6#0 : memref<1x1024xf32>, %arg19 = %5 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1]>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1]>>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %6#1 : memref<1x48xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      memref.copy %collapse_shape, %subview : memref<48xf32> to memref<48xf32, strided<[1]>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 48>>
      %subview_101 = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 48>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %7 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %7 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %8 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_2 : memref<768xf32> to memref<768xf32>
    %9:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_2 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %9#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg18 = %9#0 : memref<1x1024xf32>, %arg19 = %8 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 48>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 48>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %9#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 96>>
      %subview_101 = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 96>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %10 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %10 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %11 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_3 : memref<768xf32> to memref<768xf32>
    %12:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_3 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %12#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg18 = %12#0 : memref<1x1024xf32>, %arg19 = %11 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 96>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 96>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %12#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 144>>
      %subview_101 = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 144>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %13 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %13 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %14 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_4 : memref<768xf32> to memref<768xf32>
    %15:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_4 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %15#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg18 = %15#0 : memref<1x1024xf32>, %arg19 = %14 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 144>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 144>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %15#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 192>>
      %subview_101 = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 192>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %16 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %16 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %17 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_5 : memref<768xf32> to memref<768xf32>
    %18:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_5 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %18#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg18 = %18#0 : memref<1x1024xf32>, %arg19 = %17 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 192>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 192>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %18#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 240>>
      %subview_101 = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 240>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %19 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %19 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %20 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_6 : memref<768xf32> to memref<768xf32>
    %21:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_6 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %21#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg18 = %21#0 : memref<1x1024xf32>, %arg19 = %20 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 240>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 240>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %21#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 288>>
      %subview_101 = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 288>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %22 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %22 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %23 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_7 : memref<768xf32> to memref<768xf32>
    %24:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_7 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %24#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg18 = %24#0 : memref<1x1024xf32>, %arg19 = %23 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 288>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 288>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %24#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 336>>
      %subview_101 = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 336>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %25 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %25 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %26 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_8 : memref<768xf32> to memref<768xf32>
    %27:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_8 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %27#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg18 = %27#0 : memref<1x1024xf32>, %arg19 = %26 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 336>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 336>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %27#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 384>>
      %subview_101 = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 384>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %28 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %28 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %29 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_9 : memref<768xf32> to memref<768xf32>
    %30:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_9 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %30#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg18 = %30#0 : memref<1x1024xf32>, %arg19 = %29 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 384>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 384>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %30#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 432>>
      %subview_101 = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 432>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %31 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %31 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %32 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_10 : memref<768xf32> to memref<768xf32>
    %33:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_10 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %33#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg18 = %33#0 : memref<1x1024xf32>, %arg19 = %32 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 432>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 432>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %33#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 480>>
      %subview_101 = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 480>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %34 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %34 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %35 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_11 : memref<768xf32> to memref<768xf32>
    %36:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_11 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %36#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg18 = %36#0 : memref<1x1024xf32>, %arg19 = %35 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 480>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 480>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %36#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 528>>
      %subview_101 = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 528>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %37 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %37 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %38 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_12 : memref<768xf32> to memref<768xf32>
    %39:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_12 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %39#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg18 = %39#0 : memref<1x1024xf32>, %arg19 = %38 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 528>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 528>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %39#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 576>>
      %subview_101 = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 576>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %40 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %40 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %41 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_13 : memref<768xf32> to memref<768xf32>
    %42:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_13 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %42#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg18 = %42#0 : memref<1x1024xf32>, %arg19 = %41 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 576>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 576>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %42#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 624>>
      %subview_101 = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 624>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %43 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %43 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %44 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_14 : memref<768xf32> to memref<768xf32>
    %45:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_14 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %45#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg18 = %45#0 : memref<1x1024xf32>, %arg19 = %44 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 624>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 624>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %45#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 672>>
      %subview_101 = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 672>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %46 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %46 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %47 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_15 : memref<768xf32> to memref<768xf32>
    %48:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_15 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %48#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg18 = %48#0 : memref<1x1024xf32>, %arg19 = %47 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 672>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 672>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %48#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 720>>
      %subview_101 = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 720>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %49 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %49 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %50 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_16 : memref<768xf32> to memref<768xf32>
    %51:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_16 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %51#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg18 = %51#0 : memref<1x1024xf32>, %arg19 = %50 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 720>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 720>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %51#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %arg17, %arg18 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    %52 = cinm.compute_block (%arg16 = %arg9 : memref<6x768x768xf32>, %arg17 = %arg4 : memref<32000x768xf32>, %arg18 = %arg0 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
      %subview_100 = memref.subview %arg17[%arg18, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_101 : f32
        %320 = arith.addf %out, %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    %53 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %54 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %53 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg13 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %54 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1]>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc : memref<768xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      memref.copy %arg16, %arg17 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    cinm.compute_block (%arg16 = %arg10 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg12 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg11 : memref<6x768x2048xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc_17 : memref<2048xf32>, %arg19 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_2 : memref<f32>
      %subview = memref.subview %arg16[0, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1]>>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17, %319 : memref<768x2048xf32, strided<[2048, 1]>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
        %320 = arith.negf %out : f32
        %321 = math.exp %320 : f32
        %322 = arith.addf %321, %in_101 : f32
        %323 = arith.divf %in_101, %322 : f32
        %324 = arith.mulf %out, %323 : f32
        %325 = arith.mulf %324, %in_100 : f32
        %326 = arith.mulf %in, %325 : f32
        %327 = arith.addf %out_102, %326 : f32
        linalg.yield %325, %327 : f32, f32
      }
      cinm.yield
    }
    %55 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %56 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %55 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg5 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %56 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 768>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg6 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%arg17 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %57 = cinm.compute_block (%arg16 = %arg7 : memref<6x768x768xf32>, %arg17 = %alloc_18 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
      %subview_100 = memref.subview %arg17[1, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    cinm.compute_block (%arg16 = %arg8 : memref<6x768x768xf32>, %arg17 = %arg3 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
      %subview_100 = memref.subview %arg17[1, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#0 : f32, %arg17 = %alloc : memref<768xf32>, %arg18 = %57 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %arg2 : memref<6x1024x768xf32>, %arg20 = %arg1 : index) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_100 = arith.constant 1.000000e+04 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg21 = %c0 to %c768 step %c2 {
        %319 = arith.remui %arg21, %c48 : index
        %320 = arith.index_cast %319 : index to i64
        %321 = arith.uitofp %320 : i64 to f32
        %322 = arith.divf %321, %cst : f32
        %323 = math.powf %cst_100, %322 : f32
        %324 = arith.divf %cst_101, %323 : f32
        %325 = arith.mulf %arg16, %324 : f32
        %326 = math.cos %325 : f32
        %327 = math.sin %325 : f32
        %328 = arith.addi %arg21, %c1 : index
        %329 = memref.load %arg17[%arg21] : memref<768xf32>
        %330 = memref.load %arg17[%328] : memref<768xf32>
        %331 = arith.mulf %329, %326 : f32
        %332 = arith.mulf %330, %327 : f32
        %333 = arith.subf %331, %332 : f32
        memref.store %333, %arg17[%arg21] : memref<768xf32>
        %334 = arith.mulf %329, %327 : f32
        %335 = arith.mulf %330, %326 : f32
        %336 = arith.addf %334, %335 : f32
        memref.store %336, %arg17[%328] : memref<768xf32>
        %337 = arith.cmpi ult, %arg21, %c768 : index
        scf.if %337 {
          %338 = memref.load %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %339 = memref.load %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
          %340 = arith.mulf %338, %326 : f32
          %341 = arith.mulf %339, %327 : f32
          %342 = arith.subf %340, %341 : f32
          memref.store %342, %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %343 = arith.mulf %338, %327 : f32
          %344 = arith.mulf %339, %326 : f32
          %345 = arith.addf %343, %344 : f32
          memref.store %345, %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview = memref.subview %arg19[1, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %arg18, %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786432>>
      %subview_101 = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786432>>, memref<48xf32, strided<[1]>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %58 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %58 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %59 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %60 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>) -> memref<1x1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      cinm.yield %expand_shape : memref<1x1024xf32>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %6#1 : memref<1x48xf32>, %arg18 = %60 : memref<1x1024xf32>, %arg19 = %59 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786432>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786432>>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %6#1 : memref<1x48xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      memref.copy %collapse_shape, %subview : memref<48xf32> to memref<48xf32, strided<[1]>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786480>>
      %subview_101 = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786480>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %61 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %61 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %62 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_19 : memref<768xf32> to memref<768xf32>
    %63:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_19 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %63#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg18 = %63#0 : memref<1x1024xf32>, %arg19 = %62 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786480>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786480>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %63#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786528>>
      %subview_101 = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786528>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %64 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %64 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %65 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_20 : memref<768xf32> to memref<768xf32>
    %66:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_20 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %66#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg18 = %66#0 : memref<1x1024xf32>, %arg19 = %65 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786528>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786528>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %66#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786576>>
      %subview_101 = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786576>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %67 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %67 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %68 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_21 : memref<768xf32> to memref<768xf32>
    %69:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_21 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %69#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg18 = %69#0 : memref<1x1024xf32>, %arg19 = %68 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786576>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786576>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %69#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786624>>
      %subview_101 = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786624>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %70 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %70 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %71 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_22 : memref<768xf32> to memref<768xf32>
    %72:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_22 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %72#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg18 = %72#0 : memref<1x1024xf32>, %arg19 = %71 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786624>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786624>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %72#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786672>>
      %subview_101 = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786672>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %73 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %73 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %74 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_23 : memref<768xf32> to memref<768xf32>
    %75:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_23 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %75#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg18 = %75#0 : memref<1x1024xf32>, %arg19 = %74 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786672>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786672>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %75#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786720>>
      %subview_101 = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786720>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %76 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %76 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %77 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_24 : memref<768xf32> to memref<768xf32>
    %78:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_24 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %78#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg18 = %78#0 : memref<1x1024xf32>, %arg19 = %77 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786720>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786720>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %78#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786768>>
      %subview_101 = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786768>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %79 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %79 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %80 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_25 : memref<768xf32> to memref<768xf32>
    %81:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_25 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %81#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg18 = %81#0 : memref<1x1024xf32>, %arg19 = %80 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786768>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786768>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %81#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786816>>
      %subview_101 = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786816>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %82 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %82 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %83 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_26 : memref<768xf32> to memref<768xf32>
    %84:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_26 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %84#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg18 = %84#0 : memref<1x1024xf32>, %arg19 = %83 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786816>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786816>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %84#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786864>>
      %subview_101 = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786864>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %85 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %85 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %86 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_27 : memref<768xf32> to memref<768xf32>
    %87:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_27 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %87#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg18 = %87#0 : memref<1x1024xf32>, %arg19 = %86 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786864>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786864>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %87#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786912>>
      %subview_101 = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786912>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %88 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %88 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %89 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_28 : memref<768xf32> to memref<768xf32>
    %90:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_28 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %90#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg18 = %90#0 : memref<1x1024xf32>, %arg19 = %89 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786912>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786912>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %90#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786960>>
      %subview_101 = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 786960>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %91 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %91 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %92 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_29 : memref<768xf32> to memref<768xf32>
    %93:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_29 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %93#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg18 = %93#0 : memref<1x1024xf32>, %arg19 = %92 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786960>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786960>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %93#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787008>>
      %subview_101 = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 787008>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %94 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %94 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %95 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_30 : memref<768xf32> to memref<768xf32>
    %96:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_30 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %96#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg18 = %96#0 : memref<1x1024xf32>, %arg19 = %95 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787008>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787008>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %96#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787056>>
      %subview_101 = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 787056>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %97 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %97 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %98 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_31 : memref<768xf32> to memref<768xf32>
    %99:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_31 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %99#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg18 = %99#0 : memref<1x1024xf32>, %arg19 = %98 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787056>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787056>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %99#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787104>>
      %subview_101 = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 787104>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %100 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %100 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %101 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_32 : memref<768xf32> to memref<768xf32>
    %102:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_32 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %102#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg18 = %102#0 : memref<1x1024xf32>, %arg19 = %101 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787104>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787104>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %102#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787152>>
      %subview_101 = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 787152>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %103 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %103 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %104 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_33 : memref<768xf32> to memref<768xf32>
    %105:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_33 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %105#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg18 = %105#0 : memref<1x1024xf32>, %arg19 = %104 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787152>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787152>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %105#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %arg17, %arg18 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg9 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.addf %out, %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %106 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %107 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %106 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg13 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %107 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 768>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc : memref<768xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      memref.copy %arg16, %arg17 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg10 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg12 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg11 : memref<6x768x2048xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc_17 : memref<2048xf32>, %arg19 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_2 : memref<f32>
      %subview = memref.subview %arg16[1, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17, %319 : memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
        %320 = arith.negf %out : f32
        %321 = math.exp %320 : f32
        %322 = arith.addf %321, %in_101 : f32
        %323 = arith.divf %in_101, %322 : f32
        %324 = arith.mulf %out, %323 : f32
        %325 = arith.mulf %324, %in_100 : f32
        %326 = arith.mulf %in, %325 : f32
        %327 = arith.addf %out_102, %326 : f32
        linalg.yield %325, %327 : f32, f32
      }
      cinm.yield
    }
    %108 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %109 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %108 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg5 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %109 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 1536>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg6 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%arg17 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %110 = cinm.compute_block (%arg16 = %arg7 : memref<6x768x768xf32>, %arg17 = %alloc_34 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
      %subview_100 = memref.subview %arg17[2, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    cinm.compute_block (%arg16 = %arg8 : memref<6x768x768xf32>, %arg17 = %arg3 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
      %subview_100 = memref.subview %arg17[2, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#0 : f32, %arg17 = %alloc : memref<768xf32>, %arg18 = %110 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %arg2 : memref<6x1024x768xf32>, %arg20 = %arg1 : index) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_100 = arith.constant 1.000000e+04 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg21 = %c0 to %c768 step %c2 {
        %319 = arith.remui %arg21, %c48 : index
        %320 = arith.index_cast %319 : index to i64
        %321 = arith.uitofp %320 : i64 to f32
        %322 = arith.divf %321, %cst : f32
        %323 = math.powf %cst_100, %322 : f32
        %324 = arith.divf %cst_101, %323 : f32
        %325 = arith.mulf %arg16, %324 : f32
        %326 = math.cos %325 : f32
        %327 = math.sin %325 : f32
        %328 = arith.addi %arg21, %c1 : index
        %329 = memref.load %arg17[%arg21] : memref<768xf32>
        %330 = memref.load %arg17[%328] : memref<768xf32>
        %331 = arith.mulf %329, %326 : f32
        %332 = arith.mulf %330, %327 : f32
        %333 = arith.subf %331, %332 : f32
        memref.store %333, %arg17[%arg21] : memref<768xf32>
        %334 = arith.mulf %329, %327 : f32
        %335 = arith.mulf %330, %326 : f32
        %336 = arith.addf %334, %335 : f32
        memref.store %336, %arg17[%328] : memref<768xf32>
        %337 = arith.cmpi ult, %arg21, %c768 : index
        scf.if %337 {
          %338 = memref.load %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %339 = memref.load %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
          %340 = arith.mulf %338, %326 : f32
          %341 = arith.mulf %339, %327 : f32
          %342 = arith.subf %340, %341 : f32
          memref.store %342, %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %343 = arith.mulf %338, %327 : f32
          %344 = arith.mulf %339, %326 : f32
          %345 = arith.addf %343, %344 : f32
          memref.store %345, %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview = memref.subview %arg19[2, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %arg18, %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572864>>
      %subview_101 = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1572864>>, memref<48xf32, strided<[1]>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %111 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %111 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %112 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %6#1 : memref<1x48xf32>, %arg18 = %60 : memref<1x1024xf32>, %arg19 = %112 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572864>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1572864>>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %6#1 : memref<1x48xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      memref.copy %collapse_shape, %subview : memref<48xf32> to memref<48xf32, strided<[1]>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572912>>
      %subview_101 = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1572912>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %113 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %113 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %114 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_35 : memref<768xf32> to memref<768xf32>
    %115:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_35 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %115#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg18 = %115#0 : memref<1x1024xf32>, %arg19 = %114 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572912>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1572912>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %115#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572960>>
      %subview_101 = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1572960>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %116 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %116 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %117 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_36 : memref<768xf32> to memref<768xf32>
    %118:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_36 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %118#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg18 = %118#0 : memref<1x1024xf32>, %arg19 = %117 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572960>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1572960>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %118#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573008>>
      %subview_101 = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573008>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %119 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %119 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %120 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_37 : memref<768xf32> to memref<768xf32>
    %121:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_37 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %121#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg18 = %121#0 : memref<1x1024xf32>, %arg19 = %120 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573008>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573008>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %121#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573056>>
      %subview_101 = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573056>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %122 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %122 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %123 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_38 : memref<768xf32> to memref<768xf32>
    %124:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_38 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %124#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg18 = %124#0 : memref<1x1024xf32>, %arg19 = %123 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573056>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573056>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %124#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573104>>
      %subview_101 = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573104>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %125 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %125 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %126 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_39 : memref<768xf32> to memref<768xf32>
    %127:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_39 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %127#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg18 = %127#0 : memref<1x1024xf32>, %arg19 = %126 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573104>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573104>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %127#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573152>>
      %subview_101 = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573152>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %128 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %128 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %129 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_40 : memref<768xf32> to memref<768xf32>
    %130:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_40 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %130#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg18 = %130#0 : memref<1x1024xf32>, %arg19 = %129 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573152>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573152>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %130#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573200>>
      %subview_101 = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573200>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %131 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %131 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %132 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_41 : memref<768xf32> to memref<768xf32>
    %133:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_41 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %133#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg18 = %133#0 : memref<1x1024xf32>, %arg19 = %132 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573200>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573200>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %133#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573248>>
      %subview_101 = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573248>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %134 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %134 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %135 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_42 : memref<768xf32> to memref<768xf32>
    %136:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_42 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %136#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg18 = %136#0 : memref<1x1024xf32>, %arg19 = %135 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573248>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573248>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %136#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573296>>
      %subview_101 = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573296>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %137 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %137 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %138 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_43 : memref<768xf32> to memref<768xf32>
    %139:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_43 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %139#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg18 = %139#0 : memref<1x1024xf32>, %arg19 = %138 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573296>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573296>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %139#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573344>>
      %subview_101 = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573344>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %140 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %140 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %141 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_44 : memref<768xf32> to memref<768xf32>
    %142:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_44 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %142#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg18 = %142#0 : memref<1x1024xf32>, %arg19 = %141 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573344>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573344>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %142#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573392>>
      %subview_101 = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573392>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %143 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %143 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %144 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_45 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_45 : memref<768xf32> to memref<768xf32>
    %145:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_45 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %145#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg18 = %145#0 : memref<1x1024xf32>, %arg19 = %144 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573392>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573392>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %145#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573440>>
      %subview_101 = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573440>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %146 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %146 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %147 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_46 : memref<768xf32> to memref<768xf32>
    %148:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_46 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %148#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg18 = %148#0 : memref<1x1024xf32>, %arg19 = %147 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573440>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573440>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %148#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573488>>
      %subview_101 = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573488>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %149 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %149 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %150 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_47 : memref<768xf32> to memref<768xf32>
    %151:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_47 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %151#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg18 = %151#0 : memref<1x1024xf32>, %arg19 = %150 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573488>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573488>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %151#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573536>>
      %subview_101 = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573536>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %152 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %152 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %153 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_48 : memref<768xf32> to memref<768xf32>
    %154:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_48 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %154#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg18 = %154#0 : memref<1x1024xf32>, %arg19 = %153 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573536>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573536>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %154#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573584>>
      %subview_101 = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 1573584>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %155 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %155 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %156 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_49 : memref<768xf32> to memref<768xf32>
    %157:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_49 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %157#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg18 = %157#0 : memref<1x1024xf32>, %arg19 = %156 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573584>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573584>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %157#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %arg17, %arg18 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg9 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.addf %out, %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %158 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %159 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %158 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg13 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %159 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 1536>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc : memref<768xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      memref.copy %arg16, %arg17 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg10 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg12 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg11 : memref<6x768x2048xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc_17 : memref<2048xf32>, %arg19 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_2 : memref<f32>
      %subview = memref.subview %arg16[2, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17, %319 : memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
        %320 = arith.negf %out : f32
        %321 = math.exp %320 : f32
        %322 = arith.addf %321, %in_101 : f32
        %323 = arith.divf %in_101, %322 : f32
        %324 = arith.mulf %out, %323 : f32
        %325 = arith.mulf %324, %in_100 : f32
        %326 = arith.mulf %in, %325 : f32
        %327 = arith.addf %out_102, %326 : f32
        linalg.yield %325, %327 : f32, f32
      }
      cinm.yield
    }
    %160 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %161 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %160 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg5 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %161 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 2304>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg6 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%arg17 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %alloc_50 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %162 = cinm.compute_block (%arg16 = %arg7 : memref<6x768x768xf32>, %arg17 = %alloc_50 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
      %subview_100 = memref.subview %arg17[3, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    cinm.compute_block (%arg16 = %arg8 : memref<6x768x768xf32>, %arg17 = %arg3 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
      %subview_100 = memref.subview %arg17[3, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#0 : f32, %arg17 = %alloc : memref<768xf32>, %arg18 = %162 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %arg2 : memref<6x1024x768xf32>, %arg20 = %arg1 : index) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_100 = arith.constant 1.000000e+04 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg21 = %c0 to %c768 step %c2 {
        %319 = arith.remui %arg21, %c48 : index
        %320 = arith.index_cast %319 : index to i64
        %321 = arith.uitofp %320 : i64 to f32
        %322 = arith.divf %321, %cst : f32
        %323 = math.powf %cst_100, %322 : f32
        %324 = arith.divf %cst_101, %323 : f32
        %325 = arith.mulf %arg16, %324 : f32
        %326 = math.cos %325 : f32
        %327 = math.sin %325 : f32
        %328 = arith.addi %arg21, %c1 : index
        %329 = memref.load %arg17[%arg21] : memref<768xf32>
        %330 = memref.load %arg17[%328] : memref<768xf32>
        %331 = arith.mulf %329, %326 : f32
        %332 = arith.mulf %330, %327 : f32
        %333 = arith.subf %331, %332 : f32
        memref.store %333, %arg17[%arg21] : memref<768xf32>
        %334 = arith.mulf %329, %327 : f32
        %335 = arith.mulf %330, %326 : f32
        %336 = arith.addf %334, %335 : f32
        memref.store %336, %arg17[%328] : memref<768xf32>
        %337 = arith.cmpi ult, %arg21, %c768 : index
        scf.if %337 {
          %338 = memref.load %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %339 = memref.load %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
          %340 = arith.mulf %338, %326 : f32
          %341 = arith.mulf %339, %327 : f32
          %342 = arith.subf %340, %341 : f32
          memref.store %342, %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %343 = arith.mulf %338, %327 : f32
          %344 = arith.mulf %339, %326 : f32
          %345 = arith.addf %343, %344 : f32
          memref.store %345, %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview = memref.subview %arg19[3, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %arg18, %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359296>>
      %subview_101 = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359296>>, memref<48xf32, strided<[1]>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %163 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %163 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %164 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %6#1 : memref<1x48xf32>, %arg18 = %60 : memref<1x1024xf32>, %arg19 = %164 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359296>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359296>>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %6#1 : memref<1x48xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      memref.copy %collapse_shape, %subview : memref<48xf32> to memref<48xf32, strided<[1]>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359344>>
      %subview_101 = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359344>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %165 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %165 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %166 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_51 : memref<768xf32> to memref<768xf32>
    %167:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_51 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %167#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg18 = %167#0 : memref<1x1024xf32>, %arg19 = %166 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359344>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359344>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %167#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359392>>
      %subview_101 = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359392>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %168 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %168 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %169 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_52 : memref<768xf32> to memref<768xf32>
    %170:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_52 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %170#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg18 = %170#0 : memref<1x1024xf32>, %arg19 = %169 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359392>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359392>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %170#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359440>>
      %subview_101 = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359440>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %171 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %171 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %172 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_53 : memref<768xf32> to memref<768xf32>
    %173:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_53 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %173#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg18 = %173#0 : memref<1x1024xf32>, %arg19 = %172 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359440>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359440>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %173#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359488>>
      %subview_101 = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359488>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %174 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %174 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %175 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_54 : memref<768xf32> to memref<768xf32>
    %176:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_54 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %176#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg18 = %176#0 : memref<1x1024xf32>, %arg19 = %175 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359488>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359488>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %176#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359536>>
      %subview_101 = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359536>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %177 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %177 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %178 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_55 : memref<768xf32> to memref<768xf32>
    %179:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_55 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %179#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg18 = %179#0 : memref<1x1024xf32>, %arg19 = %178 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359536>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359536>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %179#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359584>>
      %subview_101 = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359584>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %180 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %180 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %181 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_56 : memref<768xf32> to memref<768xf32>
    %182:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_56 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %182#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg18 = %182#0 : memref<1x1024xf32>, %arg19 = %181 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359584>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359584>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %182#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359632>>
      %subview_101 = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359632>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %183 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %183 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %184 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_57 : memref<768xf32> to memref<768xf32>
    %185:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_57 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %185#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg18 = %185#0 : memref<1x1024xf32>, %arg19 = %184 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359632>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359632>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %185#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359680>>
      %subview_101 = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359680>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %186 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %186 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %187 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_58 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_58 : memref<768xf32> to memref<768xf32>
    %188:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_58 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %188#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg18 = %188#0 : memref<1x1024xf32>, %arg19 = %187 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359680>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359680>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %188#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359728>>
      %subview_101 = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359728>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %189 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %189 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %190 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_59 : memref<768xf32> to memref<768xf32>
    %191:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_59 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %191#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg18 = %191#0 : memref<1x1024xf32>, %arg19 = %190 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359728>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359728>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %191#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359776>>
      %subview_101 = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359776>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %192 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %192 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %193 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_60 : memref<768xf32> to memref<768xf32>
    %194:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_60 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %194#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg18 = %194#0 : memref<1x1024xf32>, %arg19 = %193 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359776>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359776>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %194#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359824>>
      %subview_101 = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359824>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %195 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %195 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %196 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_61 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_61 : memref<768xf32> to memref<768xf32>
    %197:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_61 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %197#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg18 = %197#0 : memref<1x1024xf32>, %arg19 = %196 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359824>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359824>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %197#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359872>>
      %subview_101 = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359872>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %198 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %198 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %199 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_62 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_62 : memref<768xf32> to memref<768xf32>
    %200:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_62 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %200#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg18 = %200#0 : memref<1x1024xf32>, %arg19 = %199 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359872>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359872>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %200#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359920>>
      %subview_101 = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359920>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %201 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %201 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %202 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_63 : memref<768xf32> to memref<768xf32>
    %203:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_63 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %203#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg18 = %203#0 : memref<1x1024xf32>, %arg19 = %202 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359920>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359920>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %203#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359968>>
      %subview_101 = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2359968>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %204 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %204 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %205 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_64 : memref<768xf32> to memref<768xf32>
    %206:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_64 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %206#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg18 = %206#0 : memref<1x1024xf32>, %arg19 = %205 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359968>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359968>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %206#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2360016>>
      %subview_101 = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 2360016>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %207 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %207 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %208 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_65 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_65 : memref<768xf32> to memref<768xf32>
    %209:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_65 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %209#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg18 = %209#0 : memref<1x1024xf32>, %arg19 = %208 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2360016>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2360016>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %209#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %arg17, %arg18 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg9 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.addf %out, %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %210 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %211 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %210 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg13 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %211 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 2304>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc : memref<768xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      memref.copy %arg16, %arg17 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg10 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg12 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg11 : memref<6x768x2048xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc_17 : memref<2048xf32>, %arg19 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_2 : memref<f32>
      %subview = memref.subview %arg16[3, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17, %319 : memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
        %320 = arith.negf %out : f32
        %321 = math.exp %320 : f32
        %322 = arith.addf %321, %in_101 : f32
        %323 = arith.divf %in_101, %322 : f32
        %324 = arith.mulf %out, %323 : f32
        %325 = arith.mulf %324, %in_100 : f32
        %326 = arith.mulf %in, %325 : f32
        %327 = arith.addf %out_102, %326 : f32
        linalg.yield %325, %327 : f32, f32
      }
      cinm.yield
    }
    %212 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %213 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %212 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg5 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %213 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3072>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg6 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%arg17 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %alloc_66 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %214 = cinm.compute_block (%arg16 = %arg7 : memref<6x768x768xf32>, %arg17 = %alloc_66 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %arg17[4, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    cinm.compute_block (%arg16 = %arg8 : memref<6x768x768xf32>, %arg17 = %arg3 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
      %subview_100 = memref.subview %arg17[4, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#0 : f32, %arg17 = %alloc : memref<768xf32>, %arg18 = %214 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %arg2 : memref<6x1024x768xf32>, %arg20 = %arg1 : index) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_100 = arith.constant 1.000000e+04 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg21 = %c0 to %c768 step %c2 {
        %319 = arith.remui %arg21, %c48 : index
        %320 = arith.index_cast %319 : index to i64
        %321 = arith.uitofp %320 : i64 to f32
        %322 = arith.divf %321, %cst : f32
        %323 = math.powf %cst_100, %322 : f32
        %324 = arith.divf %cst_101, %323 : f32
        %325 = arith.mulf %arg16, %324 : f32
        %326 = math.cos %325 : f32
        %327 = math.sin %325 : f32
        %328 = arith.addi %arg21, %c1 : index
        %329 = memref.load %arg17[%arg21] : memref<768xf32>
        %330 = memref.load %arg17[%328] : memref<768xf32>
        %331 = arith.mulf %329, %326 : f32
        %332 = arith.mulf %330, %327 : f32
        %333 = arith.subf %331, %332 : f32
        memref.store %333, %arg17[%arg21] : memref<768xf32>
        %334 = arith.mulf %329, %327 : f32
        %335 = arith.mulf %330, %326 : f32
        %336 = arith.addf %334, %335 : f32
        memref.store %336, %arg17[%328] : memref<768xf32>
        %337 = arith.cmpi ult, %arg21, %c768 : index
        scf.if %337 {
          %338 = memref.load %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %339 = memref.load %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
          %340 = arith.mulf %338, %326 : f32
          %341 = arith.mulf %339, %327 : f32
          %342 = arith.subf %340, %341 : f32
          memref.store %342, %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %343 = arith.mulf %338, %327 : f32
          %344 = arith.mulf %339, %326 : f32
          %345 = arith.addf %343, %344 : f32
          memref.store %345, %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview = memref.subview %arg19[4, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %arg18, %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145728>>
      %subview_101 = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3145728>>, memref<48xf32, strided<[1]>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %215 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %215 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %216 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %6#1 : memref<1x48xf32>, %arg18 = %60 : memref<1x1024xf32>, %arg19 = %216 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145728>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145728>>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %6#1 : memref<1x48xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      memref.copy %collapse_shape, %subview : memref<48xf32> to memref<48xf32, strided<[1]>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145776>>
      %subview_101 = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3145776>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %217 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %217 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %218 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_67 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_67 : memref<768xf32> to memref<768xf32>
    %219:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_67 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %219#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg18 = %219#0 : memref<1x1024xf32>, %arg19 = %218 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145776>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145776>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %219#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145824>>
      %subview_101 = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3145824>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %220 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %220 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %221 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_68 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_68 : memref<768xf32> to memref<768xf32>
    %222:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_68 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %222#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg18 = %222#0 : memref<1x1024xf32>, %arg19 = %221 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145824>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145824>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %222#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145872>>
      %subview_101 = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3145872>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %223 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %223 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %224 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_69 : memref<768xf32> to memref<768xf32>
    %225:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_69 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %225#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg18 = %225#0 : memref<1x1024xf32>, %arg19 = %224 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145872>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145872>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %225#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145920>>
      %subview_101 = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3145920>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %226 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %226 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %227 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_70 : memref<768xf32> to memref<768xf32>
    %228:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_70 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %228#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg18 = %228#0 : memref<1x1024xf32>, %arg19 = %227 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145920>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145920>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %228#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145968>>
      %subview_101 = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3145968>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %229 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %229 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %230 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_71 : memref<768xf32> to memref<768xf32>
    %231:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_71 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %231#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg18 = %231#0 : memref<1x1024xf32>, %arg19 = %230 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145968>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145968>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %231#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146016>>
      %subview_101 = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146016>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %232 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %232 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %233 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_72 : memref<768xf32> to memref<768xf32>
    %234:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_72 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %234#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg18 = %234#0 : memref<1x1024xf32>, %arg19 = %233 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146016>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146016>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %234#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146064>>
      %subview_101 = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146064>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %235 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %235 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %236 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_73 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_73 : memref<768xf32> to memref<768xf32>
    %237:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_73 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %237#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg18 = %237#0 : memref<1x1024xf32>, %arg19 = %236 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146064>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146064>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %237#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146112>>
      %subview_101 = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146112>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %238 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %238 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %239 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_74 : memref<768xf32> to memref<768xf32>
    %240:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_74 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %240#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg18 = %240#0 : memref<1x1024xf32>, %arg19 = %239 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146112>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146112>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %240#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146160>>
      %subview_101 = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146160>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %241 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %241 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %242 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_75 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_75 : memref<768xf32> to memref<768xf32>
    %243:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_75 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %243#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg18 = %243#0 : memref<1x1024xf32>, %arg19 = %242 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146160>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146160>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %243#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146208>>
      %subview_101 = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146208>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %244 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %244 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %245 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_76 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_76 : memref<768xf32> to memref<768xf32>
    %246:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_76 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %246#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg18 = %246#0 : memref<1x1024xf32>, %arg19 = %245 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146208>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146208>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %246#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146256>>
      %subview_101 = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146256>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %247 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %247 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %248 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_77 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_77 : memref<768xf32> to memref<768xf32>
    %249:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_77 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %249#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg18 = %249#0 : memref<1x1024xf32>, %arg19 = %248 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146256>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146256>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %249#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146304>>
      %subview_101 = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146304>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %250 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %250 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %251 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_78 : memref<768xf32> to memref<768xf32>
    %252:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_78 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %252#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg18 = %252#0 : memref<1x1024xf32>, %arg19 = %251 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146304>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146304>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %252#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146352>>
      %subview_101 = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146352>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %253 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %253 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %254 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_79 : memref<768xf32> to memref<768xf32>
    %255:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_79 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %255#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg18 = %255#0 : memref<1x1024xf32>, %arg19 = %254 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146352>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146352>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %255#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146400>>
      %subview_101 = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146400>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %256 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %256 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %257 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_80 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_80 : memref<768xf32> to memref<768xf32>
    %258:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_80 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %258#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg18 = %258#0 : memref<1x1024xf32>, %arg19 = %257 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146400>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146400>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %258#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146448>>
      %subview_101 = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3146448>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %259 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %259 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %260 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_81 : memref<768xf32> to memref<768xf32>
    %261:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_81 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %261#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg18 = %261#0 : memref<1x1024xf32>, %arg19 = %260 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146448>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146448>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %261#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %arg17, %arg18 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg9 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.addf %out, %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %262 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %263 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %262 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg13 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %263 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3072>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc : memref<768xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      memref.copy %arg16, %arg17 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg10 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg12 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg11 : memref<6x768x2048xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc_17 : memref<2048xf32>, %arg19 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_2 : memref<f32>
      %subview = memref.subview %arg16[4, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17, %319 : memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
        %320 = arith.negf %out : f32
        %321 = math.exp %320 : f32
        %322 = arith.addf %321, %in_101 : f32
        %323 = arith.divf %in_101, %322 : f32
        %324 = arith.mulf %out, %323 : f32
        %325 = arith.mulf %324, %in_100 : f32
        %326 = arith.mulf %in, %325 : f32
        %327 = arith.addf %out_102, %326 : f32
        linalg.yield %325, %327 : f32, f32
      }
      cinm.yield
    }
    %264 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %265 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %264 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg5 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %265 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3840>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg6 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%arg17 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    %alloc_82 = memref.alloc() {alignment = 64 : i64} : memref<6x1024x768xf32>
    %266 = cinm.compute_block (%arg16 = %arg7 : memref<6x768x768xf32>, %arg17 = %alloc_82 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
      %subview_100 = memref.subview %arg17[5, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield %subview_100 : memref<768xf32, strided<[1], offset: ?>>
    }
    cinm.compute_block (%arg16 = %arg8 : memref<6x768x768xf32>, %arg17 = %arg3 : memref<6x1024x768xf32>, %arg18 = %arg1 : index, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
      %subview_100 = memref.subview %arg17[5, %arg18, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg19 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%subview_100 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_101: f32, %out: f32):
        %320 = arith.mulf %in, %in_101 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#0 : f32, %arg17 = %alloc : memref<768xf32>, %arg18 = %266 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %arg2 : memref<6x1024x768xf32>, %arg20 = %arg1 : index) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_100 = arith.constant 1.000000e+04 : f32
      %cst_101 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg21 = %c0 to %c768 step %c2 {
        %319 = arith.remui %arg21, %c48 : index
        %320 = arith.index_cast %319 : index to i64
        %321 = arith.uitofp %320 : i64 to f32
        %322 = arith.divf %321, %cst : f32
        %323 = math.powf %cst_100, %322 : f32
        %324 = arith.divf %cst_101, %323 : f32
        %325 = arith.mulf %arg16, %324 : f32
        %326 = math.cos %325 : f32
        %327 = math.sin %325 : f32
        %328 = arith.addi %arg21, %c1 : index
        %329 = memref.load %arg17[%arg21] : memref<768xf32>
        %330 = memref.load %arg17[%328] : memref<768xf32>
        %331 = arith.mulf %329, %326 : f32
        %332 = arith.mulf %330, %327 : f32
        %333 = arith.subf %331, %332 : f32
        memref.store %333, %arg17[%arg21] : memref<768xf32>
        %334 = arith.mulf %329, %327 : f32
        %335 = arith.mulf %330, %326 : f32
        %336 = arith.addf %334, %335 : f32
        memref.store %336, %arg17[%328] : memref<768xf32>
        %337 = arith.cmpi ult, %arg21, %c768 : index
        scf.if %337 {
          %338 = memref.load %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %339 = memref.load %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
          %340 = arith.mulf %338, %326 : f32
          %341 = arith.mulf %339, %327 : f32
          %342 = arith.subf %340, %341 : f32
          memref.store %342, %arg18[%arg21] : memref<768xf32, strided<[1], offset: ?>>
          %343 = arith.mulf %338, %327 : f32
          %344 = arith.mulf %339, %326 : f32
          %345 = arith.addf %343, %344 : f32
          memref.store %345, %arg18[%328] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      %subview = memref.subview %arg19[5, %arg20, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
      memref.copy %arg18, %subview : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932160>>
      %subview_101 = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932160>>, memref<48xf32, strided<[1]>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %267 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %267 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %268 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %6#1 : memref<1x48xf32>, %arg18 = %60 : memref<1x1024xf32>, %arg19 = %268 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932160>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932160>>) outs(%arg17 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %6#1 : memref<1x48xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
      %subview = memref.subview %arg17[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
      memref.copy %collapse_shape, %subview : memref<48xf32> to memref<48xf32, strided<[1]>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932208>>
      %subview_101 = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932208>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %269 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %269 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %270 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_83 : memref<768xf32> to memref<768xf32>
    %271:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_83 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %271#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg18 = %271#0 : memref<1x1024xf32>, %arg19 = %270 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932208>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932208>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %271#1 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
      %subview = memref.subview %arg17[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932256>>
      %subview_101 = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932256>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %272 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %272 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %273 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_84 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_84 : memref<768xf32> to memref<768xf32>
    %274:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_84 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %274#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg18 = %274#0 : memref<1x1024xf32>, %arg19 = %273 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932256>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932256>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %274#1 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
      %subview = memref.subview %arg17[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932304>>
      %subview_101 = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932304>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %275 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %275 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %276 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_85 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_85 : memref<768xf32> to memref<768xf32>
    %277:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_85 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %277#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg18 = %277#0 : memref<1x1024xf32>, %arg19 = %276 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932304>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932304>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %277#1 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
      %subview = memref.subview %arg17[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932352>>
      %subview_101 = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932352>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %278 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %278 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %279 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_86 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_86 : memref<768xf32> to memref<768xf32>
    %280:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_86 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %280#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg18 = %280#0 : memref<1x1024xf32>, %arg19 = %279 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932352>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932352>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %280#1 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
      %subview = memref.subview %arg17[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932400>>
      %subview_101 = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932400>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %281 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %281 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %282 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_87 : memref<768xf32> to memref<768xf32>
    %283:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_87 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %283#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg18 = %283#0 : memref<1x1024xf32>, %arg19 = %282 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932400>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932400>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %283#1 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
      %subview = memref.subview %arg17[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932448>>
      %subview_101 = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932448>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %284 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %284 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %285 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_88 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_88 : memref<768xf32> to memref<768xf32>
    %286:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_88 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %286#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg18 = %286#0 : memref<1x1024xf32>, %arg19 = %285 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932448>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932448>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %286#1 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
      %subview = memref.subview %arg17[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932496>>
      %subview_101 = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932496>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %287 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %287 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %288 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_89 : memref<768xf32> to memref<768xf32>
    %289:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_89 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %289#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg18 = %289#0 : memref<1x1024xf32>, %arg19 = %288 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932496>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932496>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %289#1 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
      %subview = memref.subview %arg17[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932544>>
      %subview_101 = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932544>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %290 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %290 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %291 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_90 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_90 : memref<768xf32> to memref<768xf32>
    %292:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_90 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %292#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg18 = %292#0 : memref<1x1024xf32>, %arg19 = %291 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932544>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932544>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %292#1 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
      %subview = memref.subview %arg17[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932592>>
      %subview_101 = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932592>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %293 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %293 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %294 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_91 : memref<768xf32> to memref<768xf32>
    %295:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_91 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %295#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg18 = %295#0 : memref<1x1024xf32>, %arg19 = %294 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932592>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932592>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %295#1 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
      %subview = memref.subview %arg17[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932640>>
      %subview_101 = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932640>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %296 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %296 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %297 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_92 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_92 : memref<768xf32> to memref<768xf32>
    %298:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_92 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %298#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg18 = %298#0 : memref<1x1024xf32>, %arg19 = %297 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932640>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932640>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %298#1 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
      %subview = memref.subview %arg17[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932688>>
      %subview_101 = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932688>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %299 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %299 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %300 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_93 : memref<768xf32> to memref<768xf32>
    %301:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_93 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %301#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg18 = %301#0 : memref<1x1024xf32>, %arg19 = %300 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932688>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932688>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %301#1 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
      %subview = memref.subview %arg17[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932736>>
      %subview_101 = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932736>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %302 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %302 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %303 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_94 : memref<768xf32> to memref<768xf32>
    %304:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_94 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %304#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg18 = %304#0 : memref<1x1024xf32>, %arg19 = %303 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932736>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932736>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %304#1 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
      %subview = memref.subview %arg17[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932784>>
      %subview_101 = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932784>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %305 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %305 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %306 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_95 : memref<768xf32> to memref<768xf32>
    %307:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_95 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %307#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg18 = %307#0 : memref<1x1024xf32>, %arg19 = %306 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932784>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932784>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %307#1 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
      %subview = memref.subview %arg17[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932832>>
      %subview_101 = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932832>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %308 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %308 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %309 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_96 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_96 : memref<768xf32> to memref<768xf32>
    %310:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_96 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %310#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg18 = %310#0 : memref<1x1024xf32>, %arg19 = %309 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932832>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932832>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %310#1 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
      %subview = memref.subview %arg17[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg2 : memref<6x1024x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932880>>
      %subview_101 = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg18 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 3932880>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg18 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in, %in_102 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %319 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.divf %in, %in_100 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %3#1 : index, %arg17 = %alloc_1 : memref<1024xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield
    }
    %311 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.maxnumf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %311 : f32) attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.subf %in, %in_100 : f32
        %320 = math.exp %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %312 = cinm.compute_block (%arg16 = %alloc_1 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<1024xf32>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.addf %in, %out : f32
        linalg.yield %320 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    memref.copy %alloc, %alloc_97 : memref<768xf32> to memref<768xf32>
    %313:2 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_1 : memref<1024xf32>, %arg17 = %alloc_97 : memref<768xf32>) -> memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expand_shape = memref.expand_shape %arg16 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      %expand_shape_100 = memref.expand_shape %subview [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
      cinm.yield %expand_shape, %expand_shape_100 : memref<1x1024xf32>, memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    cinm.compute_block (%arg16 = %arg3 : memref<6x1024x768xf32>, %arg17 = %313#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg18 = %313#0 : memref<1x1024xf32>, %arg19 = %312 : f32) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
      %subview_100 = memref.subview %subview[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932880>>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg18, %arg19, %subview_100 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932880>>) outs(%arg17 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.divf %in, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.addf %out, %321 : f32
        linalg.yield %322 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %313#1 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %alloc : memref<768xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapse_shape = memref.collapse_shape %arg16 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
      %subview = memref.subview %arg17[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %collapse_shape, %subview : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
      memref.copy %arg17, %arg18 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg9 : memref<6x768x768xf32>, %arg17 = %alloc : memref<768xf32>, %arg18 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.addf %out, %319 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    %314 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %315 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %314 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_100 = arith.constant 9.99999974E-6 : f32
      %319 = arith.divf %arg16, %cst : f32
      %320 = arith.addf %319, %cst_100 : f32
      %321 = math.rsqrt %320 : f32
      cinm.yield %321 : f32
    }
    cinm.compute_block (%arg16 = %arg13 : memref<6x768xf32>, %arg17 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg18 = %315 : f32, %arg19 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %subview = memref.subview %arg16[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg17, %arg18, %subview : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3840>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32):
        %319 = arith.mulf %in, %in_100 : f32
        %320 = arith.mulf %319, %in_101 : f32
        linalg.yield %320 : f32
      }
      cinm.yield
    }
    cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc : memref<768xf32>, %arg17 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#cinm.host_platform]} {
      memref.copy %arg16, %arg17 : memref<768xf32> to memref<768xf32>
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg10 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg12 : memref<6x2048x768xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%arg17 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_100: f32, %out: f32):
        %320 = arith.mulf %in, %in_100 : f32
        %321 = arith.addf %out, %320 : f32
        linalg.yield %321 : f32
      }
      cinm.yield
    }
    cinm.compute_block (%arg16 = %arg11 : memref<6x768x2048xf32>, %arg17 = %alloc_17 : memref<2048xf32>, %arg18 = %alloc_17 : memref<2048xf32>, %arg19 = %52 : memref<768xf32, strided<[1], offset: ?>>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32_2 : memref<f32>
      %subview = memref.subview %arg16[5, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview, %arg17, %319 : memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
        %320 = arith.negf %out : f32
        %321 = math.exp %320 : f32
        %322 = arith.addf %321, %in_101 : f32
        %323 = arith.divf %in_101, %322 : f32
        %324 = arith.mulf %out, %323 : f32
        %325 = arith.mulf %324, %in_100 : f32
        %326 = arith.mulf %in, %325 : f32
        %327 = arith.addf %out_102, %326 : f32
        linalg.yield %325, %327 : f32, f32
      }
      cinm.yield
    }
    %316 = cinm.compute_block (%arg16 = %52 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_100[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg16 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_100 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %320 = arith.mulf %in, %in : f32
        %321 = arith.addf %320, %out : f32
        linalg.yield %321 : f32
      }
      %319 = memref.load %alloc_100[] : memref<f32>
      cinm.yield %319 : f32
    }
    %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<34048x768xf32>
    %317 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %316 : f32, %arg17 = %alloc_98 : memref<34048x768xf32>, %arg18 = %arg15 : memref<32000x768xf32>) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_100 = arith.constant 7.680000e+02 : f32
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      %320 = arith.divf %arg16, %cst_100 : f32
      %321 = arith.addf %320, %cst : f32
      %322 = math.rsqrt %321 : f32
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%319 : memref<f32>) outs(%arg17 : memref<34048x768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      %subview = memref.subview %arg17[0, 0] [32000, 768] [1, 1] : memref<34048x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
      memref.copy %arg18, %subview : memref<32000x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
      cinm.yield %322 : f32
    }
    %alloc_99 = memref.alloc() {alignment = 64 : i64} : memref<34048xf32>
    cinm.compute_block (%arg16 = %alloc_99 : memref<34048xf32>, %arg17 = %alloc_98 : memref<34048x768xf32>, %arg18 = %52 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %317 : f32, %arg20 = %arg14 : memref<768xf32>) attributes {cinm.available_platforms = [#upmem]} {
      %319 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%319 : memref<f32>) outs(%arg16 : memref<34048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18, %arg19, %arg20 : memref<34048x768xf32>, memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32>) outs(%arg16 : memref<34048xf32>) {
      ^bb0(%in: f32, %in_100: f32, %in_101: f32, %in_102: f32, %out: f32):
        %320 = arith.mulf %in_100, %in_101 : f32
        %321 = arith.mulf %320, %in_102 : f32
        %322 = arith.mulf %in, %321 : f32
        %323 = arith.addf %out, %322 : f32
        linalg.yield %323 : f32
      }
      cinm.yield
    }
    %318 = cinm.compute_block on platform #cinm.host_platform (%arg16 = %alloc_99 : memref<34048xf32>) -> memref<32000xf32, strided<[1]>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %subview = memref.subview %arg16[0] [32000] [1] : memref<34048xf32> to memref<32000xf32, strided<[1]>>
      cinm.yield %subview : memref<32000xf32, strided<[1]>>
    }
    %cast = memref.cast %318 : memref<32000xf32, strided<[1]>> to memref<32000xf32>
    return %cast : memref<32000xf32>
  }
  func.func @rot(%arg0: memref<768xf32>, %arg1: index, %arg2: f32, %arg3: f32) -> memref<768xf32> {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %1 = memref.load %arg0[%arg1] : memref<768xf32>
    %2 = memref.load %arg0[%0] : memref<768xf32>
    %3 = arith.mulf %1, %arg2 : f32
    %4 = arith.mulf %2, %arg3 : f32
    %5 = arith.subf %3, %4 : f32
    memref.store %5, %arg0[%arg1] : memref<768xf32>
    %6 = arith.mulf %1, %arg3 : f32
    %7 = arith.mulf %2, %arg2 : f32
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %arg0[%0] : memref<768xf32>
    return %arg0 : memref<768xf32>
  }
  func.func @mha(%arg0: memref<768xf32>, %arg1: memref<1024x768xf32>, %arg2: memref<1024x768xf32>, %arg3: index) -> memref<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @__constant_xf32_1 : memref<f32>
    %1 = memref.get_global @__constant_xf32 : memref<f32>
    %2 = arith.addi %arg3, %c1 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %subview = memref.subview %arg0[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_2 = memref.subview %arg1[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1]>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%1 : memref<f32>) outs(%alloc_4 : memref<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_5 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_2, %subview : memref<1024x48xf32, strided<[768, 1]>>, memref<48xf32, strided<[1]>>) outs(%alloc_5 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_5, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_6[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_6 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %3 = memref.load %alloc_6[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %3 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_7[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_7 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %4 = memref.load %alloc_7[] : memref<f32>
    %expand_shape = memref.expand_shape %alloc_3 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_8 = memref.subview %arg2[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1]>>
    %subview_9 = memref.subview %alloc[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %expand_shape_10 = memref.expand_shape %subview_9 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1]>> into memref<1x48xf32>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_10 : memref<1x48xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %4, %subview_8 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1]>>) outs(%expand_shape_10 : memref<1x48xf32>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_11 = memref.subview %arg0[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_12 = memref.subview %arg1[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 48>>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_13 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_12, %subview_11 : memref<1024x48xf32, strided<[768, 1], offset: 48>>, memref<48xf32, strided<[1], offset: 48>>) outs(%alloc_13 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_13, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_14[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_14 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %5 = memref.load %alloc_14[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %5 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_15[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_15 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %6 = memref.load %alloc_15[] : memref<f32>
    %subview_16 = memref.subview %arg2[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 48>>
    %subview_17 = memref.subview %alloc[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_18 = memref.expand_shape %subview_17 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_18 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %6, %subview_16 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 48>>) outs(%expand_shape_18 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_19 = memref.subview %arg0[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_20 = memref.subview %arg1[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 96>>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_21 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_20, %subview_19 : memref<1024x48xf32, strided<[768, 1], offset: 96>>, memref<48xf32, strided<[1], offset: 96>>) outs(%alloc_21 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_21, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_22[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_22 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %7 = memref.load %alloc_22[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %7 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_23[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_23 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %8 = memref.load %alloc_23[] : memref<f32>
    %subview_24 = memref.subview %arg2[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 96>>
    %subview_25 = memref.subview %alloc[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_26 = memref.expand_shape %subview_25 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_26 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %8, %subview_24 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 96>>) outs(%expand_shape_26 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_27 = memref.subview %arg0[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_28 = memref.subview %arg1[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 144>>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_29 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_28, %subview_27 : memref<1024x48xf32, strided<[768, 1], offset: 144>>, memref<48xf32, strided<[1], offset: 144>>) outs(%alloc_29 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_29, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_30[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_30 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %9 = memref.load %alloc_30[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %9 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_31[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_31 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %10 = memref.load %alloc_31[] : memref<f32>
    %subview_32 = memref.subview %arg2[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 144>>
    %subview_33 = memref.subview %alloc[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_34 = memref.expand_shape %subview_33 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_34 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %10, %subview_32 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 144>>) outs(%expand_shape_34 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_35 = memref.subview %arg0[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_36 = memref.subview %arg1[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 192>>
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_37 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_36, %subview_35 : memref<1024x48xf32, strided<[768, 1], offset: 192>>, memref<48xf32, strided<[1], offset: 192>>) outs(%alloc_37 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_37, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_38[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_38 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %11 = memref.load %alloc_38[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %11 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_39[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_39 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %12 = memref.load %alloc_39[] : memref<f32>
    %subview_40 = memref.subview %arg2[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 192>>
    %subview_41 = memref.subview %alloc[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_42 = memref.expand_shape %subview_41 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_42 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %12, %subview_40 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 192>>) outs(%expand_shape_42 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_43 = memref.subview %arg0[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_44 = memref.subview %arg1[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 240>>
    %alloc_45 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_45 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_44, %subview_43 : memref<1024x48xf32, strided<[768, 1], offset: 240>>, memref<48xf32, strided<[1], offset: 240>>) outs(%alloc_45 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_45, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_46[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_46 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %13 = memref.load %alloc_46[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %13 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_47[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_47 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %14 = memref.load %alloc_47[] : memref<f32>
    %subview_48 = memref.subview %arg2[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 240>>
    %subview_49 = memref.subview %alloc[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_50 = memref.expand_shape %subview_49 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_50 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %14, %subview_48 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 240>>) outs(%expand_shape_50 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_51 = memref.subview %arg0[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_52 = memref.subview %arg1[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 288>>
    %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_53 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_52, %subview_51 : memref<1024x48xf32, strided<[768, 1], offset: 288>>, memref<48xf32, strided<[1], offset: 288>>) outs(%alloc_53 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_53, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_54[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_54 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %15 = memref.load %alloc_54[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %15 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_55[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_55 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %16 = memref.load %alloc_55[] : memref<f32>
    %subview_56 = memref.subview %arg2[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 288>>
    %subview_57 = memref.subview %alloc[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_58 = memref.expand_shape %subview_57 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_58 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %16, %subview_56 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 288>>) outs(%expand_shape_58 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_59 = memref.subview %arg0[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_60 = memref.subview %arg1[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 336>>
    %alloc_61 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_61 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_60, %subview_59 : memref<1024x48xf32, strided<[768, 1], offset: 336>>, memref<48xf32, strided<[1], offset: 336>>) outs(%alloc_61 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_61, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_62 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_62[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_62 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %17 = memref.load %alloc_62[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %17 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_63[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_63 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %18 = memref.load %alloc_63[] : memref<f32>
    %subview_64 = memref.subview %arg2[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 336>>
    %subview_65 = memref.subview %alloc[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_66 = memref.expand_shape %subview_65 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_66 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %18, %subview_64 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 336>>) outs(%expand_shape_66 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_67 = memref.subview %arg0[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_68 = memref.subview %arg1[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 384>>
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_69 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_68, %subview_67 : memref<1024x48xf32, strided<[768, 1], offset: 384>>, memref<48xf32, strided<[1], offset: 384>>) outs(%alloc_69 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_69, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_70[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_70 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %19 = memref.load %alloc_70[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %19 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_71[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_71 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %20 = memref.load %alloc_71[] : memref<f32>
    %subview_72 = memref.subview %arg2[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 384>>
    %subview_73 = memref.subview %alloc[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_74 = memref.expand_shape %subview_73 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_74 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %20, %subview_72 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 384>>) outs(%expand_shape_74 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_75 = memref.subview %arg0[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_76 = memref.subview %arg1[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 432>>
    %alloc_77 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_77 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_76, %subview_75 : memref<1024x48xf32, strided<[768, 1], offset: 432>>, memref<48xf32, strided<[1], offset: 432>>) outs(%alloc_77 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_77, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_78[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_78 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %21 = memref.load %alloc_78[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %21 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_79[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_79 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %22 = memref.load %alloc_79[] : memref<f32>
    %subview_80 = memref.subview %arg2[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 432>>
    %subview_81 = memref.subview %alloc[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_82 = memref.expand_shape %subview_81 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_82 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %22, %subview_80 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 432>>) outs(%expand_shape_82 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_83 = memref.subview %arg0[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_84 = memref.subview %arg1[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 480>>
    %alloc_85 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_85 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_84, %subview_83 : memref<1024x48xf32, strided<[768, 1], offset: 480>>, memref<48xf32, strided<[1], offset: 480>>) outs(%alloc_85 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_85, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_86 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_86[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_86 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %23 = memref.load %alloc_86[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %23 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_87[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_87 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %24 = memref.load %alloc_87[] : memref<f32>
    %subview_88 = memref.subview %arg2[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 480>>
    %subview_89 = memref.subview %alloc[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_90 = memref.expand_shape %subview_89 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_90 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %24, %subview_88 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 480>>) outs(%expand_shape_90 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_91 = memref.subview %arg0[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_92 = memref.subview %arg1[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 528>>
    %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_93 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_92, %subview_91 : memref<1024x48xf32, strided<[768, 1], offset: 528>>, memref<48xf32, strided<[1], offset: 528>>) outs(%alloc_93 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_93, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_94[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_94 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %25 = memref.load %alloc_94[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %25 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_95[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_95 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %26 = memref.load %alloc_95[] : memref<f32>
    %subview_96 = memref.subview %arg2[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 528>>
    %subview_97 = memref.subview %alloc[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_98 = memref.expand_shape %subview_97 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_98 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %26, %subview_96 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 528>>) outs(%expand_shape_98 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_99 = memref.subview %arg0[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_100 = memref.subview %arg1[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 576>>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_101 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_100, %subview_99 : memref<1024x48xf32, strided<[768, 1], offset: 576>>, memref<48xf32, strided<[1], offset: 576>>) outs(%alloc_101 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_101, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_102[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_102 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %27 = memref.load %alloc_102[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %27 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_103 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_103[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_103 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %28 = memref.load %alloc_103[] : memref<f32>
    %subview_104 = memref.subview %arg2[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 576>>
    %subview_105 = memref.subview %alloc[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_106 = memref.expand_shape %subview_105 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_106 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %28, %subview_104 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 576>>) outs(%expand_shape_106 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_107 = memref.subview %arg0[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_108 = memref.subview %arg1[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 624>>
    %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_109 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_108, %subview_107 : memref<1024x48xf32, strided<[768, 1], offset: 624>>, memref<48xf32, strided<[1], offset: 624>>) outs(%alloc_109 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_109, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_110[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_110 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %29 = memref.load %alloc_110[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %29 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_111 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_111[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_111 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %30 = memref.load %alloc_111[] : memref<f32>
    %subview_112 = memref.subview %arg2[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 624>>
    %subview_113 = memref.subview %alloc[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_114 = memref.expand_shape %subview_113 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_114 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %30, %subview_112 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 624>>) outs(%expand_shape_114 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_115 = memref.subview %arg0[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_116 = memref.subview %arg1[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 672>>
    %alloc_117 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    memref.copy %alloc_4, %alloc_117 : memref<1024xf32> to memref<1024xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_116, %subview_115 : memref<1024x48xf32, strided<[768, 1], offset: 672>>, memref<48xf32, strided<[1], offset: 672>>) outs(%alloc_117 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_117, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_118 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_118[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_118 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %31 = memref.load %alloc_118[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %31 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_119 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_119[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_119 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %32 = memref.load %alloc_119[] : memref<f32>
    %subview_120 = memref.subview %arg2[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 672>>
    %subview_121 = memref.subview %alloc[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_122 = memref.expand_shape %subview_121 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_122 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %32, %subview_120 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 672>>) outs(%expand_shape_122 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    %subview_123 = memref.subview %arg0[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_124 = memref.subview %arg1[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 720>>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_124, %subview_123 : memref<1024x48xf32, strided<[768, 1], offset: 720>>, memref<48xf32, strided<[1], offset: 720>>) outs(%alloc_4 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.mulf %in, %in_130 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_4, %0 : memref<1024xf32>, memref<f32>) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      linalg.yield %35 : f32
    }
    scf.for %arg4 = %2 to %c1024 step %c1 {
      memref.store %cst_1, %alloc_3[%arg4] : memref<1024xf32>
    }
    %alloc_125 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc_125[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_125 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.maxnumf %in, %out : f32
      linalg.yield %35 : f32
    }
    %33 = memref.load %alloc_125[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc_3, %33 : memref<1024xf32>, f32) outs(%alloc_3 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_130: f32, %out: f32):
      %35 = arith.subf %in, %in_130 : f32
      %36 = math.exp %35 : f32
      linalg.yield %36 : f32
    }
    %alloc_126 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_126[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%alloc_3 : memref<1024xf32>) outs(%alloc_126 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    }
    %34 = memref.load %alloc_126[] : memref<f32>
    %subview_127 = memref.subview %arg2[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32> to memref<1024x48xf32, strided<[768, 1], offset: 720>>
    %subview_128 = memref.subview %alloc[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_129 = memref.expand_shape %subview_128 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<f32>) outs(%expand_shape_129 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %34, %subview_127 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 720>>) outs(%expand_shape_129 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
    ^bb0(%in: f32, %in_130: f32, %in_131: f32, %out: f32):
      %35 = arith.divf %in, %in_130 : f32
      %36 = arith.mulf %35, %in_131 : f32
      %37 = arith.addf %out, %36 : f32
      linalg.yield %37 : f32
    }
    return %alloc : memref<768xf32>
  }
  func.func @rmsnorm(%arg0: memref<768xf32>, %arg1: memref<768xf32>) -> memref<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 7.680000e+02 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<768xf32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.mulf %in, %in : f32
      %5 = arith.addf %4, %out : f32
      linalg.yield %5 : f32
    }
    %0 = memref.load %alloc[] : memref<f32>
    %1 = arith.divf %0, %cst_0 : f32
    %2 = arith.addf %1, %cst_1 : f32
    %3 = math.rsqrt %2 : f32
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %3, %arg1 : memref<768xf32>, f32, memref<768xf32>) outs(%alloc_2 : memref<768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %in_4: f32, %out: f32):
      %4 = arith.mulf %in, %in_3 : f32
      %5 = arith.mulf %4, %in_4 : f32
      linalg.yield %5 : f32
    }
    return %alloc_2 : memref<768xf32>
  }
  func.func @softmax(%arg0: memref<1024xf32>) -> memref<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<1024xf32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxnumf %in, %out : f32
      linalg.yield %2 : f32
    }
    %0 = memref.load %alloc[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg0, %0 : memref<1024xf32>, f32) outs(%arg0 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %2 = arith.subf %in, %in_2 : f32
      %3 = math.exp %2 : f32
      linalg.yield %3 : f32
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_1[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<1024xf32>) outs(%alloc_1 : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %out : f32
      linalg.yield %2 : f32
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg0, %1 : memref<1024xf32>, f32) outs(%arg0 : memref<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %2 = arith.divf %in, %in_2 : f32
      linalg.yield %2 : f32
    }
    return %arg0 : memref<1024xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} attributes {sym_name = "forward"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["scf.for"]} in %0 : (!transform.any_op) -> !transform.any_op
      %2:2 = transform.split_handle %1 {overflow_result = 1 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.loop.unroll %2#1 {factor = 6 : i64} : !transform.any_op
      %3 = transform.structured.match ops{["func.func"]} attributes {sym_name = "mha"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %4 = transform.structured.match ops{["scf.for"]} in %3 : (!transform.any_op) -> !transform.any_op
      %5:2 = transform.split_handle %4 {overflow_result = 1 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.loop.unroll %5#1 {factor = 16 : i64} : !transform.any_op
      transform.yield
    }
  }
}
