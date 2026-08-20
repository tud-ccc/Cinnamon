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
    %subview = memref.subview %arg4[%arg0, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %arg5[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %0 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %subview : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %1 = cinm.compute_block (%arg16 = %0 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %2 = cinm.compute_block (%arg16 = %subview : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %1 : f32, %arg18 = %subview_0 : memref<768xf32, strided<[1]>>, %arg19 = %alloc_1 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1]>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    %subview_2 = memref.subview %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %subview_3 = memref.subview %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %subview_4 = memref.subview %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %3 = cinm.compute_block (%arg16 = %alloc_1 : memref<768xf32>, %arg17 = %subview_2 : memref<768x768xf32, strided<[768, 1]>>, %arg18 = %2 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg16 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32>
    }
    %subview_5 = memref.subview %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %4 = cinm.compute_block (%arg16 = %subview_5 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_3 : memref<768x768xf32, strided<[768, 1]>>, %arg18 = %2 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_6 = memref.subview %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %5 = cinm.compute_block (%arg16 = %subview_6 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_4 : memref<768x768xf32, strided<[768, 1]>>, %arg18 = %2 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %5, %subview_6 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %6:3 = cinm.compute_block (%arg16 = %arg1 : index, %arg17 = %3 : memref<768xf32>, %arg18 = %4 : memref<768xf32, strided<[1], offset: ?>>) -> f32, memref<768xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_751 = arith.constant 1.000000e+04 : f32
      %cst_752 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %760 = arith.index_cast %arg16 : index to i64
      %761 = arith.uitofp %760 : i64 to f32
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %762 = arith.remui %arg19, %c48 : index
        %763 = arith.index_cast %762 : index to i64
        %764 = arith.uitofp %763 : i64 to f32
        %765 = arith.divf %764, %cst : f32
        %766 = math.powf %cst_751, %765 : f32
        %767 = arith.divf %cst_752, %766 : f32
        %768 = arith.mulf %761, %767 : f32
        %769 = math.cos %768 : f32
        %770 = math.sin %768 : f32
        %771 = arith.addi %arg19, %c1 : index
        %772 = memref.load %arg17[%arg19] : memref<768xf32>
        %773 = memref.load %arg17[%771] : memref<768xf32>
        %774 = arith.mulf %772, %769 : f32
        %775 = arith.mulf %773, %770 : f32
        %776 = arith.subf %774, %775 : f32
        memref.store %776, %arg17[%arg19] : memref<768xf32>
        %777 = arith.mulf %772, %770 : f32
        %778 = arith.mulf %773, %769 : f32
        %779 = arith.addf %777, %778 : f32
        memref.store %779, %arg17[%771] : memref<768xf32>
        %780 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %780 {
          %781 = memref.load %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %782 = memref.load %arg18[%771] : memref<768xf32, strided<[1], offset: ?>>
          %783 = arith.mulf %781, %769 : f32
          %784 = arith.mulf %782, %770 : f32
          %785 = arith.subf %783, %784 : f32
          memref.store %785, %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %786 = arith.mulf %781, %770 : f32
          %787 = arith.mulf %782, %769 : f32
          %788 = arith.addf %786, %787 : f32
          memref.store %788, %arg18[%771] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      cinm.yield %761, %arg17, %arg18 : f32, memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %6#2, %subview_5 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_7 = memref.subview %arg2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
    %subview_8 = memref.subview %arg3[0, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1]>>
    %7 = cinm.compute_block (%arg16 = %arg1 : index) -> index attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c1 = arith.constant 1 : index
      %760 = arith.addi %arg16, %c1 : index
      cinm.yield %760 : index
    }
    %subview_9 = memref.subview %6#1[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_10 = memref.subview %subview_7[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1]>>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    %8 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_10 : memref<1024x48xf32, strided<[768, 1]>>, %arg18 = %subview_9 : memref<48xf32, strided<[1]>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1]>>, memref<48xf32, strided<[1]>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %9 = cinm.compute_block (%arg16 = %8 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %10 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %9 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %11 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %10 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %12 = cinm.compute_block (%arg16 = %10 : memref<1024xf32>, %arg17 = %11 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %13 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %12 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape = memref.expand_shape %12 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_12 = memref.subview %subview_8[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1]>>
    %subview_13 = memref.subview %alloc_1[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %expand_shape_14 = memref.expand_shape %subview_13 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1]>> into memref<1x48xf32>
    %14 = cinm.compute_block (%arg16 = %expand_shape_14 : memref<1x48xf32>, %arg17 = %expand_shape : memref<1x1024xf32>, %arg18 = %13 : f32, %arg19 = %subview_12 : memref<1024x48xf32, strided<[768, 1]>>) -> memref<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1]>>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32>
    }
    %collapse_shape = memref.collapse_shape %14 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
    %subview_15 = memref.subview %2[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    memref.copy %collapse_shape, %subview_15 : memref<48xf32> to memref<48xf32, strided<[1]>>
    %subview_16 = memref.subview %6#1[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_17 = memref.subview %subview_7[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 48>>
    %15 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_17 : memref<1024x48xf32, strided<[768, 1], offset: 48>>, %arg18 = %subview_16 : memref<48xf32, strided<[1], offset: 48>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 48>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %16 = cinm.compute_block (%arg16 = %15 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %17 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %16 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %18 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %17 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %19 = cinm.compute_block (%arg16 = %17 : memref<1024xf32>, %arg17 = %18 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %20 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %19 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_18 = memref.expand_shape %19 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_19 = memref.subview %subview_8[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 48>>
    %subview_20 = memref.subview %2[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_21 = memref.expand_shape %subview_20 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    %21 = cinm.compute_block (%arg16 = %expand_shape_21 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %expand_shape_18 : memref<1x1024xf32>, %arg18 = %20 : f32, %arg19 = %subview_19 : memref<1024x48xf32, strided<[768, 1], offset: 48>>) -> memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 48>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    %collapse_shape_22 = memref.collapse_shape %21 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
    memref.copy %collapse_shape_22, %subview_20 : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
    %subview_23 = memref.subview %6#1[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_24 = memref.subview %subview_7[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 96>>
    %22 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_24 : memref<1024x48xf32, strided<[768, 1], offset: 96>>, %arg18 = %subview_23 : memref<48xf32, strided<[1], offset: 96>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 96>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %23 = cinm.compute_block (%arg16 = %22 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %24 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %23 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %25 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %24 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %26 = cinm.compute_block (%arg16 = %24 : memref<1024xf32>, %arg17 = %25 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %27 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %26 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_25 = memref.expand_shape %26 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_26 = memref.subview %subview_8[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 96>>
    %subview_27 = memref.subview %2[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_28 = memref.expand_shape %subview_27 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    %28 = cinm.compute_block (%arg16 = %expand_shape_28 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %expand_shape_25 : memref<1x1024xf32>, %arg18 = %27 : f32, %arg19 = %subview_26 : memref<1024x48xf32, strided<[768, 1], offset: 96>>) -> memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 96>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    %collapse_shape_29 = memref.collapse_shape %28 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
    memref.copy %collapse_shape_29, %subview_27 : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
    %subview_30 = memref.subview %6#1[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_31 = memref.subview %subview_7[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 144>>
    %29 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_31 : memref<1024x48xf32, strided<[768, 1], offset: 144>>, %arg18 = %subview_30 : memref<48xf32, strided<[1], offset: 144>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 144>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %30 = cinm.compute_block (%arg16 = %29 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %31 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %30 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %32 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %31 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %33 = cinm.compute_block (%arg16 = %31 : memref<1024xf32>, %arg17 = %32 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %34 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %33 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_32 = memref.expand_shape %33 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_33 = memref.subview %subview_8[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 144>>
    %subview_34 = memref.subview %2[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_35 = memref.expand_shape %subview_34 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    %35 = cinm.compute_block (%arg16 = %expand_shape_35 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %expand_shape_32 : memref<1x1024xf32>, %arg18 = %34 : f32, %arg19 = %subview_33 : memref<1024x48xf32, strided<[768, 1], offset: 144>>) -> memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 144>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    %collapse_shape_36 = memref.collapse_shape %35 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
    memref.copy %collapse_shape_36, %subview_34 : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
    %subview_37 = memref.subview %6#1[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_38 = memref.subview %subview_7[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 192>>
    %36 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_38 : memref<1024x48xf32, strided<[768, 1], offset: 192>>, %arg18 = %subview_37 : memref<48xf32, strided<[1], offset: 192>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 192>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %37 = cinm.compute_block (%arg16 = %36 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %38 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %37 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %39 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %38 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %40 = cinm.compute_block (%arg16 = %38 : memref<1024xf32>, %arg17 = %39 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %41 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %40 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_39 = memref.expand_shape %40 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_40 = memref.subview %subview_8[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 192>>
    %subview_41 = memref.subview %2[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_42 = memref.expand_shape %subview_41 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    %42 = cinm.compute_block (%arg16 = %expand_shape_42 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %expand_shape_39 : memref<1x1024xf32>, %arg18 = %41 : f32, %arg19 = %subview_40 : memref<1024x48xf32, strided<[768, 1], offset: 192>>) -> memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 192>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    %collapse_shape_43 = memref.collapse_shape %42 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
    memref.copy %collapse_shape_43, %subview_41 : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
    %subview_44 = memref.subview %6#1[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_45 = memref.subview %subview_7[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 240>>
    %43 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_45 : memref<1024x48xf32, strided<[768, 1], offset: 240>>, %arg18 = %subview_44 : memref<48xf32, strided<[1], offset: 240>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 240>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %44 = cinm.compute_block (%arg16 = %43 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %45 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %44 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %46 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %45 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %47 = cinm.compute_block (%arg16 = %45 : memref<1024xf32>, %arg17 = %46 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %48 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %47 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_46 = memref.expand_shape %47 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_47 = memref.subview %subview_8[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 240>>
    %subview_48 = memref.subview %2[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_49 = memref.expand_shape %subview_48 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    %49 = cinm.compute_block (%arg16 = %expand_shape_49 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %expand_shape_46 : memref<1x1024xf32>, %arg18 = %48 : f32, %arg19 = %subview_47 : memref<1024x48xf32, strided<[768, 1], offset: 240>>) -> memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 240>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    %collapse_shape_50 = memref.collapse_shape %49 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
    memref.copy %collapse_shape_50, %subview_48 : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
    %subview_51 = memref.subview %6#1[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_52 = memref.subview %subview_7[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 288>>
    %50 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_52 : memref<1024x48xf32, strided<[768, 1], offset: 288>>, %arg18 = %subview_51 : memref<48xf32, strided<[1], offset: 288>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 288>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %51 = cinm.compute_block (%arg16 = %50 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %52 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %51 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %53 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %52 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %54 = cinm.compute_block (%arg16 = %52 : memref<1024xf32>, %arg17 = %53 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %55 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %54 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_53 = memref.expand_shape %54 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_54 = memref.subview %subview_8[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 288>>
    %subview_55 = memref.subview %2[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_56 = memref.expand_shape %subview_55 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    %56 = cinm.compute_block (%arg16 = %expand_shape_56 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %expand_shape_53 : memref<1x1024xf32>, %arg18 = %55 : f32, %arg19 = %subview_54 : memref<1024x48xf32, strided<[768, 1], offset: 288>>) -> memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 288>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    %collapse_shape_57 = memref.collapse_shape %56 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
    memref.copy %collapse_shape_57, %subview_55 : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
    %subview_58 = memref.subview %6#1[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_59 = memref.subview %subview_7[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 336>>
    %57 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_59 : memref<1024x48xf32, strided<[768, 1], offset: 336>>, %arg18 = %subview_58 : memref<48xf32, strided<[1], offset: 336>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 336>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %58 = cinm.compute_block (%arg16 = %57 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %59 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %58 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %60 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %59 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %61 = cinm.compute_block (%arg16 = %59 : memref<1024xf32>, %arg17 = %60 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %62 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %61 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_60 = memref.expand_shape %61 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_61 = memref.subview %subview_8[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 336>>
    %subview_62 = memref.subview %2[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_63 = memref.expand_shape %subview_62 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    %63 = cinm.compute_block (%arg16 = %expand_shape_63 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %expand_shape_60 : memref<1x1024xf32>, %arg18 = %62 : f32, %arg19 = %subview_61 : memref<1024x48xf32, strided<[768, 1], offset: 336>>) -> memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 336>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    %collapse_shape_64 = memref.collapse_shape %63 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
    memref.copy %collapse_shape_64, %subview_62 : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
    %subview_65 = memref.subview %6#1[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_66 = memref.subview %subview_7[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 384>>
    %64 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_66 : memref<1024x48xf32, strided<[768, 1], offset: 384>>, %arg18 = %subview_65 : memref<48xf32, strided<[1], offset: 384>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 384>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %65 = cinm.compute_block (%arg16 = %64 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %66 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %65 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %67 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %66 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %68 = cinm.compute_block (%arg16 = %66 : memref<1024xf32>, %arg17 = %67 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %69 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %68 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_67 = memref.expand_shape %68 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_68 = memref.subview %subview_8[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 384>>
    %subview_69 = memref.subview %2[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_70 = memref.expand_shape %subview_69 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    %70 = cinm.compute_block (%arg16 = %expand_shape_70 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %expand_shape_67 : memref<1x1024xf32>, %arg18 = %69 : f32, %arg19 = %subview_68 : memref<1024x48xf32, strided<[768, 1], offset: 384>>) -> memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 384>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    %collapse_shape_71 = memref.collapse_shape %70 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
    memref.copy %collapse_shape_71, %subview_69 : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
    %subview_72 = memref.subview %6#1[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_73 = memref.subview %subview_7[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 432>>
    %71 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_73 : memref<1024x48xf32, strided<[768, 1], offset: 432>>, %arg18 = %subview_72 : memref<48xf32, strided<[1], offset: 432>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 432>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %72 = cinm.compute_block (%arg16 = %71 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %73 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %72 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %74 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %73 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %75 = cinm.compute_block (%arg16 = %73 : memref<1024xf32>, %arg17 = %74 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %76 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %75 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_74 = memref.expand_shape %75 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_75 = memref.subview %subview_8[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 432>>
    %subview_76 = memref.subview %2[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_77 = memref.expand_shape %subview_76 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    %77 = cinm.compute_block (%arg16 = %expand_shape_77 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %expand_shape_74 : memref<1x1024xf32>, %arg18 = %76 : f32, %arg19 = %subview_75 : memref<1024x48xf32, strided<[768, 1], offset: 432>>) -> memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 432>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    %collapse_shape_78 = memref.collapse_shape %77 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
    memref.copy %collapse_shape_78, %subview_76 : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
    %subview_79 = memref.subview %6#1[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_80 = memref.subview %subview_7[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 480>>
    %78 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_80 : memref<1024x48xf32, strided<[768, 1], offset: 480>>, %arg18 = %subview_79 : memref<48xf32, strided<[1], offset: 480>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 480>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %79 = cinm.compute_block (%arg16 = %78 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %80 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %79 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %81 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %80 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %82 = cinm.compute_block (%arg16 = %80 : memref<1024xf32>, %arg17 = %81 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %83 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %82 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_81 = memref.expand_shape %82 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_82 = memref.subview %subview_8[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 480>>
    %subview_83 = memref.subview %2[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_84 = memref.expand_shape %subview_83 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    %84 = cinm.compute_block (%arg16 = %expand_shape_84 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %expand_shape_81 : memref<1x1024xf32>, %arg18 = %83 : f32, %arg19 = %subview_82 : memref<1024x48xf32, strided<[768, 1], offset: 480>>) -> memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 480>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    %collapse_shape_85 = memref.collapse_shape %84 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
    memref.copy %collapse_shape_85, %subview_83 : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
    %subview_86 = memref.subview %6#1[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_87 = memref.subview %subview_7[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 528>>
    %85 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_87 : memref<1024x48xf32, strided<[768, 1], offset: 528>>, %arg18 = %subview_86 : memref<48xf32, strided<[1], offset: 528>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 528>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %86 = cinm.compute_block (%arg16 = %85 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %87 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %86 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %88 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %87 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %89 = cinm.compute_block (%arg16 = %87 : memref<1024xf32>, %arg17 = %88 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %90 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %89 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_88 = memref.expand_shape %89 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_89 = memref.subview %subview_8[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 528>>
    %subview_90 = memref.subview %2[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_91 = memref.expand_shape %subview_90 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    %91 = cinm.compute_block (%arg16 = %expand_shape_91 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %expand_shape_88 : memref<1x1024xf32>, %arg18 = %90 : f32, %arg19 = %subview_89 : memref<1024x48xf32, strided<[768, 1], offset: 528>>) -> memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 528>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    %collapse_shape_92 = memref.collapse_shape %91 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
    memref.copy %collapse_shape_92, %subview_90 : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
    %subview_93 = memref.subview %6#1[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_94 = memref.subview %subview_7[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 576>>
    %92 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_94 : memref<1024x48xf32, strided<[768, 1], offset: 576>>, %arg18 = %subview_93 : memref<48xf32, strided<[1], offset: 576>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 576>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %93 = cinm.compute_block (%arg16 = %92 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %94 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %93 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %95 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %94 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %96 = cinm.compute_block (%arg16 = %94 : memref<1024xf32>, %arg17 = %95 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %97 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %96 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_95 = memref.expand_shape %96 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_96 = memref.subview %subview_8[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 576>>
    %subview_97 = memref.subview %2[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_98 = memref.expand_shape %subview_97 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    %98 = cinm.compute_block (%arg16 = %expand_shape_98 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %expand_shape_95 : memref<1x1024xf32>, %arg18 = %97 : f32, %arg19 = %subview_96 : memref<1024x48xf32, strided<[768, 1], offset: 576>>) -> memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 576>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    %collapse_shape_99 = memref.collapse_shape %98 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
    memref.copy %collapse_shape_99, %subview_97 : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
    %subview_100 = memref.subview %6#1[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_101 = memref.subview %subview_7[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 624>>
    %99 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_101 : memref<1024x48xf32, strided<[768, 1], offset: 624>>, %arg18 = %subview_100 : memref<48xf32, strided<[1], offset: 624>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 624>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %100 = cinm.compute_block (%arg16 = %99 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %101 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %100 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %102 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %101 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %103 = cinm.compute_block (%arg16 = %101 : memref<1024xf32>, %arg17 = %102 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %104 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %103 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_102 = memref.expand_shape %103 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_103 = memref.subview %subview_8[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 624>>
    %subview_104 = memref.subview %2[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_105 = memref.expand_shape %subview_104 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    %105 = cinm.compute_block (%arg16 = %expand_shape_105 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %expand_shape_102 : memref<1x1024xf32>, %arg18 = %104 : f32, %arg19 = %subview_103 : memref<1024x48xf32, strided<[768, 1], offset: 624>>) -> memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 624>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    %collapse_shape_106 = memref.collapse_shape %105 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
    memref.copy %collapse_shape_106, %subview_104 : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
    %subview_107 = memref.subview %6#1[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_108 = memref.subview %subview_7[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 672>>
    %106 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_108 : memref<1024x48xf32, strided<[768, 1], offset: 672>>, %arg18 = %subview_107 : memref<48xf32, strided<[1], offset: 672>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 672>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %107 = cinm.compute_block (%arg16 = %106 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %108 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %107 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %109 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %108 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %110 = cinm.compute_block (%arg16 = %108 : memref<1024xf32>, %arg17 = %109 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %111 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %110 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_109 = memref.expand_shape %110 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_110 = memref.subview %subview_8[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 672>>
    %subview_111 = memref.subview %2[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_112 = memref.expand_shape %subview_111 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    %112 = cinm.compute_block (%arg16 = %expand_shape_112 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %expand_shape_109 : memref<1x1024xf32>, %arg18 = %111 : f32, %arg19 = %subview_110 : memref<1024x48xf32, strided<[768, 1], offset: 672>>) -> memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 672>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    %collapse_shape_113 = memref.collapse_shape %112 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
    memref.copy %collapse_shape_113, %subview_111 : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
    %subview_114 = memref.subview %6#1[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_115 = memref.subview %subview_7[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 720>>
    %113 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_115 : memref<1024x48xf32, strided<[768, 1], offset: 720>>, %arg18 = %subview_114 : memref<48xf32, strided<[1], offset: 720>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 720>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %114 = cinm.compute_block (%arg16 = %113 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %115 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %114 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %116 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %115 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %117 = cinm.compute_block (%arg16 = %115 : memref<1024xf32>, %arg17 = %116 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %118 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %117 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_116 = memref.expand_shape %117 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_117 = memref.subview %subview_8[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1]>> to memref<1024x48xf32, strided<[768, 1], offset: 720>>
    %subview_118 = memref.subview %2[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_119 = memref.expand_shape %subview_118 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    %119 = cinm.compute_block (%arg16 = %expand_shape_119 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %expand_shape_116 : memref<1x1024xf32>, %arg18 = %118 : f32, %arg19 = %subview_117 : memref<1024x48xf32, strided<[768, 1], offset: 720>>) -> memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 720>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    %collapse_shape_120 = memref.collapse_shape %119 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
    memref.copy %collapse_shape_120, %subview_118 : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
    %subview_121 = memref.subview %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1]>>
    %120 = cinm.compute_block (%arg16 = %subview_121 : memref<768x768xf32, strided<[768, 1]>>, %arg17 = %2 : memref<768xf32>, %arg18 = %subview : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17 : memref<768x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.addf %out, %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg18 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_122 = memref.subview %arg13[0, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1]>>
    %121 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %120 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %122 = cinm.compute_block (%arg16 = %121 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %123 = cinm.compute_block (%arg16 = %120 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %122 : f32, %arg18 = %subview_122 : memref<768xf32, strided<[1]>>, %arg19 = %2 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1]>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    memref.copy %123, %2 : memref<768xf32> to memref<768xf32>
    %subview_123 = memref.subview %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
    %subview_124 = memref.subview %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1]>>
    %alloc_125 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
    %124 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_123 : memref<2048x768xf32, strided<[768, 1]>>, %arg18 = %2 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %125 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_124 : memref<2048x768xf32, strided<[768, 1]>>, %arg18 = %2 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1]>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %subview_126 = memref.subview %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1]>>
    %126:2 = cinm.compute_block (%arg16 = %subview_126 : memref<768x2048xf32, strided<[2048, 1]>>, %arg17 = %125 : memref<2048xf32>, %arg18 = %124 : memref<2048xf32>, %arg19 = %120 : memref<768xf32, strided<[1], offset: ?>>) -> memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_2 : memref<f32>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17, %760 : memref<768x2048xf32, strided<[2048, 1]>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32, %out_753: f32):
        %761 = arith.negf %out : f32
        %762 = math.exp %761 : f32
        %763 = arith.addf %762, %in_752 : f32
        %764 = arith.divf %in_752, %763 : f32
        %765 = arith.mulf %out, %764 : f32
        %766 = arith.mulf %765, %in_751 : f32
        %767 = arith.mulf %in, %766 : f32
        %768 = arith.addf %out_753, %767 : f32
        linalg.yield %766, %768 : f32, f32
      }
      cinm.yield %arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_127 = memref.subview %arg5[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
    %127 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %126#1 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %128 = cinm.compute_block (%arg16 = %127 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %129 = cinm.compute_block (%arg16 = %126#1 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %128 : f32, %arg18 = %subview_127 : memref<768xf32, strided<[1], offset: 768>>, %arg19 = %alloc_1 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 768>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    %subview_128 = memref.subview %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %subview_129 = memref.subview %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %subview_130 = memref.subview %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %130 = cinm.compute_block (%arg16 = %alloc_1 : memref<768xf32>, %arg17 = %subview_128 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, %arg18 = %129 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%arg16 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32>
    }
    %subview_131 = memref.subview %arg2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %131 = cinm.compute_block (%arg16 = %subview_131 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_129 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, %arg18 = %129 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_132 = memref.subview %arg3[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %132 = cinm.compute_block (%arg16 = %subview_132 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_130 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, %arg18 = %129 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %132, %subview_132 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %133:2 = cinm.compute_block (%arg16 = %6#0 : f32, %arg17 = %130 : memref<768xf32>, %arg18 = %131 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_751 = arith.constant 1.000000e+04 : f32
      %cst_752 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %760 = arith.remui %arg19, %c48 : index
        %761 = arith.index_cast %760 : index to i64
        %762 = arith.uitofp %761 : i64 to f32
        %763 = arith.divf %762, %cst : f32
        %764 = math.powf %cst_751, %763 : f32
        %765 = arith.divf %cst_752, %764 : f32
        %766 = arith.mulf %arg16, %765 : f32
        %767 = math.cos %766 : f32
        %768 = math.sin %766 : f32
        %769 = arith.addi %arg19, %c1 : index
        %770 = memref.load %arg17[%arg19] : memref<768xf32>
        %771 = memref.load %arg17[%769] : memref<768xf32>
        %772 = arith.mulf %770, %767 : f32
        %773 = arith.mulf %771, %768 : f32
        %774 = arith.subf %772, %773 : f32
        memref.store %774, %arg17[%arg19] : memref<768xf32>
        %775 = arith.mulf %770, %768 : f32
        %776 = arith.mulf %771, %767 : f32
        %777 = arith.addf %775, %776 : f32
        memref.store %777, %arg17[%769] : memref<768xf32>
        %778 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %778 {
          %779 = memref.load %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %780 = memref.load %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
          %781 = arith.mulf %779, %767 : f32
          %782 = arith.mulf %780, %768 : f32
          %783 = arith.subf %781, %782 : f32
          memref.store %783, %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %784 = arith.mulf %779, %768 : f32
          %785 = arith.mulf %780, %767 : f32
          %786 = arith.addf %784, %785 : f32
          memref.store %786, %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      cinm.yield %arg17, %arg18 : memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %133#1, %subview_131 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_133 = memref.subview %arg2[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
    %subview_134 = memref.subview %arg3[1, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 786432>>
    %subview_135 = memref.subview %133#0[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_136 = memref.subview %subview_133[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786432>>
    %134 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_136 : memref<1024x48xf32, strided<[768, 1], offset: 786432>>, %arg18 = %subview_135 : memref<48xf32, strided<[1]>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786432>>, memref<48xf32, strided<[1]>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %135 = cinm.compute_block (%arg16 = %134 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %136 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %135 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %137 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %136 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %138 = cinm.compute_block (%arg16 = %136 : memref<1024xf32>, %arg17 = %137 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %139 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %138 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_137 = memref.expand_shape %138 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_138 = memref.subview %subview_134[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786432>>
    %140 = cinm.compute_block (%arg16 = %expand_shape_14 : memref<1x48xf32>, %arg17 = %expand_shape_137 : memref<1x1024xf32>, %arg18 = %139 : f32, %arg19 = %subview_138 : memref<1024x48xf32, strided<[768, 1], offset: 786432>>) -> memref<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786432>>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32>
    }
    %collapse_shape_139 = memref.collapse_shape %140 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
    %subview_140 = memref.subview %129[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    memref.copy %collapse_shape_139, %subview_140 : memref<48xf32> to memref<48xf32, strided<[1]>>
    %subview_141 = memref.subview %133#0[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_142 = memref.subview %subview_133[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786480>>
    %141 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_142 : memref<1024x48xf32, strided<[768, 1], offset: 786480>>, %arg18 = %subview_141 : memref<48xf32, strided<[1], offset: 48>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786480>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %142 = cinm.compute_block (%arg16 = %141 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %143 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %142 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %144 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %143 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %145 = cinm.compute_block (%arg16 = %143 : memref<1024xf32>, %arg17 = %144 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %146 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %145 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_143 = memref.expand_shape %145 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_144 = memref.subview %subview_134[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786480>>
    %subview_145 = memref.subview %129[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_146 = memref.expand_shape %subview_145 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    %147 = cinm.compute_block (%arg16 = %expand_shape_146 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %expand_shape_143 : memref<1x1024xf32>, %arg18 = %146 : f32, %arg19 = %subview_144 : memref<1024x48xf32, strided<[768, 1], offset: 786480>>) -> memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786480>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    %collapse_shape_147 = memref.collapse_shape %147 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
    memref.copy %collapse_shape_147, %subview_145 : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
    %subview_148 = memref.subview %133#0[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_149 = memref.subview %subview_133[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786528>>
    %148 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_149 : memref<1024x48xf32, strided<[768, 1], offset: 786528>>, %arg18 = %subview_148 : memref<48xf32, strided<[1], offset: 96>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786528>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %149 = cinm.compute_block (%arg16 = %148 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %150 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %149 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %151 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %150 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %152 = cinm.compute_block (%arg16 = %150 : memref<1024xf32>, %arg17 = %151 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %153 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %152 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_150 = memref.expand_shape %152 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_151 = memref.subview %subview_134[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786528>>
    %subview_152 = memref.subview %129[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_153 = memref.expand_shape %subview_152 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    %154 = cinm.compute_block (%arg16 = %expand_shape_153 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %expand_shape_150 : memref<1x1024xf32>, %arg18 = %153 : f32, %arg19 = %subview_151 : memref<1024x48xf32, strided<[768, 1], offset: 786528>>) -> memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786528>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    %collapse_shape_154 = memref.collapse_shape %154 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
    memref.copy %collapse_shape_154, %subview_152 : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
    %subview_155 = memref.subview %133#0[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_156 = memref.subview %subview_133[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786576>>
    %155 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_156 : memref<1024x48xf32, strided<[768, 1], offset: 786576>>, %arg18 = %subview_155 : memref<48xf32, strided<[1], offset: 144>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786576>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %156 = cinm.compute_block (%arg16 = %155 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %157 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %156 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %158 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %157 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %159 = cinm.compute_block (%arg16 = %157 : memref<1024xf32>, %arg17 = %158 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %160 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %159 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_157 = memref.expand_shape %159 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_158 = memref.subview %subview_134[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786576>>
    %subview_159 = memref.subview %129[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_160 = memref.expand_shape %subview_159 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    %161 = cinm.compute_block (%arg16 = %expand_shape_160 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %expand_shape_157 : memref<1x1024xf32>, %arg18 = %160 : f32, %arg19 = %subview_158 : memref<1024x48xf32, strided<[768, 1], offset: 786576>>) -> memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786576>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    %collapse_shape_161 = memref.collapse_shape %161 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
    memref.copy %collapse_shape_161, %subview_159 : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
    %subview_162 = memref.subview %133#0[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_163 = memref.subview %subview_133[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786624>>
    %162 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_163 : memref<1024x48xf32, strided<[768, 1], offset: 786624>>, %arg18 = %subview_162 : memref<48xf32, strided<[1], offset: 192>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786624>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %163 = cinm.compute_block (%arg16 = %162 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %164 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %163 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %165 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %164 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %166 = cinm.compute_block (%arg16 = %164 : memref<1024xf32>, %arg17 = %165 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %167 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %166 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_164 = memref.expand_shape %166 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_165 = memref.subview %subview_134[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786624>>
    %subview_166 = memref.subview %129[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_167 = memref.expand_shape %subview_166 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    %168 = cinm.compute_block (%arg16 = %expand_shape_167 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %expand_shape_164 : memref<1x1024xf32>, %arg18 = %167 : f32, %arg19 = %subview_165 : memref<1024x48xf32, strided<[768, 1], offset: 786624>>) -> memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786624>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    %collapse_shape_168 = memref.collapse_shape %168 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
    memref.copy %collapse_shape_168, %subview_166 : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
    %subview_169 = memref.subview %133#0[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_170 = memref.subview %subview_133[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786672>>
    %169 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_170 : memref<1024x48xf32, strided<[768, 1], offset: 786672>>, %arg18 = %subview_169 : memref<48xf32, strided<[1], offset: 240>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786672>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %170 = cinm.compute_block (%arg16 = %169 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %171 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %170 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %172 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %171 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %173 = cinm.compute_block (%arg16 = %171 : memref<1024xf32>, %arg17 = %172 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %174 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %173 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_171 = memref.expand_shape %173 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_172 = memref.subview %subview_134[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786672>>
    %subview_173 = memref.subview %129[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_174 = memref.expand_shape %subview_173 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    %175 = cinm.compute_block (%arg16 = %expand_shape_174 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %expand_shape_171 : memref<1x1024xf32>, %arg18 = %174 : f32, %arg19 = %subview_172 : memref<1024x48xf32, strided<[768, 1], offset: 786672>>) -> memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786672>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    %collapse_shape_175 = memref.collapse_shape %175 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
    memref.copy %collapse_shape_175, %subview_173 : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
    %subview_176 = memref.subview %133#0[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_177 = memref.subview %subview_133[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786720>>
    %176 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_177 : memref<1024x48xf32, strided<[768, 1], offset: 786720>>, %arg18 = %subview_176 : memref<48xf32, strided<[1], offset: 288>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786720>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %177 = cinm.compute_block (%arg16 = %176 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %178 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %177 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %179 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %178 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %180 = cinm.compute_block (%arg16 = %178 : memref<1024xf32>, %arg17 = %179 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %181 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %180 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_178 = memref.expand_shape %180 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_179 = memref.subview %subview_134[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786720>>
    %subview_180 = memref.subview %129[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_181 = memref.expand_shape %subview_180 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    %182 = cinm.compute_block (%arg16 = %expand_shape_181 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %expand_shape_178 : memref<1x1024xf32>, %arg18 = %181 : f32, %arg19 = %subview_179 : memref<1024x48xf32, strided<[768, 1], offset: 786720>>) -> memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786720>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    %collapse_shape_182 = memref.collapse_shape %182 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
    memref.copy %collapse_shape_182, %subview_180 : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
    %subview_183 = memref.subview %133#0[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_184 = memref.subview %subview_133[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786768>>
    %183 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_184 : memref<1024x48xf32, strided<[768, 1], offset: 786768>>, %arg18 = %subview_183 : memref<48xf32, strided<[1], offset: 336>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786768>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %184 = cinm.compute_block (%arg16 = %183 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %185 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %184 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %186 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %185 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %187 = cinm.compute_block (%arg16 = %185 : memref<1024xf32>, %arg17 = %186 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %188 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %187 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_185 = memref.expand_shape %187 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_186 = memref.subview %subview_134[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786768>>
    %subview_187 = memref.subview %129[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_188 = memref.expand_shape %subview_187 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    %189 = cinm.compute_block (%arg16 = %expand_shape_188 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %expand_shape_185 : memref<1x1024xf32>, %arg18 = %188 : f32, %arg19 = %subview_186 : memref<1024x48xf32, strided<[768, 1], offset: 786768>>) -> memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786768>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    %collapse_shape_189 = memref.collapse_shape %189 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
    memref.copy %collapse_shape_189, %subview_187 : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
    %subview_190 = memref.subview %133#0[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_191 = memref.subview %subview_133[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786816>>
    %190 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_191 : memref<1024x48xf32, strided<[768, 1], offset: 786816>>, %arg18 = %subview_190 : memref<48xf32, strided<[1], offset: 384>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786816>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %191 = cinm.compute_block (%arg16 = %190 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %192 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %191 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %193 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %192 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %194 = cinm.compute_block (%arg16 = %192 : memref<1024xf32>, %arg17 = %193 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %195 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %194 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_192 = memref.expand_shape %194 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_193 = memref.subview %subview_134[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786816>>
    %subview_194 = memref.subview %129[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_195 = memref.expand_shape %subview_194 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    %196 = cinm.compute_block (%arg16 = %expand_shape_195 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %expand_shape_192 : memref<1x1024xf32>, %arg18 = %195 : f32, %arg19 = %subview_193 : memref<1024x48xf32, strided<[768, 1], offset: 786816>>) -> memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786816>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    %collapse_shape_196 = memref.collapse_shape %196 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
    memref.copy %collapse_shape_196, %subview_194 : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
    %subview_197 = memref.subview %133#0[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_198 = memref.subview %subview_133[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786864>>
    %197 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_198 : memref<1024x48xf32, strided<[768, 1], offset: 786864>>, %arg18 = %subview_197 : memref<48xf32, strided<[1], offset: 432>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786864>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %198 = cinm.compute_block (%arg16 = %197 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %199 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %198 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %200 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %199 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %201 = cinm.compute_block (%arg16 = %199 : memref<1024xf32>, %arg17 = %200 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %202 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %201 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_199 = memref.expand_shape %201 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_200 = memref.subview %subview_134[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786864>>
    %subview_201 = memref.subview %129[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_202 = memref.expand_shape %subview_201 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    %203 = cinm.compute_block (%arg16 = %expand_shape_202 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %expand_shape_199 : memref<1x1024xf32>, %arg18 = %202 : f32, %arg19 = %subview_200 : memref<1024x48xf32, strided<[768, 1], offset: 786864>>) -> memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786864>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    %collapse_shape_203 = memref.collapse_shape %203 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
    memref.copy %collapse_shape_203, %subview_201 : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
    %subview_204 = memref.subview %133#0[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_205 = memref.subview %subview_133[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786912>>
    %204 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_205 : memref<1024x48xf32, strided<[768, 1], offset: 786912>>, %arg18 = %subview_204 : memref<48xf32, strided<[1], offset: 480>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786912>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %205 = cinm.compute_block (%arg16 = %204 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %206 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %205 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %207 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %206 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %208 = cinm.compute_block (%arg16 = %206 : memref<1024xf32>, %arg17 = %207 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %209 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %208 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_206 = memref.expand_shape %208 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_207 = memref.subview %subview_134[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786912>>
    %subview_208 = memref.subview %129[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_209 = memref.expand_shape %subview_208 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    %210 = cinm.compute_block (%arg16 = %expand_shape_209 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %expand_shape_206 : memref<1x1024xf32>, %arg18 = %209 : f32, %arg19 = %subview_207 : memref<1024x48xf32, strided<[768, 1], offset: 786912>>) -> memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786912>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    %collapse_shape_210 = memref.collapse_shape %210 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
    memref.copy %collapse_shape_210, %subview_208 : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
    %subview_211 = memref.subview %133#0[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_212 = memref.subview %subview_133[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786960>>
    %211 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_212 : memref<1024x48xf32, strided<[768, 1], offset: 786960>>, %arg18 = %subview_211 : memref<48xf32, strided<[1], offset: 528>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 786960>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %212 = cinm.compute_block (%arg16 = %211 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %213 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %212 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %214 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %213 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %215 = cinm.compute_block (%arg16 = %213 : memref<1024xf32>, %arg17 = %214 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %216 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %215 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_213 = memref.expand_shape %215 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_214 = memref.subview %subview_134[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 786960>>
    %subview_215 = memref.subview %129[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_216 = memref.expand_shape %subview_215 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    %217 = cinm.compute_block (%arg16 = %expand_shape_216 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %expand_shape_213 : memref<1x1024xf32>, %arg18 = %216 : f32, %arg19 = %subview_214 : memref<1024x48xf32, strided<[768, 1], offset: 786960>>) -> memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 786960>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    %collapse_shape_217 = memref.collapse_shape %217 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
    memref.copy %collapse_shape_217, %subview_215 : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
    %subview_218 = memref.subview %133#0[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_219 = memref.subview %subview_133[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787008>>
    %218 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_219 : memref<1024x48xf32, strided<[768, 1], offset: 787008>>, %arg18 = %subview_218 : memref<48xf32, strided<[1], offset: 576>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 787008>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %219 = cinm.compute_block (%arg16 = %218 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %220 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %219 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %221 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %220 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %222 = cinm.compute_block (%arg16 = %220 : memref<1024xf32>, %arg17 = %221 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %223 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %222 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_220 = memref.expand_shape %222 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_221 = memref.subview %subview_134[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787008>>
    %subview_222 = memref.subview %129[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_223 = memref.expand_shape %subview_222 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    %224 = cinm.compute_block (%arg16 = %expand_shape_223 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %expand_shape_220 : memref<1x1024xf32>, %arg18 = %223 : f32, %arg19 = %subview_221 : memref<1024x48xf32, strided<[768, 1], offset: 787008>>) -> memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787008>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    %collapse_shape_224 = memref.collapse_shape %224 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
    memref.copy %collapse_shape_224, %subview_222 : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
    %subview_225 = memref.subview %133#0[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_226 = memref.subview %subview_133[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787056>>
    %225 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_226 : memref<1024x48xf32, strided<[768, 1], offset: 787056>>, %arg18 = %subview_225 : memref<48xf32, strided<[1], offset: 624>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 787056>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %226 = cinm.compute_block (%arg16 = %225 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %227 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %226 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %228 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %227 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %229 = cinm.compute_block (%arg16 = %227 : memref<1024xf32>, %arg17 = %228 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %230 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %229 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_227 = memref.expand_shape %229 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_228 = memref.subview %subview_134[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787056>>
    %subview_229 = memref.subview %129[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_230 = memref.expand_shape %subview_229 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    %231 = cinm.compute_block (%arg16 = %expand_shape_230 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %expand_shape_227 : memref<1x1024xf32>, %arg18 = %230 : f32, %arg19 = %subview_228 : memref<1024x48xf32, strided<[768, 1], offset: 787056>>) -> memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787056>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    %collapse_shape_231 = memref.collapse_shape %231 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
    memref.copy %collapse_shape_231, %subview_229 : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
    %subview_232 = memref.subview %133#0[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_233 = memref.subview %subview_133[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787104>>
    %232 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_233 : memref<1024x48xf32, strided<[768, 1], offset: 787104>>, %arg18 = %subview_232 : memref<48xf32, strided<[1], offset: 672>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 787104>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %233 = cinm.compute_block (%arg16 = %232 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %234 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %233 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %235 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %234 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %236 = cinm.compute_block (%arg16 = %234 : memref<1024xf32>, %arg17 = %235 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %237 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %236 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_234 = memref.expand_shape %236 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_235 = memref.subview %subview_134[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787104>>
    %subview_236 = memref.subview %129[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_237 = memref.expand_shape %subview_236 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    %238 = cinm.compute_block (%arg16 = %expand_shape_237 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %expand_shape_234 : memref<1x1024xf32>, %arg18 = %237 : f32, %arg19 = %subview_235 : memref<1024x48xf32, strided<[768, 1], offset: 787104>>) -> memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787104>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    %collapse_shape_238 = memref.collapse_shape %238 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
    memref.copy %collapse_shape_238, %subview_236 : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
    %subview_239 = memref.subview %133#0[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_240 = memref.subview %subview_133[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787152>>
    %239 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_240 : memref<1024x48xf32, strided<[768, 1], offset: 787152>>, %arg18 = %subview_239 : memref<48xf32, strided<[1], offset: 720>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 787152>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %240 = cinm.compute_block (%arg16 = %239 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %241 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %240 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %242 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %241 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %243 = cinm.compute_block (%arg16 = %241 : memref<1024xf32>, %arg17 = %242 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %244 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %243 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_241 = memref.expand_shape %243 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_242 = memref.subview %subview_134[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 786432>> to memref<1024x48xf32, strided<[768, 1], offset: 787152>>
    %subview_243 = memref.subview %129[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_244 = memref.expand_shape %subview_243 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    %245 = cinm.compute_block (%arg16 = %expand_shape_244 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %expand_shape_241 : memref<1x1024xf32>, %arg18 = %244 : f32, %arg19 = %subview_242 : memref<1024x48xf32, strided<[768, 1], offset: 787152>>) -> memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 787152>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    %collapse_shape_245 = memref.collapse_shape %245 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
    memref.copy %collapse_shape_245, %subview_243 : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
    %subview_246 = memref.subview %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 589824>>
    %246 = cinm.compute_block (%arg16 = %subview_246 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, %arg17 = %129 : memref<768xf32>, %arg18 = %126#1 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 589824>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.addf %out, %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg18 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_247 = memref.subview %arg13[1, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 768>>
    %247 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %246 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %248 = cinm.compute_block (%arg16 = %247 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %249 = cinm.compute_block (%arg16 = %246 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %248 : f32, %arg18 = %subview_247 : memref<768xf32, strided<[1], offset: 768>>, %arg19 = %129 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 768>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    memref.copy %249, %129 : memref<768xf32> to memref<768xf32>
    %subview_248 = memref.subview %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_249 = memref.subview %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 1572864>>
    %250 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_248 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, %arg18 = %129 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %251 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_249 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, %arg18 = %129 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 1572864>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %subview_250 = memref.subview %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>
    %252:2 = cinm.compute_block (%arg16 = %subview_250 : memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>, %arg17 = %251 : memref<2048xf32>, %arg18 = %250 : memref<2048xf32>, %arg19 = %246 : memref<768xf32, strided<[1], offset: ?>>) -> memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_2 : memref<f32>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17, %760 : memref<768x2048xf32, strided<[2048, 1], offset: 1572864>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32, %out_753: f32):
        %761 = arith.negf %out : f32
        %762 = math.exp %761 : f32
        %763 = arith.addf %762, %in_752 : f32
        %764 = arith.divf %in_752, %763 : f32
        %765 = arith.mulf %out, %764 : f32
        %766 = arith.mulf %765, %in_751 : f32
        %767 = arith.mulf %in, %766 : f32
        %768 = arith.addf %out_753, %767 : f32
        linalg.yield %766, %768 : f32, f32
      }
      cinm.yield %arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_251 = memref.subview %arg5[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
    %253 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %252#1 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %254 = cinm.compute_block (%arg16 = %253 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %255 = cinm.compute_block (%arg16 = %252#1 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %254 : f32, %arg18 = %subview_251 : memref<768xf32, strided<[1], offset: 1536>>, %arg19 = %alloc_1 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 1536>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    %subview_252 = memref.subview %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %subview_253 = memref.subview %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %subview_254 = memref.subview %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %256 = cinm.compute_block (%arg16 = %alloc_1 : memref<768xf32>, %arg17 = %subview_252 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, %arg18 = %255 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%arg16 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32>
    }
    %subview_255 = memref.subview %arg2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %257 = cinm.compute_block (%arg16 = %subview_255 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_253 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, %arg18 = %255 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_256 = memref.subview %arg3[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %258 = cinm.compute_block (%arg16 = %subview_256 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_254 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, %arg18 = %255 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %258, %subview_256 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %259:2 = cinm.compute_block (%arg16 = %6#0 : f32, %arg17 = %256 : memref<768xf32>, %arg18 = %257 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_751 = arith.constant 1.000000e+04 : f32
      %cst_752 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %760 = arith.remui %arg19, %c48 : index
        %761 = arith.index_cast %760 : index to i64
        %762 = arith.uitofp %761 : i64 to f32
        %763 = arith.divf %762, %cst : f32
        %764 = math.powf %cst_751, %763 : f32
        %765 = arith.divf %cst_752, %764 : f32
        %766 = arith.mulf %arg16, %765 : f32
        %767 = math.cos %766 : f32
        %768 = math.sin %766 : f32
        %769 = arith.addi %arg19, %c1 : index
        %770 = memref.load %arg17[%arg19] : memref<768xf32>
        %771 = memref.load %arg17[%769] : memref<768xf32>
        %772 = arith.mulf %770, %767 : f32
        %773 = arith.mulf %771, %768 : f32
        %774 = arith.subf %772, %773 : f32
        memref.store %774, %arg17[%arg19] : memref<768xf32>
        %775 = arith.mulf %770, %768 : f32
        %776 = arith.mulf %771, %767 : f32
        %777 = arith.addf %775, %776 : f32
        memref.store %777, %arg17[%769] : memref<768xf32>
        %778 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %778 {
          %779 = memref.load %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %780 = memref.load %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
          %781 = arith.mulf %779, %767 : f32
          %782 = arith.mulf %780, %768 : f32
          %783 = arith.subf %781, %782 : f32
          memref.store %783, %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %784 = arith.mulf %779, %768 : f32
          %785 = arith.mulf %780, %767 : f32
          %786 = arith.addf %784, %785 : f32
          memref.store %786, %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      cinm.yield %arg17, %arg18 : memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %259#1, %subview_255 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_257 = memref.subview %arg2[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_258 = memref.subview %arg3[2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 1572864>>
    %subview_259 = memref.subview %259#0[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_260 = memref.subview %subview_257[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572864>>
    %260 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_260 : memref<1024x48xf32, strided<[768, 1], offset: 1572864>>, %arg18 = %subview_259 : memref<48xf32, strided<[1]>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1572864>>, memref<48xf32, strided<[1]>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %261 = cinm.compute_block (%arg16 = %260 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %262 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %261 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %263 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %262 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %264 = cinm.compute_block (%arg16 = %262 : memref<1024xf32>, %arg17 = %263 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %265 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %264 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_261 = memref.expand_shape %264 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_262 = memref.subview %subview_258[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572864>>
    %266 = cinm.compute_block (%arg16 = %expand_shape_14 : memref<1x48xf32>, %arg17 = %expand_shape_261 : memref<1x1024xf32>, %arg18 = %265 : f32, %arg19 = %subview_262 : memref<1024x48xf32, strided<[768, 1], offset: 1572864>>) -> memref<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1572864>>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32>
    }
    %collapse_shape_263 = memref.collapse_shape %266 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
    %subview_264 = memref.subview %255[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    memref.copy %collapse_shape_263, %subview_264 : memref<48xf32> to memref<48xf32, strided<[1]>>
    %subview_265 = memref.subview %259#0[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_266 = memref.subview %subview_257[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572912>>
    %267 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_266 : memref<1024x48xf32, strided<[768, 1], offset: 1572912>>, %arg18 = %subview_265 : memref<48xf32, strided<[1], offset: 48>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1572912>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %268 = cinm.compute_block (%arg16 = %267 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %269 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %268 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %270 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %269 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %271 = cinm.compute_block (%arg16 = %269 : memref<1024xf32>, %arg17 = %270 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %272 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %271 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_267 = memref.expand_shape %271 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_268 = memref.subview %subview_258[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572912>>
    %subview_269 = memref.subview %255[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_270 = memref.expand_shape %subview_269 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    %273 = cinm.compute_block (%arg16 = %expand_shape_270 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %expand_shape_267 : memref<1x1024xf32>, %arg18 = %272 : f32, %arg19 = %subview_268 : memref<1024x48xf32, strided<[768, 1], offset: 1572912>>) -> memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1572912>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    %collapse_shape_271 = memref.collapse_shape %273 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
    memref.copy %collapse_shape_271, %subview_269 : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
    %subview_272 = memref.subview %259#0[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_273 = memref.subview %subview_257[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572960>>
    %274 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_273 : memref<1024x48xf32, strided<[768, 1], offset: 1572960>>, %arg18 = %subview_272 : memref<48xf32, strided<[1], offset: 96>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1572960>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %275 = cinm.compute_block (%arg16 = %274 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %276 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %275 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %277 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %276 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %278 = cinm.compute_block (%arg16 = %276 : memref<1024xf32>, %arg17 = %277 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %279 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %278 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_274 = memref.expand_shape %278 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_275 = memref.subview %subview_258[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1572960>>
    %subview_276 = memref.subview %255[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_277 = memref.expand_shape %subview_276 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    %280 = cinm.compute_block (%arg16 = %expand_shape_277 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %expand_shape_274 : memref<1x1024xf32>, %arg18 = %279 : f32, %arg19 = %subview_275 : memref<1024x48xf32, strided<[768, 1], offset: 1572960>>) -> memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1572960>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    %collapse_shape_278 = memref.collapse_shape %280 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
    memref.copy %collapse_shape_278, %subview_276 : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
    %subview_279 = memref.subview %259#0[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_280 = memref.subview %subview_257[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573008>>
    %281 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_280 : memref<1024x48xf32, strided<[768, 1], offset: 1573008>>, %arg18 = %subview_279 : memref<48xf32, strided<[1], offset: 144>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573008>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %282 = cinm.compute_block (%arg16 = %281 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %283 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %282 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %284 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %283 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %285 = cinm.compute_block (%arg16 = %283 : memref<1024xf32>, %arg17 = %284 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %286 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %285 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_281 = memref.expand_shape %285 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_282 = memref.subview %subview_258[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573008>>
    %subview_283 = memref.subview %255[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_284 = memref.expand_shape %subview_283 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    %287 = cinm.compute_block (%arg16 = %expand_shape_284 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %expand_shape_281 : memref<1x1024xf32>, %arg18 = %286 : f32, %arg19 = %subview_282 : memref<1024x48xf32, strided<[768, 1], offset: 1573008>>) -> memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573008>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    %collapse_shape_285 = memref.collapse_shape %287 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
    memref.copy %collapse_shape_285, %subview_283 : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
    %subview_286 = memref.subview %259#0[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_287 = memref.subview %subview_257[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573056>>
    %288 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_287 : memref<1024x48xf32, strided<[768, 1], offset: 1573056>>, %arg18 = %subview_286 : memref<48xf32, strided<[1], offset: 192>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573056>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %289 = cinm.compute_block (%arg16 = %288 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %290 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %289 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %291 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %290 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %292 = cinm.compute_block (%arg16 = %290 : memref<1024xf32>, %arg17 = %291 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %293 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %292 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_288 = memref.expand_shape %292 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_289 = memref.subview %subview_258[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573056>>
    %subview_290 = memref.subview %255[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_291 = memref.expand_shape %subview_290 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    %294 = cinm.compute_block (%arg16 = %expand_shape_291 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %expand_shape_288 : memref<1x1024xf32>, %arg18 = %293 : f32, %arg19 = %subview_289 : memref<1024x48xf32, strided<[768, 1], offset: 1573056>>) -> memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573056>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    %collapse_shape_292 = memref.collapse_shape %294 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
    memref.copy %collapse_shape_292, %subview_290 : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
    %subview_293 = memref.subview %259#0[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_294 = memref.subview %subview_257[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573104>>
    %295 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_294 : memref<1024x48xf32, strided<[768, 1], offset: 1573104>>, %arg18 = %subview_293 : memref<48xf32, strided<[1], offset: 240>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573104>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %296 = cinm.compute_block (%arg16 = %295 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %297 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %296 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %298 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %297 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %299 = cinm.compute_block (%arg16 = %297 : memref<1024xf32>, %arg17 = %298 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %300 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %299 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_295 = memref.expand_shape %299 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_296 = memref.subview %subview_258[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573104>>
    %subview_297 = memref.subview %255[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_298 = memref.expand_shape %subview_297 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    %301 = cinm.compute_block (%arg16 = %expand_shape_298 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %expand_shape_295 : memref<1x1024xf32>, %arg18 = %300 : f32, %arg19 = %subview_296 : memref<1024x48xf32, strided<[768, 1], offset: 1573104>>) -> memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573104>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    %collapse_shape_299 = memref.collapse_shape %301 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
    memref.copy %collapse_shape_299, %subview_297 : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
    %subview_300 = memref.subview %259#0[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_301 = memref.subview %subview_257[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573152>>
    %302 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_301 : memref<1024x48xf32, strided<[768, 1], offset: 1573152>>, %arg18 = %subview_300 : memref<48xf32, strided<[1], offset: 288>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573152>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %303 = cinm.compute_block (%arg16 = %302 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %304 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %303 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %305 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %304 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %306 = cinm.compute_block (%arg16 = %304 : memref<1024xf32>, %arg17 = %305 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %307 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %306 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_302 = memref.expand_shape %306 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_303 = memref.subview %subview_258[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573152>>
    %subview_304 = memref.subview %255[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_305 = memref.expand_shape %subview_304 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    %308 = cinm.compute_block (%arg16 = %expand_shape_305 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %expand_shape_302 : memref<1x1024xf32>, %arg18 = %307 : f32, %arg19 = %subview_303 : memref<1024x48xf32, strided<[768, 1], offset: 1573152>>) -> memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573152>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    %collapse_shape_306 = memref.collapse_shape %308 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
    memref.copy %collapse_shape_306, %subview_304 : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
    %subview_307 = memref.subview %259#0[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_308 = memref.subview %subview_257[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573200>>
    %309 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_308 : memref<1024x48xf32, strided<[768, 1], offset: 1573200>>, %arg18 = %subview_307 : memref<48xf32, strided<[1], offset: 336>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573200>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %310 = cinm.compute_block (%arg16 = %309 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %311 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %310 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %312 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %311 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %313 = cinm.compute_block (%arg16 = %311 : memref<1024xf32>, %arg17 = %312 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %314 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %313 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_309 = memref.expand_shape %313 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_310 = memref.subview %subview_258[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573200>>
    %subview_311 = memref.subview %255[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_312 = memref.expand_shape %subview_311 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    %315 = cinm.compute_block (%arg16 = %expand_shape_312 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %expand_shape_309 : memref<1x1024xf32>, %arg18 = %314 : f32, %arg19 = %subview_310 : memref<1024x48xf32, strided<[768, 1], offset: 1573200>>) -> memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573200>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    %collapse_shape_313 = memref.collapse_shape %315 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
    memref.copy %collapse_shape_313, %subview_311 : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
    %subview_314 = memref.subview %259#0[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_315 = memref.subview %subview_257[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573248>>
    %316 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_315 : memref<1024x48xf32, strided<[768, 1], offset: 1573248>>, %arg18 = %subview_314 : memref<48xf32, strided<[1], offset: 384>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573248>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %317 = cinm.compute_block (%arg16 = %316 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %318 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %317 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %319 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %318 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %320 = cinm.compute_block (%arg16 = %318 : memref<1024xf32>, %arg17 = %319 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %321 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %320 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_316 = memref.expand_shape %320 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_317 = memref.subview %subview_258[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573248>>
    %subview_318 = memref.subview %255[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_319 = memref.expand_shape %subview_318 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    %322 = cinm.compute_block (%arg16 = %expand_shape_319 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %expand_shape_316 : memref<1x1024xf32>, %arg18 = %321 : f32, %arg19 = %subview_317 : memref<1024x48xf32, strided<[768, 1], offset: 1573248>>) -> memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573248>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    %collapse_shape_320 = memref.collapse_shape %322 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
    memref.copy %collapse_shape_320, %subview_318 : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
    %subview_321 = memref.subview %259#0[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_322 = memref.subview %subview_257[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573296>>
    %323 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_322 : memref<1024x48xf32, strided<[768, 1], offset: 1573296>>, %arg18 = %subview_321 : memref<48xf32, strided<[1], offset: 432>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573296>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %324 = cinm.compute_block (%arg16 = %323 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %325 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %324 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %326 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %325 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %327 = cinm.compute_block (%arg16 = %325 : memref<1024xf32>, %arg17 = %326 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %328 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %327 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_323 = memref.expand_shape %327 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_324 = memref.subview %subview_258[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573296>>
    %subview_325 = memref.subview %255[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_326 = memref.expand_shape %subview_325 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    %329 = cinm.compute_block (%arg16 = %expand_shape_326 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %expand_shape_323 : memref<1x1024xf32>, %arg18 = %328 : f32, %arg19 = %subview_324 : memref<1024x48xf32, strided<[768, 1], offset: 1573296>>) -> memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573296>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    %collapse_shape_327 = memref.collapse_shape %329 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
    memref.copy %collapse_shape_327, %subview_325 : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
    %subview_328 = memref.subview %259#0[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_329 = memref.subview %subview_257[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573344>>
    %330 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_329 : memref<1024x48xf32, strided<[768, 1], offset: 1573344>>, %arg18 = %subview_328 : memref<48xf32, strided<[1], offset: 480>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573344>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %331 = cinm.compute_block (%arg16 = %330 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %332 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %331 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %333 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %332 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %334 = cinm.compute_block (%arg16 = %332 : memref<1024xf32>, %arg17 = %333 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %335 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %334 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_330 = memref.expand_shape %334 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_331 = memref.subview %subview_258[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573344>>
    %subview_332 = memref.subview %255[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_333 = memref.expand_shape %subview_332 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    %336 = cinm.compute_block (%arg16 = %expand_shape_333 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %expand_shape_330 : memref<1x1024xf32>, %arg18 = %335 : f32, %arg19 = %subview_331 : memref<1024x48xf32, strided<[768, 1], offset: 1573344>>) -> memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573344>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    %collapse_shape_334 = memref.collapse_shape %336 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
    memref.copy %collapse_shape_334, %subview_332 : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
    %subview_335 = memref.subview %259#0[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_336 = memref.subview %subview_257[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573392>>
    %337 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_336 : memref<1024x48xf32, strided<[768, 1], offset: 1573392>>, %arg18 = %subview_335 : memref<48xf32, strided<[1], offset: 528>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573392>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %338 = cinm.compute_block (%arg16 = %337 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %339 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %338 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %340 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %339 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %341 = cinm.compute_block (%arg16 = %339 : memref<1024xf32>, %arg17 = %340 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %342 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %341 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_337 = memref.expand_shape %341 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_338 = memref.subview %subview_258[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573392>>
    %subview_339 = memref.subview %255[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_340 = memref.expand_shape %subview_339 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    %343 = cinm.compute_block (%arg16 = %expand_shape_340 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %expand_shape_337 : memref<1x1024xf32>, %arg18 = %342 : f32, %arg19 = %subview_338 : memref<1024x48xf32, strided<[768, 1], offset: 1573392>>) -> memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573392>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    %collapse_shape_341 = memref.collapse_shape %343 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
    memref.copy %collapse_shape_341, %subview_339 : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
    %subview_342 = memref.subview %259#0[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_343 = memref.subview %subview_257[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573440>>
    %344 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_343 : memref<1024x48xf32, strided<[768, 1], offset: 1573440>>, %arg18 = %subview_342 : memref<48xf32, strided<[1], offset: 576>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573440>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %345 = cinm.compute_block (%arg16 = %344 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %346 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %345 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %347 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %346 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %348 = cinm.compute_block (%arg16 = %346 : memref<1024xf32>, %arg17 = %347 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %349 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %348 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_344 = memref.expand_shape %348 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_345 = memref.subview %subview_258[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573440>>
    %subview_346 = memref.subview %255[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_347 = memref.expand_shape %subview_346 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    %350 = cinm.compute_block (%arg16 = %expand_shape_347 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %expand_shape_344 : memref<1x1024xf32>, %arg18 = %349 : f32, %arg19 = %subview_345 : memref<1024x48xf32, strided<[768, 1], offset: 1573440>>) -> memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573440>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    %collapse_shape_348 = memref.collapse_shape %350 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
    memref.copy %collapse_shape_348, %subview_346 : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
    %subview_349 = memref.subview %259#0[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_350 = memref.subview %subview_257[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573488>>
    %351 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_350 : memref<1024x48xf32, strided<[768, 1], offset: 1573488>>, %arg18 = %subview_349 : memref<48xf32, strided<[1], offset: 624>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573488>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %352 = cinm.compute_block (%arg16 = %351 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %353 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %352 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %354 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %353 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %355 = cinm.compute_block (%arg16 = %353 : memref<1024xf32>, %arg17 = %354 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %356 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %355 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_351 = memref.expand_shape %355 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_352 = memref.subview %subview_258[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573488>>
    %subview_353 = memref.subview %255[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_354 = memref.expand_shape %subview_353 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    %357 = cinm.compute_block (%arg16 = %expand_shape_354 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %expand_shape_351 : memref<1x1024xf32>, %arg18 = %356 : f32, %arg19 = %subview_352 : memref<1024x48xf32, strided<[768, 1], offset: 1573488>>) -> memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573488>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    %collapse_shape_355 = memref.collapse_shape %357 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
    memref.copy %collapse_shape_355, %subview_353 : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
    %subview_356 = memref.subview %259#0[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_357 = memref.subview %subview_257[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573536>>
    %358 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_357 : memref<1024x48xf32, strided<[768, 1], offset: 1573536>>, %arg18 = %subview_356 : memref<48xf32, strided<[1], offset: 672>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573536>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %359 = cinm.compute_block (%arg16 = %358 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %360 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %359 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %361 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %360 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %362 = cinm.compute_block (%arg16 = %360 : memref<1024xf32>, %arg17 = %361 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %363 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %362 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_358 = memref.expand_shape %362 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_359 = memref.subview %subview_258[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573536>>
    %subview_360 = memref.subview %255[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_361 = memref.expand_shape %subview_360 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    %364 = cinm.compute_block (%arg16 = %expand_shape_361 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %expand_shape_358 : memref<1x1024xf32>, %arg18 = %363 : f32, %arg19 = %subview_359 : memref<1024x48xf32, strided<[768, 1], offset: 1573536>>) -> memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573536>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    %collapse_shape_362 = memref.collapse_shape %364 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
    memref.copy %collapse_shape_362, %subview_360 : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
    %subview_363 = memref.subview %259#0[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_364 = memref.subview %subview_257[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573584>>
    %365 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_364 : memref<1024x48xf32, strided<[768, 1], offset: 1573584>>, %arg18 = %subview_363 : memref<48xf32, strided<[1], offset: 720>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 1573584>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %366 = cinm.compute_block (%arg16 = %365 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %367 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %366 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %368 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %367 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %369 = cinm.compute_block (%arg16 = %367 : memref<1024xf32>, %arg17 = %368 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %370 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %369 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_365 = memref.expand_shape %369 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_366 = memref.subview %subview_258[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 1572864>> to memref<1024x48xf32, strided<[768, 1], offset: 1573584>>
    %subview_367 = memref.subview %255[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_368 = memref.expand_shape %subview_367 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    %371 = cinm.compute_block (%arg16 = %expand_shape_368 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %expand_shape_365 : memref<1x1024xf32>, %arg18 = %370 : f32, %arg19 = %subview_366 : memref<1024x48xf32, strided<[768, 1], offset: 1573584>>) -> memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 1573584>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    %collapse_shape_369 = memref.collapse_shape %371 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
    memref.copy %collapse_shape_369, %subview_367 : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
    %subview_370 = memref.subview %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1179648>>
    %372 = cinm.compute_block (%arg16 = %subview_370 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, %arg17 = %255 : memref<768xf32>, %arg18 = %252#1 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 1179648>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.addf %out, %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg18 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_371 = memref.subview %arg13[2, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 1536>>
    %373 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %372 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %374 = cinm.compute_block (%arg16 = %373 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %375 = cinm.compute_block (%arg16 = %372 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %374 : f32, %arg18 = %subview_371 : memref<768xf32, strided<[1], offset: 1536>>, %arg19 = %255 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 1536>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    memref.copy %375, %255 : memref<768xf32> to memref<768xf32>
    %subview_372 = memref.subview %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_373 = memref.subview %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 3145728>>
    %376 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_372 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, %arg18 = %255 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %377 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_373 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, %arg18 = %255 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 3145728>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %subview_374 = memref.subview %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>
    %378:2 = cinm.compute_block (%arg16 = %subview_374 : memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>, %arg17 = %377 : memref<2048xf32>, %arg18 = %376 : memref<2048xf32>, %arg19 = %372 : memref<768xf32, strided<[1], offset: ?>>) -> memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_2 : memref<f32>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17, %760 : memref<768x2048xf32, strided<[2048, 1], offset: 3145728>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32, %out_753: f32):
        %761 = arith.negf %out : f32
        %762 = math.exp %761 : f32
        %763 = arith.addf %762, %in_752 : f32
        %764 = arith.divf %in_752, %763 : f32
        %765 = arith.mulf %out, %764 : f32
        %766 = arith.mulf %765, %in_751 : f32
        %767 = arith.mulf %in, %766 : f32
        %768 = arith.addf %out_753, %767 : f32
        linalg.yield %766, %768 : f32, f32
      }
      cinm.yield %arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_375 = memref.subview %arg5[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
    %379 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %378#1 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %380 = cinm.compute_block (%arg16 = %379 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %381 = cinm.compute_block (%arg16 = %378#1 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %380 : f32, %arg18 = %subview_375 : memref<768xf32, strided<[1], offset: 2304>>, %arg19 = %alloc_1 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 2304>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    %subview_376 = memref.subview %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %subview_377 = memref.subview %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %subview_378 = memref.subview %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %382 = cinm.compute_block (%arg16 = %alloc_1 : memref<768xf32>, %arg17 = %subview_376 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, %arg18 = %381 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%arg16 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32>
    }
    %subview_379 = memref.subview %arg2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %383 = cinm.compute_block (%arg16 = %subview_379 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_377 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, %arg18 = %381 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_380 = memref.subview %arg3[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %384 = cinm.compute_block (%arg16 = %subview_380 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_378 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, %arg18 = %381 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %384, %subview_380 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %385:2 = cinm.compute_block (%arg16 = %6#0 : f32, %arg17 = %382 : memref<768xf32>, %arg18 = %383 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_751 = arith.constant 1.000000e+04 : f32
      %cst_752 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %760 = arith.remui %arg19, %c48 : index
        %761 = arith.index_cast %760 : index to i64
        %762 = arith.uitofp %761 : i64 to f32
        %763 = arith.divf %762, %cst : f32
        %764 = math.powf %cst_751, %763 : f32
        %765 = arith.divf %cst_752, %764 : f32
        %766 = arith.mulf %arg16, %765 : f32
        %767 = math.cos %766 : f32
        %768 = math.sin %766 : f32
        %769 = arith.addi %arg19, %c1 : index
        %770 = memref.load %arg17[%arg19] : memref<768xf32>
        %771 = memref.load %arg17[%769] : memref<768xf32>
        %772 = arith.mulf %770, %767 : f32
        %773 = arith.mulf %771, %768 : f32
        %774 = arith.subf %772, %773 : f32
        memref.store %774, %arg17[%arg19] : memref<768xf32>
        %775 = arith.mulf %770, %768 : f32
        %776 = arith.mulf %771, %767 : f32
        %777 = arith.addf %775, %776 : f32
        memref.store %777, %arg17[%769] : memref<768xf32>
        %778 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %778 {
          %779 = memref.load %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %780 = memref.load %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
          %781 = arith.mulf %779, %767 : f32
          %782 = arith.mulf %780, %768 : f32
          %783 = arith.subf %781, %782 : f32
          memref.store %783, %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %784 = arith.mulf %779, %768 : f32
          %785 = arith.mulf %780, %767 : f32
          %786 = arith.addf %784, %785 : f32
          memref.store %786, %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      cinm.yield %arg17, %arg18 : memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %385#1, %subview_379 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_381 = memref.subview %arg2[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_382 = memref.subview %arg3[3, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_383 = memref.subview %385#0[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_384 = memref.subview %subview_381[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359296>>
    %386 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_384 : memref<1024x48xf32, strided<[768, 1], offset: 2359296>>, %arg18 = %subview_383 : memref<48xf32, strided<[1]>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359296>>, memref<48xf32, strided<[1]>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %387 = cinm.compute_block (%arg16 = %386 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %388 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %387 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %389 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %388 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %390 = cinm.compute_block (%arg16 = %388 : memref<1024xf32>, %arg17 = %389 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %391 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %390 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_385 = memref.expand_shape %390 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_386 = memref.subview %subview_382[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359296>>
    %392 = cinm.compute_block (%arg16 = %expand_shape_14 : memref<1x48xf32>, %arg17 = %expand_shape_385 : memref<1x1024xf32>, %arg18 = %391 : f32, %arg19 = %subview_386 : memref<1024x48xf32, strided<[768, 1], offset: 2359296>>) -> memref<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359296>>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32>
    }
    %collapse_shape_387 = memref.collapse_shape %392 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
    %subview_388 = memref.subview %381[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    memref.copy %collapse_shape_387, %subview_388 : memref<48xf32> to memref<48xf32, strided<[1]>>
    %subview_389 = memref.subview %385#0[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_390 = memref.subview %subview_381[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359344>>
    %393 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_390 : memref<1024x48xf32, strided<[768, 1], offset: 2359344>>, %arg18 = %subview_389 : memref<48xf32, strided<[1], offset: 48>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359344>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %394 = cinm.compute_block (%arg16 = %393 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %395 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %394 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %396 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %395 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %397 = cinm.compute_block (%arg16 = %395 : memref<1024xf32>, %arg17 = %396 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %398 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %397 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_391 = memref.expand_shape %397 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_392 = memref.subview %subview_382[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359344>>
    %subview_393 = memref.subview %381[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_394 = memref.expand_shape %subview_393 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    %399 = cinm.compute_block (%arg16 = %expand_shape_394 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %expand_shape_391 : memref<1x1024xf32>, %arg18 = %398 : f32, %arg19 = %subview_392 : memref<1024x48xf32, strided<[768, 1], offset: 2359344>>) -> memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359344>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    %collapse_shape_395 = memref.collapse_shape %399 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
    memref.copy %collapse_shape_395, %subview_393 : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
    %subview_396 = memref.subview %385#0[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_397 = memref.subview %subview_381[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359392>>
    %400 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_397 : memref<1024x48xf32, strided<[768, 1], offset: 2359392>>, %arg18 = %subview_396 : memref<48xf32, strided<[1], offset: 96>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359392>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %401 = cinm.compute_block (%arg16 = %400 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %402 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %401 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %403 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %402 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %404 = cinm.compute_block (%arg16 = %402 : memref<1024xf32>, %arg17 = %403 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %405 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %404 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_398 = memref.expand_shape %404 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_399 = memref.subview %subview_382[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359392>>
    %subview_400 = memref.subview %381[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_401 = memref.expand_shape %subview_400 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    %406 = cinm.compute_block (%arg16 = %expand_shape_401 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %expand_shape_398 : memref<1x1024xf32>, %arg18 = %405 : f32, %arg19 = %subview_399 : memref<1024x48xf32, strided<[768, 1], offset: 2359392>>) -> memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359392>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    %collapse_shape_402 = memref.collapse_shape %406 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
    memref.copy %collapse_shape_402, %subview_400 : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
    %subview_403 = memref.subview %385#0[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_404 = memref.subview %subview_381[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359440>>
    %407 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_404 : memref<1024x48xf32, strided<[768, 1], offset: 2359440>>, %arg18 = %subview_403 : memref<48xf32, strided<[1], offset: 144>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359440>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %408 = cinm.compute_block (%arg16 = %407 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %409 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %408 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %410 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %409 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %411 = cinm.compute_block (%arg16 = %409 : memref<1024xf32>, %arg17 = %410 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %412 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %411 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_405 = memref.expand_shape %411 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_406 = memref.subview %subview_382[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359440>>
    %subview_407 = memref.subview %381[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_408 = memref.expand_shape %subview_407 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    %413 = cinm.compute_block (%arg16 = %expand_shape_408 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %expand_shape_405 : memref<1x1024xf32>, %arg18 = %412 : f32, %arg19 = %subview_406 : memref<1024x48xf32, strided<[768, 1], offset: 2359440>>) -> memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359440>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    %collapse_shape_409 = memref.collapse_shape %413 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
    memref.copy %collapse_shape_409, %subview_407 : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
    %subview_410 = memref.subview %385#0[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_411 = memref.subview %subview_381[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359488>>
    %414 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_411 : memref<1024x48xf32, strided<[768, 1], offset: 2359488>>, %arg18 = %subview_410 : memref<48xf32, strided<[1], offset: 192>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359488>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %415 = cinm.compute_block (%arg16 = %414 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %416 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %415 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %417 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %416 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %418 = cinm.compute_block (%arg16 = %416 : memref<1024xf32>, %arg17 = %417 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %419 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %418 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_412 = memref.expand_shape %418 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_413 = memref.subview %subview_382[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359488>>
    %subview_414 = memref.subview %381[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_415 = memref.expand_shape %subview_414 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    %420 = cinm.compute_block (%arg16 = %expand_shape_415 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %expand_shape_412 : memref<1x1024xf32>, %arg18 = %419 : f32, %arg19 = %subview_413 : memref<1024x48xf32, strided<[768, 1], offset: 2359488>>) -> memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359488>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    %collapse_shape_416 = memref.collapse_shape %420 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
    memref.copy %collapse_shape_416, %subview_414 : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
    %subview_417 = memref.subview %385#0[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_418 = memref.subview %subview_381[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359536>>
    %421 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_418 : memref<1024x48xf32, strided<[768, 1], offset: 2359536>>, %arg18 = %subview_417 : memref<48xf32, strided<[1], offset: 240>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359536>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %422 = cinm.compute_block (%arg16 = %421 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %423 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %422 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %424 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %423 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %425 = cinm.compute_block (%arg16 = %423 : memref<1024xf32>, %arg17 = %424 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %426 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %425 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_419 = memref.expand_shape %425 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_420 = memref.subview %subview_382[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359536>>
    %subview_421 = memref.subview %381[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_422 = memref.expand_shape %subview_421 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    %427 = cinm.compute_block (%arg16 = %expand_shape_422 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %expand_shape_419 : memref<1x1024xf32>, %arg18 = %426 : f32, %arg19 = %subview_420 : memref<1024x48xf32, strided<[768, 1], offset: 2359536>>) -> memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359536>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    %collapse_shape_423 = memref.collapse_shape %427 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
    memref.copy %collapse_shape_423, %subview_421 : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
    %subview_424 = memref.subview %385#0[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_425 = memref.subview %subview_381[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359584>>
    %428 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_425 : memref<1024x48xf32, strided<[768, 1], offset: 2359584>>, %arg18 = %subview_424 : memref<48xf32, strided<[1], offset: 288>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359584>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %429 = cinm.compute_block (%arg16 = %428 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %430 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %429 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %431 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %430 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %432 = cinm.compute_block (%arg16 = %430 : memref<1024xf32>, %arg17 = %431 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %433 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %432 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_426 = memref.expand_shape %432 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_427 = memref.subview %subview_382[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359584>>
    %subview_428 = memref.subview %381[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_429 = memref.expand_shape %subview_428 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    %434 = cinm.compute_block (%arg16 = %expand_shape_429 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %expand_shape_426 : memref<1x1024xf32>, %arg18 = %433 : f32, %arg19 = %subview_427 : memref<1024x48xf32, strided<[768, 1], offset: 2359584>>) -> memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359584>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    %collapse_shape_430 = memref.collapse_shape %434 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
    memref.copy %collapse_shape_430, %subview_428 : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
    %subview_431 = memref.subview %385#0[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_432 = memref.subview %subview_381[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359632>>
    %435 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_432 : memref<1024x48xf32, strided<[768, 1], offset: 2359632>>, %arg18 = %subview_431 : memref<48xf32, strided<[1], offset: 336>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359632>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %436 = cinm.compute_block (%arg16 = %435 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %437 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %436 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %438 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %437 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %439 = cinm.compute_block (%arg16 = %437 : memref<1024xf32>, %arg17 = %438 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %440 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %439 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_433 = memref.expand_shape %439 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_434 = memref.subview %subview_382[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359632>>
    %subview_435 = memref.subview %381[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_436 = memref.expand_shape %subview_435 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    %441 = cinm.compute_block (%arg16 = %expand_shape_436 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %expand_shape_433 : memref<1x1024xf32>, %arg18 = %440 : f32, %arg19 = %subview_434 : memref<1024x48xf32, strided<[768, 1], offset: 2359632>>) -> memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359632>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    %collapse_shape_437 = memref.collapse_shape %441 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
    memref.copy %collapse_shape_437, %subview_435 : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
    %subview_438 = memref.subview %385#0[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_439 = memref.subview %subview_381[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359680>>
    %442 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_439 : memref<1024x48xf32, strided<[768, 1], offset: 2359680>>, %arg18 = %subview_438 : memref<48xf32, strided<[1], offset: 384>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359680>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %443 = cinm.compute_block (%arg16 = %442 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %444 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %443 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %445 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %444 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %446 = cinm.compute_block (%arg16 = %444 : memref<1024xf32>, %arg17 = %445 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %447 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %446 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_440 = memref.expand_shape %446 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_441 = memref.subview %subview_382[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359680>>
    %subview_442 = memref.subview %381[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_443 = memref.expand_shape %subview_442 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    %448 = cinm.compute_block (%arg16 = %expand_shape_443 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %expand_shape_440 : memref<1x1024xf32>, %arg18 = %447 : f32, %arg19 = %subview_441 : memref<1024x48xf32, strided<[768, 1], offset: 2359680>>) -> memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359680>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    %collapse_shape_444 = memref.collapse_shape %448 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
    memref.copy %collapse_shape_444, %subview_442 : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
    %subview_445 = memref.subview %385#0[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_446 = memref.subview %subview_381[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359728>>
    %449 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_446 : memref<1024x48xf32, strided<[768, 1], offset: 2359728>>, %arg18 = %subview_445 : memref<48xf32, strided<[1], offset: 432>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359728>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %450 = cinm.compute_block (%arg16 = %449 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %451 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %450 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %452 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %451 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %453 = cinm.compute_block (%arg16 = %451 : memref<1024xf32>, %arg17 = %452 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %454 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %453 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_447 = memref.expand_shape %453 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_448 = memref.subview %subview_382[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359728>>
    %subview_449 = memref.subview %381[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_450 = memref.expand_shape %subview_449 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    %455 = cinm.compute_block (%arg16 = %expand_shape_450 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %expand_shape_447 : memref<1x1024xf32>, %arg18 = %454 : f32, %arg19 = %subview_448 : memref<1024x48xf32, strided<[768, 1], offset: 2359728>>) -> memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359728>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    %collapse_shape_451 = memref.collapse_shape %455 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
    memref.copy %collapse_shape_451, %subview_449 : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
    %subview_452 = memref.subview %385#0[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_453 = memref.subview %subview_381[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359776>>
    %456 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_453 : memref<1024x48xf32, strided<[768, 1], offset: 2359776>>, %arg18 = %subview_452 : memref<48xf32, strided<[1], offset: 480>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359776>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %457 = cinm.compute_block (%arg16 = %456 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %458 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %457 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %459 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %458 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %460 = cinm.compute_block (%arg16 = %458 : memref<1024xf32>, %arg17 = %459 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %461 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %460 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_454 = memref.expand_shape %460 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_455 = memref.subview %subview_382[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359776>>
    %subview_456 = memref.subview %381[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_457 = memref.expand_shape %subview_456 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    %462 = cinm.compute_block (%arg16 = %expand_shape_457 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %expand_shape_454 : memref<1x1024xf32>, %arg18 = %461 : f32, %arg19 = %subview_455 : memref<1024x48xf32, strided<[768, 1], offset: 2359776>>) -> memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359776>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    %collapse_shape_458 = memref.collapse_shape %462 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
    memref.copy %collapse_shape_458, %subview_456 : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
    %subview_459 = memref.subview %385#0[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_460 = memref.subview %subview_381[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359824>>
    %463 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_460 : memref<1024x48xf32, strided<[768, 1], offset: 2359824>>, %arg18 = %subview_459 : memref<48xf32, strided<[1], offset: 528>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359824>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %464 = cinm.compute_block (%arg16 = %463 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %465 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %464 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %466 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %465 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %467 = cinm.compute_block (%arg16 = %465 : memref<1024xf32>, %arg17 = %466 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %468 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %467 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_461 = memref.expand_shape %467 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_462 = memref.subview %subview_382[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359824>>
    %subview_463 = memref.subview %381[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_464 = memref.expand_shape %subview_463 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    %469 = cinm.compute_block (%arg16 = %expand_shape_464 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %expand_shape_461 : memref<1x1024xf32>, %arg18 = %468 : f32, %arg19 = %subview_462 : memref<1024x48xf32, strided<[768, 1], offset: 2359824>>) -> memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359824>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    %collapse_shape_465 = memref.collapse_shape %469 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
    memref.copy %collapse_shape_465, %subview_463 : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
    %subview_466 = memref.subview %385#0[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_467 = memref.subview %subview_381[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359872>>
    %470 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_467 : memref<1024x48xf32, strided<[768, 1], offset: 2359872>>, %arg18 = %subview_466 : memref<48xf32, strided<[1], offset: 576>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359872>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %471 = cinm.compute_block (%arg16 = %470 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %472 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %471 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %473 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %472 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %474 = cinm.compute_block (%arg16 = %472 : memref<1024xf32>, %arg17 = %473 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %475 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %474 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_468 = memref.expand_shape %474 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_469 = memref.subview %subview_382[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359872>>
    %subview_470 = memref.subview %381[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_471 = memref.expand_shape %subview_470 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    %476 = cinm.compute_block (%arg16 = %expand_shape_471 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %expand_shape_468 : memref<1x1024xf32>, %arg18 = %475 : f32, %arg19 = %subview_469 : memref<1024x48xf32, strided<[768, 1], offset: 2359872>>) -> memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359872>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    %collapse_shape_472 = memref.collapse_shape %476 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
    memref.copy %collapse_shape_472, %subview_470 : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
    %subview_473 = memref.subview %385#0[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_474 = memref.subview %subview_381[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359920>>
    %477 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_474 : memref<1024x48xf32, strided<[768, 1], offset: 2359920>>, %arg18 = %subview_473 : memref<48xf32, strided<[1], offset: 624>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359920>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %478 = cinm.compute_block (%arg16 = %477 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %479 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %478 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %480 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %479 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %481 = cinm.compute_block (%arg16 = %479 : memref<1024xf32>, %arg17 = %480 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %482 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %481 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_475 = memref.expand_shape %481 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_476 = memref.subview %subview_382[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359920>>
    %subview_477 = memref.subview %381[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_478 = memref.expand_shape %subview_477 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    %483 = cinm.compute_block (%arg16 = %expand_shape_478 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %expand_shape_475 : memref<1x1024xf32>, %arg18 = %482 : f32, %arg19 = %subview_476 : memref<1024x48xf32, strided<[768, 1], offset: 2359920>>) -> memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359920>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    %collapse_shape_479 = memref.collapse_shape %483 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
    memref.copy %collapse_shape_479, %subview_477 : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
    %subview_480 = memref.subview %385#0[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_481 = memref.subview %subview_381[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359968>>
    %484 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_481 : memref<1024x48xf32, strided<[768, 1], offset: 2359968>>, %arg18 = %subview_480 : memref<48xf32, strided<[1], offset: 672>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2359968>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %485 = cinm.compute_block (%arg16 = %484 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %486 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %485 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %487 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %486 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %488 = cinm.compute_block (%arg16 = %486 : memref<1024xf32>, %arg17 = %487 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %489 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %488 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_482 = memref.expand_shape %488 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_483 = memref.subview %subview_382[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2359968>>
    %subview_484 = memref.subview %381[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_485 = memref.expand_shape %subview_484 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    %490 = cinm.compute_block (%arg16 = %expand_shape_485 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %expand_shape_482 : memref<1x1024xf32>, %arg18 = %489 : f32, %arg19 = %subview_483 : memref<1024x48xf32, strided<[768, 1], offset: 2359968>>) -> memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2359968>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    %collapse_shape_486 = memref.collapse_shape %490 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
    memref.copy %collapse_shape_486, %subview_484 : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
    %subview_487 = memref.subview %385#0[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_488 = memref.subview %subview_381[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2360016>>
    %491 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_488 : memref<1024x48xf32, strided<[768, 1], offset: 2360016>>, %arg18 = %subview_487 : memref<48xf32, strided<[1], offset: 720>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 2360016>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %492 = cinm.compute_block (%arg16 = %491 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %493 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %492 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %494 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %493 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %495 = cinm.compute_block (%arg16 = %493 : memref<1024xf32>, %arg17 = %494 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %496 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %495 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_489 = memref.expand_shape %495 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_490 = memref.subview %subview_382[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 2359296>> to memref<1024x48xf32, strided<[768, 1], offset: 2360016>>
    %subview_491 = memref.subview %381[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_492 = memref.expand_shape %subview_491 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    %497 = cinm.compute_block (%arg16 = %expand_shape_492 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %expand_shape_489 : memref<1x1024xf32>, %arg18 = %496 : f32, %arg19 = %subview_490 : memref<1024x48xf32, strided<[768, 1], offset: 2360016>>) -> memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 2360016>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    %collapse_shape_493 = memref.collapse_shape %497 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
    memref.copy %collapse_shape_493, %subview_491 : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
    %subview_494 = memref.subview %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 1769472>>
    %498 = cinm.compute_block (%arg16 = %subview_494 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, %arg17 = %381 : memref<768xf32>, %arg18 = %378#1 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 1769472>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.addf %out, %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg18 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_495 = memref.subview %arg13[3, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 2304>>
    %499 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %498 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %500 = cinm.compute_block (%arg16 = %499 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %501 = cinm.compute_block (%arg16 = %498 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %500 : f32, %arg18 = %subview_495 : memref<768xf32, strided<[1], offset: 2304>>, %arg19 = %381 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 2304>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    memref.copy %501, %381 : memref<768xf32> to memref<768xf32>
    %subview_496 = memref.subview %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
    %subview_497 = memref.subview %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 4718592>>
    %502 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_496 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, %arg18 = %381 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %503 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_497 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, %arg18 = %381 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 4718592>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %subview_498 = memref.subview %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>
    %504:2 = cinm.compute_block (%arg16 = %subview_498 : memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>, %arg17 = %503 : memref<2048xf32>, %arg18 = %502 : memref<2048xf32>, %arg19 = %498 : memref<768xf32, strided<[1], offset: ?>>) -> memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_2 : memref<f32>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17, %760 : memref<768x2048xf32, strided<[2048, 1], offset: 4718592>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32, %out_753: f32):
        %761 = arith.negf %out : f32
        %762 = math.exp %761 : f32
        %763 = arith.addf %762, %in_752 : f32
        %764 = arith.divf %in_752, %763 : f32
        %765 = arith.mulf %out, %764 : f32
        %766 = arith.mulf %765, %in_751 : f32
        %767 = arith.mulf %in, %766 : f32
        %768 = arith.addf %out_753, %767 : f32
        linalg.yield %766, %768 : f32, f32
      }
      cinm.yield %arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_499 = memref.subview %arg5[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
    %505 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %504#1 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %506 = cinm.compute_block (%arg16 = %505 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %507 = cinm.compute_block (%arg16 = %504#1 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %506 : f32, %arg18 = %subview_499 : memref<768xf32, strided<[1], offset: 3072>>, %arg19 = %alloc_1 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3072>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    %subview_500 = memref.subview %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_501 = memref.subview %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %subview_502 = memref.subview %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %508 = cinm.compute_block (%arg16 = %alloc_1 : memref<768xf32>, %arg17 = %subview_500 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, %arg18 = %507 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%arg16 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32>
    }
    %subview_503 = memref.subview %arg2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %509 = cinm.compute_block (%arg16 = %subview_503 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_501 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, %arg18 = %507 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_504 = memref.subview %arg3[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %510 = cinm.compute_block (%arg16 = %subview_504 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_502 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, %arg18 = %507 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %510, %subview_504 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %511:2 = cinm.compute_block (%arg16 = %6#0 : f32, %arg17 = %508 : memref<768xf32>, %arg18 = %509 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_751 = arith.constant 1.000000e+04 : f32
      %cst_752 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %760 = arith.remui %arg19, %c48 : index
        %761 = arith.index_cast %760 : index to i64
        %762 = arith.uitofp %761 : i64 to f32
        %763 = arith.divf %762, %cst : f32
        %764 = math.powf %cst_751, %763 : f32
        %765 = arith.divf %cst_752, %764 : f32
        %766 = arith.mulf %arg16, %765 : f32
        %767 = math.cos %766 : f32
        %768 = math.sin %766 : f32
        %769 = arith.addi %arg19, %c1 : index
        %770 = memref.load %arg17[%arg19] : memref<768xf32>
        %771 = memref.load %arg17[%769] : memref<768xf32>
        %772 = arith.mulf %770, %767 : f32
        %773 = arith.mulf %771, %768 : f32
        %774 = arith.subf %772, %773 : f32
        memref.store %774, %arg17[%arg19] : memref<768xf32>
        %775 = arith.mulf %770, %768 : f32
        %776 = arith.mulf %771, %767 : f32
        %777 = arith.addf %775, %776 : f32
        memref.store %777, %arg17[%769] : memref<768xf32>
        %778 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %778 {
          %779 = memref.load %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %780 = memref.load %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
          %781 = arith.mulf %779, %767 : f32
          %782 = arith.mulf %780, %768 : f32
          %783 = arith.subf %781, %782 : f32
          memref.store %783, %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %784 = arith.mulf %779, %768 : f32
          %785 = arith.mulf %780, %767 : f32
          %786 = arith.addf %784, %785 : f32
          memref.store %786, %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      cinm.yield %arg17, %arg18 : memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %511#1, %subview_503 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_505 = memref.subview %arg2[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_506 = memref.subview %arg3[4, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3145728>>
    %subview_507 = memref.subview %511#0[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_508 = memref.subview %subview_505[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145728>>
    %512 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_508 : memref<1024x48xf32, strided<[768, 1], offset: 3145728>>, %arg18 = %subview_507 : memref<48xf32, strided<[1]>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3145728>>, memref<48xf32, strided<[1]>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %513 = cinm.compute_block (%arg16 = %512 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %514 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %513 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %515 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %514 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %516 = cinm.compute_block (%arg16 = %514 : memref<1024xf32>, %arg17 = %515 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %517 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %516 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_509 = memref.expand_shape %516 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_510 = memref.subview %subview_506[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145728>>
    %518 = cinm.compute_block (%arg16 = %expand_shape_14 : memref<1x48xf32>, %arg17 = %expand_shape_509 : memref<1x1024xf32>, %arg18 = %517 : f32, %arg19 = %subview_510 : memref<1024x48xf32, strided<[768, 1], offset: 3145728>>) -> memref<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145728>>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32>
    }
    %collapse_shape_511 = memref.collapse_shape %518 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
    %subview_512 = memref.subview %507[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    memref.copy %collapse_shape_511, %subview_512 : memref<48xf32> to memref<48xf32, strided<[1]>>
    %subview_513 = memref.subview %511#0[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_514 = memref.subview %subview_505[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145776>>
    %519 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_514 : memref<1024x48xf32, strided<[768, 1], offset: 3145776>>, %arg18 = %subview_513 : memref<48xf32, strided<[1], offset: 48>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3145776>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %520 = cinm.compute_block (%arg16 = %519 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %521 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %520 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %522 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %521 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %523 = cinm.compute_block (%arg16 = %521 : memref<1024xf32>, %arg17 = %522 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %524 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %523 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_515 = memref.expand_shape %523 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_516 = memref.subview %subview_506[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145776>>
    %subview_517 = memref.subview %507[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_518 = memref.expand_shape %subview_517 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    %525 = cinm.compute_block (%arg16 = %expand_shape_518 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %expand_shape_515 : memref<1x1024xf32>, %arg18 = %524 : f32, %arg19 = %subview_516 : memref<1024x48xf32, strided<[768, 1], offset: 3145776>>) -> memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145776>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    %collapse_shape_519 = memref.collapse_shape %525 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
    memref.copy %collapse_shape_519, %subview_517 : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
    %subview_520 = memref.subview %511#0[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_521 = memref.subview %subview_505[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145824>>
    %526 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_521 : memref<1024x48xf32, strided<[768, 1], offset: 3145824>>, %arg18 = %subview_520 : memref<48xf32, strided<[1], offset: 96>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3145824>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %527 = cinm.compute_block (%arg16 = %526 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %528 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %527 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %529 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %528 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %530 = cinm.compute_block (%arg16 = %528 : memref<1024xf32>, %arg17 = %529 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %531 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %530 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_522 = memref.expand_shape %530 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_523 = memref.subview %subview_506[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145824>>
    %subview_524 = memref.subview %507[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_525 = memref.expand_shape %subview_524 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    %532 = cinm.compute_block (%arg16 = %expand_shape_525 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %expand_shape_522 : memref<1x1024xf32>, %arg18 = %531 : f32, %arg19 = %subview_523 : memref<1024x48xf32, strided<[768, 1], offset: 3145824>>) -> memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145824>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    %collapse_shape_526 = memref.collapse_shape %532 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
    memref.copy %collapse_shape_526, %subview_524 : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
    %subview_527 = memref.subview %511#0[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_528 = memref.subview %subview_505[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145872>>
    %533 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_528 : memref<1024x48xf32, strided<[768, 1], offset: 3145872>>, %arg18 = %subview_527 : memref<48xf32, strided<[1], offset: 144>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3145872>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %534 = cinm.compute_block (%arg16 = %533 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %535 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %534 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %536 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %535 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %537 = cinm.compute_block (%arg16 = %535 : memref<1024xf32>, %arg17 = %536 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %538 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %537 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_529 = memref.expand_shape %537 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_530 = memref.subview %subview_506[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145872>>
    %subview_531 = memref.subview %507[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_532 = memref.expand_shape %subview_531 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    %539 = cinm.compute_block (%arg16 = %expand_shape_532 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %expand_shape_529 : memref<1x1024xf32>, %arg18 = %538 : f32, %arg19 = %subview_530 : memref<1024x48xf32, strided<[768, 1], offset: 3145872>>) -> memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145872>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    %collapse_shape_533 = memref.collapse_shape %539 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
    memref.copy %collapse_shape_533, %subview_531 : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
    %subview_534 = memref.subview %511#0[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_535 = memref.subview %subview_505[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145920>>
    %540 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_535 : memref<1024x48xf32, strided<[768, 1], offset: 3145920>>, %arg18 = %subview_534 : memref<48xf32, strided<[1], offset: 192>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3145920>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %541 = cinm.compute_block (%arg16 = %540 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %542 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %541 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %543 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %542 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %544 = cinm.compute_block (%arg16 = %542 : memref<1024xf32>, %arg17 = %543 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %545 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %544 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_536 = memref.expand_shape %544 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_537 = memref.subview %subview_506[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145920>>
    %subview_538 = memref.subview %507[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_539 = memref.expand_shape %subview_538 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    %546 = cinm.compute_block (%arg16 = %expand_shape_539 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %expand_shape_536 : memref<1x1024xf32>, %arg18 = %545 : f32, %arg19 = %subview_537 : memref<1024x48xf32, strided<[768, 1], offset: 3145920>>) -> memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145920>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    %collapse_shape_540 = memref.collapse_shape %546 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
    memref.copy %collapse_shape_540, %subview_538 : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
    %subview_541 = memref.subview %511#0[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_542 = memref.subview %subview_505[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145968>>
    %547 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_542 : memref<1024x48xf32, strided<[768, 1], offset: 3145968>>, %arg18 = %subview_541 : memref<48xf32, strided<[1], offset: 240>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3145968>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %548 = cinm.compute_block (%arg16 = %547 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %549 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %548 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %550 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %549 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %551 = cinm.compute_block (%arg16 = %549 : memref<1024xf32>, %arg17 = %550 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %552 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %551 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_543 = memref.expand_shape %551 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_544 = memref.subview %subview_506[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3145968>>
    %subview_545 = memref.subview %507[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_546 = memref.expand_shape %subview_545 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    %553 = cinm.compute_block (%arg16 = %expand_shape_546 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %expand_shape_543 : memref<1x1024xf32>, %arg18 = %552 : f32, %arg19 = %subview_544 : memref<1024x48xf32, strided<[768, 1], offset: 3145968>>) -> memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3145968>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    %collapse_shape_547 = memref.collapse_shape %553 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
    memref.copy %collapse_shape_547, %subview_545 : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
    %subview_548 = memref.subview %511#0[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_549 = memref.subview %subview_505[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146016>>
    %554 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_549 : memref<1024x48xf32, strided<[768, 1], offset: 3146016>>, %arg18 = %subview_548 : memref<48xf32, strided<[1], offset: 288>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146016>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %555 = cinm.compute_block (%arg16 = %554 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %556 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %555 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %557 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %556 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %558 = cinm.compute_block (%arg16 = %556 : memref<1024xf32>, %arg17 = %557 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %559 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %558 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_550 = memref.expand_shape %558 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_551 = memref.subview %subview_506[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146016>>
    %subview_552 = memref.subview %507[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_553 = memref.expand_shape %subview_552 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    %560 = cinm.compute_block (%arg16 = %expand_shape_553 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %expand_shape_550 : memref<1x1024xf32>, %arg18 = %559 : f32, %arg19 = %subview_551 : memref<1024x48xf32, strided<[768, 1], offset: 3146016>>) -> memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146016>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    %collapse_shape_554 = memref.collapse_shape %560 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
    memref.copy %collapse_shape_554, %subview_552 : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
    %subview_555 = memref.subview %511#0[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_556 = memref.subview %subview_505[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146064>>
    %561 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_556 : memref<1024x48xf32, strided<[768, 1], offset: 3146064>>, %arg18 = %subview_555 : memref<48xf32, strided<[1], offset: 336>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146064>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %562 = cinm.compute_block (%arg16 = %561 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %563 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %562 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %564 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %563 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %565 = cinm.compute_block (%arg16 = %563 : memref<1024xf32>, %arg17 = %564 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %566 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %565 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_557 = memref.expand_shape %565 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_558 = memref.subview %subview_506[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146064>>
    %subview_559 = memref.subview %507[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_560 = memref.expand_shape %subview_559 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    %567 = cinm.compute_block (%arg16 = %expand_shape_560 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %expand_shape_557 : memref<1x1024xf32>, %arg18 = %566 : f32, %arg19 = %subview_558 : memref<1024x48xf32, strided<[768, 1], offset: 3146064>>) -> memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146064>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    %collapse_shape_561 = memref.collapse_shape %567 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
    memref.copy %collapse_shape_561, %subview_559 : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
    %subview_562 = memref.subview %511#0[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_563 = memref.subview %subview_505[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146112>>
    %568 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_563 : memref<1024x48xf32, strided<[768, 1], offset: 3146112>>, %arg18 = %subview_562 : memref<48xf32, strided<[1], offset: 384>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146112>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %569 = cinm.compute_block (%arg16 = %568 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %570 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %569 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %571 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %570 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %572 = cinm.compute_block (%arg16 = %570 : memref<1024xf32>, %arg17 = %571 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %573 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %572 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_564 = memref.expand_shape %572 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_565 = memref.subview %subview_506[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146112>>
    %subview_566 = memref.subview %507[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_567 = memref.expand_shape %subview_566 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    %574 = cinm.compute_block (%arg16 = %expand_shape_567 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %expand_shape_564 : memref<1x1024xf32>, %arg18 = %573 : f32, %arg19 = %subview_565 : memref<1024x48xf32, strided<[768, 1], offset: 3146112>>) -> memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146112>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    %collapse_shape_568 = memref.collapse_shape %574 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
    memref.copy %collapse_shape_568, %subview_566 : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
    %subview_569 = memref.subview %511#0[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_570 = memref.subview %subview_505[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146160>>
    %575 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_570 : memref<1024x48xf32, strided<[768, 1], offset: 3146160>>, %arg18 = %subview_569 : memref<48xf32, strided<[1], offset: 432>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146160>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %576 = cinm.compute_block (%arg16 = %575 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %577 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %576 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %578 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %577 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %579 = cinm.compute_block (%arg16 = %577 : memref<1024xf32>, %arg17 = %578 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %580 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %579 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_571 = memref.expand_shape %579 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_572 = memref.subview %subview_506[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146160>>
    %subview_573 = memref.subview %507[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_574 = memref.expand_shape %subview_573 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    %581 = cinm.compute_block (%arg16 = %expand_shape_574 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %expand_shape_571 : memref<1x1024xf32>, %arg18 = %580 : f32, %arg19 = %subview_572 : memref<1024x48xf32, strided<[768, 1], offset: 3146160>>) -> memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146160>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    %collapse_shape_575 = memref.collapse_shape %581 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
    memref.copy %collapse_shape_575, %subview_573 : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
    %subview_576 = memref.subview %511#0[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_577 = memref.subview %subview_505[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146208>>
    %582 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_577 : memref<1024x48xf32, strided<[768, 1], offset: 3146208>>, %arg18 = %subview_576 : memref<48xf32, strided<[1], offset: 480>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146208>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %583 = cinm.compute_block (%arg16 = %582 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %584 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %583 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %585 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %584 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %586 = cinm.compute_block (%arg16 = %584 : memref<1024xf32>, %arg17 = %585 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %587 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %586 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_578 = memref.expand_shape %586 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_579 = memref.subview %subview_506[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146208>>
    %subview_580 = memref.subview %507[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_581 = memref.expand_shape %subview_580 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    %588 = cinm.compute_block (%arg16 = %expand_shape_581 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %expand_shape_578 : memref<1x1024xf32>, %arg18 = %587 : f32, %arg19 = %subview_579 : memref<1024x48xf32, strided<[768, 1], offset: 3146208>>) -> memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146208>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    %collapse_shape_582 = memref.collapse_shape %588 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
    memref.copy %collapse_shape_582, %subview_580 : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
    %subview_583 = memref.subview %511#0[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_584 = memref.subview %subview_505[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146256>>
    %589 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_584 : memref<1024x48xf32, strided<[768, 1], offset: 3146256>>, %arg18 = %subview_583 : memref<48xf32, strided<[1], offset: 528>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146256>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %590 = cinm.compute_block (%arg16 = %589 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %591 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %590 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %592 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %591 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %593 = cinm.compute_block (%arg16 = %591 : memref<1024xf32>, %arg17 = %592 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %594 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %593 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_585 = memref.expand_shape %593 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_586 = memref.subview %subview_506[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146256>>
    %subview_587 = memref.subview %507[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_588 = memref.expand_shape %subview_587 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    %595 = cinm.compute_block (%arg16 = %expand_shape_588 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %expand_shape_585 : memref<1x1024xf32>, %arg18 = %594 : f32, %arg19 = %subview_586 : memref<1024x48xf32, strided<[768, 1], offset: 3146256>>) -> memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146256>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    %collapse_shape_589 = memref.collapse_shape %595 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
    memref.copy %collapse_shape_589, %subview_587 : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
    %subview_590 = memref.subview %511#0[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_591 = memref.subview %subview_505[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146304>>
    %596 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_591 : memref<1024x48xf32, strided<[768, 1], offset: 3146304>>, %arg18 = %subview_590 : memref<48xf32, strided<[1], offset: 576>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146304>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %597 = cinm.compute_block (%arg16 = %596 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %598 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %597 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %599 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %598 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %600 = cinm.compute_block (%arg16 = %598 : memref<1024xf32>, %arg17 = %599 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %601 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %600 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_592 = memref.expand_shape %600 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_593 = memref.subview %subview_506[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146304>>
    %subview_594 = memref.subview %507[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_595 = memref.expand_shape %subview_594 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    %602 = cinm.compute_block (%arg16 = %expand_shape_595 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %expand_shape_592 : memref<1x1024xf32>, %arg18 = %601 : f32, %arg19 = %subview_593 : memref<1024x48xf32, strided<[768, 1], offset: 3146304>>) -> memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146304>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    %collapse_shape_596 = memref.collapse_shape %602 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
    memref.copy %collapse_shape_596, %subview_594 : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
    %subview_597 = memref.subview %511#0[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_598 = memref.subview %subview_505[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146352>>
    %603 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_598 : memref<1024x48xf32, strided<[768, 1], offset: 3146352>>, %arg18 = %subview_597 : memref<48xf32, strided<[1], offset: 624>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146352>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %604 = cinm.compute_block (%arg16 = %603 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %605 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %604 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %606 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %605 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %607 = cinm.compute_block (%arg16 = %605 : memref<1024xf32>, %arg17 = %606 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %608 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %607 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_599 = memref.expand_shape %607 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_600 = memref.subview %subview_506[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146352>>
    %subview_601 = memref.subview %507[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_602 = memref.expand_shape %subview_601 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    %609 = cinm.compute_block (%arg16 = %expand_shape_602 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %expand_shape_599 : memref<1x1024xf32>, %arg18 = %608 : f32, %arg19 = %subview_600 : memref<1024x48xf32, strided<[768, 1], offset: 3146352>>) -> memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146352>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    %collapse_shape_603 = memref.collapse_shape %609 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
    memref.copy %collapse_shape_603, %subview_601 : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
    %subview_604 = memref.subview %511#0[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_605 = memref.subview %subview_505[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146400>>
    %610 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_605 : memref<1024x48xf32, strided<[768, 1], offset: 3146400>>, %arg18 = %subview_604 : memref<48xf32, strided<[1], offset: 672>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146400>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %611 = cinm.compute_block (%arg16 = %610 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %612 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %611 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %613 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %612 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %614 = cinm.compute_block (%arg16 = %612 : memref<1024xf32>, %arg17 = %613 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %615 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %614 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_606 = memref.expand_shape %614 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_607 = memref.subview %subview_506[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146400>>
    %subview_608 = memref.subview %507[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_609 = memref.expand_shape %subview_608 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    %616 = cinm.compute_block (%arg16 = %expand_shape_609 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %expand_shape_606 : memref<1x1024xf32>, %arg18 = %615 : f32, %arg19 = %subview_607 : memref<1024x48xf32, strided<[768, 1], offset: 3146400>>) -> memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146400>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    %collapse_shape_610 = memref.collapse_shape %616 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
    memref.copy %collapse_shape_610, %subview_608 : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
    %subview_611 = memref.subview %511#0[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_612 = memref.subview %subview_505[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146448>>
    %617 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_612 : memref<1024x48xf32, strided<[768, 1], offset: 3146448>>, %arg18 = %subview_611 : memref<48xf32, strided<[1], offset: 720>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3146448>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %618 = cinm.compute_block (%arg16 = %617 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %619 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %618 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %620 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %619 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %621 = cinm.compute_block (%arg16 = %619 : memref<1024xf32>, %arg17 = %620 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %622 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %621 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_613 = memref.expand_shape %621 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_614 = memref.subview %subview_506[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3145728>> to memref<1024x48xf32, strided<[768, 1], offset: 3146448>>
    %subview_615 = memref.subview %507[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_616 = memref.expand_shape %subview_615 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    %623 = cinm.compute_block (%arg16 = %expand_shape_616 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %expand_shape_613 : memref<1x1024xf32>, %arg18 = %622 : f32, %arg19 = %subview_614 : memref<1024x48xf32, strided<[768, 1], offset: 3146448>>) -> memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3146448>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    %collapse_shape_617 = memref.collapse_shape %623 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
    memref.copy %collapse_shape_617, %subview_615 : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
    %subview_618 = memref.subview %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2359296>>
    %624 = cinm.compute_block (%arg16 = %subview_618 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, %arg17 = %507 : memref<768xf32>, %arg18 = %504#1 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 2359296>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.addf %out, %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg18 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_619 = memref.subview %arg13[4, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3072>>
    %625 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %624 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %626 = cinm.compute_block (%arg16 = %625 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %627 = cinm.compute_block (%arg16 = %624 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %626 : f32, %arg18 = %subview_619 : memref<768xf32, strided<[1], offset: 3072>>, %arg19 = %507 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3072>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    memref.copy %627, %507 : memref<768xf32> to memref<768xf32>
    %subview_620 = memref.subview %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
    %subview_621 = memref.subview %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 6291456>>
    %628 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_620 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, %arg18 = %507 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %629 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_621 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, %arg18 = %507 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 6291456>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %subview_622 = memref.subview %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>
    %630:2 = cinm.compute_block (%arg16 = %subview_622 : memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>, %arg17 = %629 : memref<2048xf32>, %arg18 = %628 : memref<2048xf32>, %arg19 = %624 : memref<768xf32, strided<[1], offset: ?>>) -> memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_2 : memref<f32>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17, %760 : memref<768x2048xf32, strided<[2048, 1], offset: 6291456>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32, %out_753: f32):
        %761 = arith.negf %out : f32
        %762 = math.exp %761 : f32
        %763 = arith.addf %762, %in_752 : f32
        %764 = arith.divf %in_752, %763 : f32
        %765 = arith.mulf %out, %764 : f32
        %766 = arith.mulf %765, %in_751 : f32
        %767 = arith.mulf %in, %766 : f32
        %768 = arith.addf %out_753, %767 : f32
        linalg.yield %766, %768 : f32, f32
      }
      cinm.yield %arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_623 = memref.subview %arg5[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
    %631 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %630#1 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %632 = cinm.compute_block (%arg16 = %631 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %633 = cinm.compute_block (%arg16 = %630#1 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %632 : f32, %arg18 = %subview_623 : memref<768xf32, strided<[1], offset: 3840>>, %arg19 = %alloc_1 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3840>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    %subview_624 = memref.subview %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %subview_625 = memref.subview %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %subview_626 = memref.subview %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %634 = cinm.compute_block (%arg16 = %alloc_1 : memref<768xf32>, %arg17 = %subview_624 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, %arg18 = %633 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%arg16 : memref<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32>
    }
    %subview_627 = memref.subview %arg2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %635 = cinm.compute_block (%arg16 = %subview_627 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_625 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, %arg18 = %633 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_628 = memref.subview %arg3[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    %636 = cinm.compute_block (%arg16 = %subview_628 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %subview_626 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, %arg18 = %633 : memref<768xf32>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%arg16 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %636, %subview_628 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %637:2 = cinm.compute_block (%arg16 = %6#0 : f32, %arg17 = %634 : memref<768xf32>, %arg18 = %635 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %c48 = arith.constant 48 : index
      %cst = arith.constant 4.800000e+01 : f32
      %cst_751 = arith.constant 1.000000e+04 : f32
      %cst_752 = arith.constant 1.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c768 = arith.constant 768 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg19 = %c0 to %c768 step %c2 {
        %760 = arith.remui %arg19, %c48 : index
        %761 = arith.index_cast %760 : index to i64
        %762 = arith.uitofp %761 : i64 to f32
        %763 = arith.divf %762, %cst : f32
        %764 = math.powf %cst_751, %763 : f32
        %765 = arith.divf %cst_752, %764 : f32
        %766 = arith.mulf %arg16, %765 : f32
        %767 = math.cos %766 : f32
        %768 = math.sin %766 : f32
        %769 = arith.addi %arg19, %c1 : index
        %770 = memref.load %arg17[%arg19] : memref<768xf32>
        %771 = memref.load %arg17[%769] : memref<768xf32>
        %772 = arith.mulf %770, %767 : f32
        %773 = arith.mulf %771, %768 : f32
        %774 = arith.subf %772, %773 : f32
        memref.store %774, %arg17[%arg19] : memref<768xf32>
        %775 = arith.mulf %770, %768 : f32
        %776 = arith.mulf %771, %767 : f32
        %777 = arith.addf %775, %776 : f32
        memref.store %777, %arg17[%769] : memref<768xf32>
        %778 = arith.cmpi ult, %arg19, %c768 : index
        scf.if %778 {
          %779 = memref.load %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %780 = memref.load %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
          %781 = arith.mulf %779, %767 : f32
          %782 = arith.mulf %780, %768 : f32
          %783 = arith.subf %781, %782 : f32
          memref.store %783, %arg18[%arg19] : memref<768xf32, strided<[1], offset: ?>>
          %784 = arith.mulf %779, %768 : f32
          %785 = arith.mulf %780, %767 : f32
          %786 = arith.addf %784, %785 : f32
          memref.store %786, %arg18[%769] : memref<768xf32, strided<[1], offset: ?>>
        }
      }
      cinm.yield %arg17, %arg18 : memref<768xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    memref.copy %637#1, %subview_627 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_629 = memref.subview %arg2[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
    %subview_630 = memref.subview %arg3[5, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: 3932160>>
    %subview_631 = memref.subview %637#0[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    %subview_632 = memref.subview %subview_629[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932160>>
    %638 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_632 : memref<1024x48xf32, strided<[768, 1], offset: 3932160>>, %arg18 = %subview_631 : memref<48xf32, strided<[1]>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932160>>, memref<48xf32, strided<[1]>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %639 = cinm.compute_block (%arg16 = %638 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %640 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %639 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %641 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %640 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %642 = cinm.compute_block (%arg16 = %640 : memref<1024xf32>, %arg17 = %641 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %643 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %642 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_633 = memref.expand_shape %642 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_634 = memref.subview %subview_630[0, 0] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932160>>
    %644 = cinm.compute_block (%arg16 = %expand_shape_14 : memref<1x48xf32>, %arg17 = %expand_shape_633 : memref<1x1024xf32>, %arg18 = %643 : f32, %arg19 = %subview_634 : memref<1024x48xf32, strided<[768, 1], offset: 3932160>>) -> memref<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932160>>) outs(%arg16 : memref<1x48xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32>
    }
    %collapse_shape_635 = memref.collapse_shape %644 [[0, 1]] : memref<1x48xf32> into memref<48xf32>
    %subview_636 = memref.subview %633[0] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1]>>
    memref.copy %collapse_shape_635, %subview_636 : memref<48xf32> to memref<48xf32, strided<[1]>>
    %subview_637 = memref.subview %637#0[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %subview_638 = memref.subview %subview_629[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932208>>
    %645 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_638 : memref<1024x48xf32, strided<[768, 1], offset: 3932208>>, %arg18 = %subview_637 : memref<48xf32, strided<[1], offset: 48>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932208>>, memref<48xf32, strided<[1], offset: 48>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %646 = cinm.compute_block (%arg16 = %645 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %647 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %646 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %648 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %647 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %649 = cinm.compute_block (%arg16 = %647 : memref<1024xf32>, %arg17 = %648 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %650 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %649 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_639 = memref.expand_shape %649 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_640 = memref.subview %subview_630[0, 48] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932208>>
    %subview_641 = memref.subview %633[48] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 48>>
    %expand_shape_642 = memref.expand_shape %subview_641 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 48>> into memref<1x48xf32, strided<[48, 1], offset: 48>>
    %651 = cinm.compute_block (%arg16 = %expand_shape_642 : memref<1x48xf32, strided<[48, 1], offset: 48>>, %arg17 = %expand_shape_639 : memref<1x1024xf32>, %arg18 = %650 : f32, %arg19 = %subview_640 : memref<1024x48xf32, strided<[768, 1], offset: 3932208>>) -> memref<1x48xf32, strided<[48, 1], offset: 48>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932208>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 48>>
    }
    %collapse_shape_643 = memref.collapse_shape %651 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 48>> into memref<48xf32, strided<[1], offset: 48>>
    memref.copy %collapse_shape_643, %subview_641 : memref<48xf32, strided<[1], offset: 48>> to memref<48xf32, strided<[1], offset: 48>>
    %subview_644 = memref.subview %637#0[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %subview_645 = memref.subview %subview_629[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932256>>
    %652 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_645 : memref<1024x48xf32, strided<[768, 1], offset: 3932256>>, %arg18 = %subview_644 : memref<48xf32, strided<[1], offset: 96>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932256>>, memref<48xf32, strided<[1], offset: 96>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %653 = cinm.compute_block (%arg16 = %652 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %654 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %653 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %655 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %654 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %656 = cinm.compute_block (%arg16 = %654 : memref<1024xf32>, %arg17 = %655 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %657 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %656 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_646 = memref.expand_shape %656 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_647 = memref.subview %subview_630[0, 96] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932256>>
    %subview_648 = memref.subview %633[96] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 96>>
    %expand_shape_649 = memref.expand_shape %subview_648 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 96>> into memref<1x48xf32, strided<[48, 1], offset: 96>>
    %658 = cinm.compute_block (%arg16 = %expand_shape_649 : memref<1x48xf32, strided<[48, 1], offset: 96>>, %arg17 = %expand_shape_646 : memref<1x1024xf32>, %arg18 = %657 : f32, %arg19 = %subview_647 : memref<1024x48xf32, strided<[768, 1], offset: 3932256>>) -> memref<1x48xf32, strided<[48, 1], offset: 96>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932256>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 96>>
    }
    %collapse_shape_650 = memref.collapse_shape %658 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 96>> into memref<48xf32, strided<[1], offset: 96>>
    memref.copy %collapse_shape_650, %subview_648 : memref<48xf32, strided<[1], offset: 96>> to memref<48xf32, strided<[1], offset: 96>>
    %subview_651 = memref.subview %637#0[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %subview_652 = memref.subview %subview_629[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932304>>
    %659 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_652 : memref<1024x48xf32, strided<[768, 1], offset: 3932304>>, %arg18 = %subview_651 : memref<48xf32, strided<[1], offset: 144>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932304>>, memref<48xf32, strided<[1], offset: 144>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %660 = cinm.compute_block (%arg16 = %659 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %661 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %660 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %662 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %661 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %663 = cinm.compute_block (%arg16 = %661 : memref<1024xf32>, %arg17 = %662 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %664 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %663 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_653 = memref.expand_shape %663 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_654 = memref.subview %subview_630[0, 144] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932304>>
    %subview_655 = memref.subview %633[144] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 144>>
    %expand_shape_656 = memref.expand_shape %subview_655 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 144>> into memref<1x48xf32, strided<[48, 1], offset: 144>>
    %665 = cinm.compute_block (%arg16 = %expand_shape_656 : memref<1x48xf32, strided<[48, 1], offset: 144>>, %arg17 = %expand_shape_653 : memref<1x1024xf32>, %arg18 = %664 : f32, %arg19 = %subview_654 : memref<1024x48xf32, strided<[768, 1], offset: 3932304>>) -> memref<1x48xf32, strided<[48, 1], offset: 144>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932304>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 144>>
    }
    %collapse_shape_657 = memref.collapse_shape %665 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 144>> into memref<48xf32, strided<[1], offset: 144>>
    memref.copy %collapse_shape_657, %subview_655 : memref<48xf32, strided<[1], offset: 144>> to memref<48xf32, strided<[1], offset: 144>>
    %subview_658 = memref.subview %637#0[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %subview_659 = memref.subview %subview_629[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932352>>
    %666 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_659 : memref<1024x48xf32, strided<[768, 1], offset: 3932352>>, %arg18 = %subview_658 : memref<48xf32, strided<[1], offset: 192>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932352>>, memref<48xf32, strided<[1], offset: 192>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %667 = cinm.compute_block (%arg16 = %666 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %668 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %667 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %669 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %668 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %670 = cinm.compute_block (%arg16 = %668 : memref<1024xf32>, %arg17 = %669 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %671 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %670 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_660 = memref.expand_shape %670 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_661 = memref.subview %subview_630[0, 192] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932352>>
    %subview_662 = memref.subview %633[192] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 192>>
    %expand_shape_663 = memref.expand_shape %subview_662 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 192>> into memref<1x48xf32, strided<[48, 1], offset: 192>>
    %672 = cinm.compute_block (%arg16 = %expand_shape_663 : memref<1x48xf32, strided<[48, 1], offset: 192>>, %arg17 = %expand_shape_660 : memref<1x1024xf32>, %arg18 = %671 : f32, %arg19 = %subview_661 : memref<1024x48xf32, strided<[768, 1], offset: 3932352>>) -> memref<1x48xf32, strided<[48, 1], offset: 192>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932352>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 192>>
    }
    %collapse_shape_664 = memref.collapse_shape %672 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 192>> into memref<48xf32, strided<[1], offset: 192>>
    memref.copy %collapse_shape_664, %subview_662 : memref<48xf32, strided<[1], offset: 192>> to memref<48xf32, strided<[1], offset: 192>>
    %subview_665 = memref.subview %637#0[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %subview_666 = memref.subview %subview_629[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932400>>
    %673 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_666 : memref<1024x48xf32, strided<[768, 1], offset: 3932400>>, %arg18 = %subview_665 : memref<48xf32, strided<[1], offset: 240>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932400>>, memref<48xf32, strided<[1], offset: 240>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %674 = cinm.compute_block (%arg16 = %673 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %675 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %674 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %676 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %675 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %677 = cinm.compute_block (%arg16 = %675 : memref<1024xf32>, %arg17 = %676 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %678 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %677 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_667 = memref.expand_shape %677 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_668 = memref.subview %subview_630[0, 240] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932400>>
    %subview_669 = memref.subview %633[240] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 240>>
    %expand_shape_670 = memref.expand_shape %subview_669 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 240>> into memref<1x48xf32, strided<[48, 1], offset: 240>>
    %679 = cinm.compute_block (%arg16 = %expand_shape_670 : memref<1x48xf32, strided<[48, 1], offset: 240>>, %arg17 = %expand_shape_667 : memref<1x1024xf32>, %arg18 = %678 : f32, %arg19 = %subview_668 : memref<1024x48xf32, strided<[768, 1], offset: 3932400>>) -> memref<1x48xf32, strided<[48, 1], offset: 240>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932400>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 240>>
    }
    %collapse_shape_671 = memref.collapse_shape %679 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 240>> into memref<48xf32, strided<[1], offset: 240>>
    memref.copy %collapse_shape_671, %subview_669 : memref<48xf32, strided<[1], offset: 240>> to memref<48xf32, strided<[1], offset: 240>>
    %subview_672 = memref.subview %637#0[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %subview_673 = memref.subview %subview_629[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932448>>
    %680 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_673 : memref<1024x48xf32, strided<[768, 1], offset: 3932448>>, %arg18 = %subview_672 : memref<48xf32, strided<[1], offset: 288>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932448>>, memref<48xf32, strided<[1], offset: 288>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %681 = cinm.compute_block (%arg16 = %680 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %682 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %681 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %683 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %682 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %684 = cinm.compute_block (%arg16 = %682 : memref<1024xf32>, %arg17 = %683 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %685 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %684 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_674 = memref.expand_shape %684 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_675 = memref.subview %subview_630[0, 288] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932448>>
    %subview_676 = memref.subview %633[288] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 288>>
    %expand_shape_677 = memref.expand_shape %subview_676 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 288>> into memref<1x48xf32, strided<[48, 1], offset: 288>>
    %686 = cinm.compute_block (%arg16 = %expand_shape_677 : memref<1x48xf32, strided<[48, 1], offset: 288>>, %arg17 = %expand_shape_674 : memref<1x1024xf32>, %arg18 = %685 : f32, %arg19 = %subview_675 : memref<1024x48xf32, strided<[768, 1], offset: 3932448>>) -> memref<1x48xf32, strided<[48, 1], offset: 288>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932448>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 288>>
    }
    %collapse_shape_678 = memref.collapse_shape %686 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 288>> into memref<48xf32, strided<[1], offset: 288>>
    memref.copy %collapse_shape_678, %subview_676 : memref<48xf32, strided<[1], offset: 288>> to memref<48xf32, strided<[1], offset: 288>>
    %subview_679 = memref.subview %637#0[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %subview_680 = memref.subview %subview_629[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932496>>
    %687 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_680 : memref<1024x48xf32, strided<[768, 1], offset: 3932496>>, %arg18 = %subview_679 : memref<48xf32, strided<[1], offset: 336>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932496>>, memref<48xf32, strided<[1], offset: 336>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %688 = cinm.compute_block (%arg16 = %687 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %689 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %688 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %690 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %689 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %691 = cinm.compute_block (%arg16 = %689 : memref<1024xf32>, %arg17 = %690 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %692 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %691 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_681 = memref.expand_shape %691 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_682 = memref.subview %subview_630[0, 336] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932496>>
    %subview_683 = memref.subview %633[336] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 336>>
    %expand_shape_684 = memref.expand_shape %subview_683 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 336>> into memref<1x48xf32, strided<[48, 1], offset: 336>>
    %693 = cinm.compute_block (%arg16 = %expand_shape_684 : memref<1x48xf32, strided<[48, 1], offset: 336>>, %arg17 = %expand_shape_681 : memref<1x1024xf32>, %arg18 = %692 : f32, %arg19 = %subview_682 : memref<1024x48xf32, strided<[768, 1], offset: 3932496>>) -> memref<1x48xf32, strided<[48, 1], offset: 336>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932496>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 336>>
    }
    %collapse_shape_685 = memref.collapse_shape %693 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 336>> into memref<48xf32, strided<[1], offset: 336>>
    memref.copy %collapse_shape_685, %subview_683 : memref<48xf32, strided<[1], offset: 336>> to memref<48xf32, strided<[1], offset: 336>>
    %subview_686 = memref.subview %637#0[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %subview_687 = memref.subview %subview_629[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932544>>
    %694 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_687 : memref<1024x48xf32, strided<[768, 1], offset: 3932544>>, %arg18 = %subview_686 : memref<48xf32, strided<[1], offset: 384>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932544>>, memref<48xf32, strided<[1], offset: 384>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %695 = cinm.compute_block (%arg16 = %694 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %696 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %695 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %697 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %696 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %698 = cinm.compute_block (%arg16 = %696 : memref<1024xf32>, %arg17 = %697 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %699 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %698 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_688 = memref.expand_shape %698 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_689 = memref.subview %subview_630[0, 384] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932544>>
    %subview_690 = memref.subview %633[384] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 384>>
    %expand_shape_691 = memref.expand_shape %subview_690 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 384>> into memref<1x48xf32, strided<[48, 1], offset: 384>>
    %700 = cinm.compute_block (%arg16 = %expand_shape_691 : memref<1x48xf32, strided<[48, 1], offset: 384>>, %arg17 = %expand_shape_688 : memref<1x1024xf32>, %arg18 = %699 : f32, %arg19 = %subview_689 : memref<1024x48xf32, strided<[768, 1], offset: 3932544>>) -> memref<1x48xf32, strided<[48, 1], offset: 384>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932544>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 384>>
    }
    %collapse_shape_692 = memref.collapse_shape %700 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 384>> into memref<48xf32, strided<[1], offset: 384>>
    memref.copy %collapse_shape_692, %subview_690 : memref<48xf32, strided<[1], offset: 384>> to memref<48xf32, strided<[1], offset: 384>>
    %subview_693 = memref.subview %637#0[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %subview_694 = memref.subview %subview_629[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932592>>
    %701 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_694 : memref<1024x48xf32, strided<[768, 1], offset: 3932592>>, %arg18 = %subview_693 : memref<48xf32, strided<[1], offset: 432>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932592>>, memref<48xf32, strided<[1], offset: 432>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %702 = cinm.compute_block (%arg16 = %701 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %703 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %702 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %704 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %703 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %705 = cinm.compute_block (%arg16 = %703 : memref<1024xf32>, %arg17 = %704 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %706 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %705 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_695 = memref.expand_shape %705 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_696 = memref.subview %subview_630[0, 432] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932592>>
    %subview_697 = memref.subview %633[432] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 432>>
    %expand_shape_698 = memref.expand_shape %subview_697 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 432>> into memref<1x48xf32, strided<[48, 1], offset: 432>>
    %707 = cinm.compute_block (%arg16 = %expand_shape_698 : memref<1x48xf32, strided<[48, 1], offset: 432>>, %arg17 = %expand_shape_695 : memref<1x1024xf32>, %arg18 = %706 : f32, %arg19 = %subview_696 : memref<1024x48xf32, strided<[768, 1], offset: 3932592>>) -> memref<1x48xf32, strided<[48, 1], offset: 432>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932592>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 432>>
    }
    %collapse_shape_699 = memref.collapse_shape %707 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 432>> into memref<48xf32, strided<[1], offset: 432>>
    memref.copy %collapse_shape_699, %subview_697 : memref<48xf32, strided<[1], offset: 432>> to memref<48xf32, strided<[1], offset: 432>>
    %subview_700 = memref.subview %637#0[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %subview_701 = memref.subview %subview_629[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932640>>
    %708 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_701 : memref<1024x48xf32, strided<[768, 1], offset: 3932640>>, %arg18 = %subview_700 : memref<48xf32, strided<[1], offset: 480>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932640>>, memref<48xf32, strided<[1], offset: 480>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %709 = cinm.compute_block (%arg16 = %708 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %710 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %709 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %711 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %710 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %712 = cinm.compute_block (%arg16 = %710 : memref<1024xf32>, %arg17 = %711 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %713 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %712 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_702 = memref.expand_shape %712 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_703 = memref.subview %subview_630[0, 480] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932640>>
    %subview_704 = memref.subview %633[480] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 480>>
    %expand_shape_705 = memref.expand_shape %subview_704 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 480>> into memref<1x48xf32, strided<[48, 1], offset: 480>>
    %714 = cinm.compute_block (%arg16 = %expand_shape_705 : memref<1x48xf32, strided<[48, 1], offset: 480>>, %arg17 = %expand_shape_702 : memref<1x1024xf32>, %arg18 = %713 : f32, %arg19 = %subview_703 : memref<1024x48xf32, strided<[768, 1], offset: 3932640>>) -> memref<1x48xf32, strided<[48, 1], offset: 480>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932640>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 480>>
    }
    %collapse_shape_706 = memref.collapse_shape %714 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 480>> into memref<48xf32, strided<[1], offset: 480>>
    memref.copy %collapse_shape_706, %subview_704 : memref<48xf32, strided<[1], offset: 480>> to memref<48xf32, strided<[1], offset: 480>>
    %subview_707 = memref.subview %637#0[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %subview_708 = memref.subview %subview_629[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932688>>
    %715 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_708 : memref<1024x48xf32, strided<[768, 1], offset: 3932688>>, %arg18 = %subview_707 : memref<48xf32, strided<[1], offset: 528>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932688>>, memref<48xf32, strided<[1], offset: 528>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %716 = cinm.compute_block (%arg16 = %715 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %717 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %716 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %718 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %717 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %719 = cinm.compute_block (%arg16 = %717 : memref<1024xf32>, %arg17 = %718 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %720 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %719 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_709 = memref.expand_shape %719 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_710 = memref.subview %subview_630[0, 528] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932688>>
    %subview_711 = memref.subview %633[528] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 528>>
    %expand_shape_712 = memref.expand_shape %subview_711 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 528>> into memref<1x48xf32, strided<[48, 1], offset: 528>>
    %721 = cinm.compute_block (%arg16 = %expand_shape_712 : memref<1x48xf32, strided<[48, 1], offset: 528>>, %arg17 = %expand_shape_709 : memref<1x1024xf32>, %arg18 = %720 : f32, %arg19 = %subview_710 : memref<1024x48xf32, strided<[768, 1], offset: 3932688>>) -> memref<1x48xf32, strided<[48, 1], offset: 528>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932688>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 528>>
    }
    %collapse_shape_713 = memref.collapse_shape %721 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 528>> into memref<48xf32, strided<[1], offset: 528>>
    memref.copy %collapse_shape_713, %subview_711 : memref<48xf32, strided<[1], offset: 528>> to memref<48xf32, strided<[1], offset: 528>>
    %subview_714 = memref.subview %637#0[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %subview_715 = memref.subview %subview_629[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932736>>
    %722 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_715 : memref<1024x48xf32, strided<[768, 1], offset: 3932736>>, %arg18 = %subview_714 : memref<48xf32, strided<[1], offset: 576>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932736>>, memref<48xf32, strided<[1], offset: 576>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %723 = cinm.compute_block (%arg16 = %722 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %724 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %723 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %725 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %724 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %726 = cinm.compute_block (%arg16 = %724 : memref<1024xf32>, %arg17 = %725 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %727 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %726 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_716 = memref.expand_shape %726 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_717 = memref.subview %subview_630[0, 576] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932736>>
    %subview_718 = memref.subview %633[576] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 576>>
    %expand_shape_719 = memref.expand_shape %subview_718 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 576>> into memref<1x48xf32, strided<[48, 1], offset: 576>>
    %728 = cinm.compute_block (%arg16 = %expand_shape_719 : memref<1x48xf32, strided<[48, 1], offset: 576>>, %arg17 = %expand_shape_716 : memref<1x1024xf32>, %arg18 = %727 : f32, %arg19 = %subview_717 : memref<1024x48xf32, strided<[768, 1], offset: 3932736>>) -> memref<1x48xf32, strided<[48, 1], offset: 576>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932736>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 576>>
    }
    %collapse_shape_720 = memref.collapse_shape %728 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 576>> into memref<48xf32, strided<[1], offset: 576>>
    memref.copy %collapse_shape_720, %subview_718 : memref<48xf32, strided<[1], offset: 576>> to memref<48xf32, strided<[1], offset: 576>>
    %subview_721 = memref.subview %637#0[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %subview_722 = memref.subview %subview_629[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932784>>
    %729 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_722 : memref<1024x48xf32, strided<[768, 1], offset: 3932784>>, %arg18 = %subview_721 : memref<48xf32, strided<[1], offset: 624>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932784>>, memref<48xf32, strided<[1], offset: 624>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %730 = cinm.compute_block (%arg16 = %729 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %731 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %730 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %732 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %731 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %733 = cinm.compute_block (%arg16 = %731 : memref<1024xf32>, %arg17 = %732 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %734 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %733 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_723 = memref.expand_shape %733 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_724 = memref.subview %subview_630[0, 624] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932784>>
    %subview_725 = memref.subview %633[624] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 624>>
    %expand_shape_726 = memref.expand_shape %subview_725 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 624>> into memref<1x48xf32, strided<[48, 1], offset: 624>>
    %735 = cinm.compute_block (%arg16 = %expand_shape_726 : memref<1x48xf32, strided<[48, 1], offset: 624>>, %arg17 = %expand_shape_723 : memref<1x1024xf32>, %arg18 = %734 : f32, %arg19 = %subview_724 : memref<1024x48xf32, strided<[768, 1], offset: 3932784>>) -> memref<1x48xf32, strided<[48, 1], offset: 624>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932784>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 624>>
    }
    %collapse_shape_727 = memref.collapse_shape %735 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 624>> into memref<48xf32, strided<[1], offset: 624>>
    memref.copy %collapse_shape_727, %subview_725 : memref<48xf32, strided<[1], offset: 624>> to memref<48xf32, strided<[1], offset: 624>>
    %subview_728 = memref.subview %637#0[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %subview_729 = memref.subview %subview_629[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932832>>
    %736 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_729 : memref<1024x48xf32, strided<[768, 1], offset: 3932832>>, %arg18 = %subview_728 : memref<48xf32, strided<[1], offset: 672>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932832>>, memref<48xf32, strided<[1], offset: 672>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %737 = cinm.compute_block (%arg16 = %736 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %738 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %737 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %739 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %738 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %740 = cinm.compute_block (%arg16 = %738 : memref<1024xf32>, %arg17 = %739 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %741 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %740 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_730 = memref.expand_shape %740 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_731 = memref.subview %subview_630[0, 672] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932832>>
    %subview_732 = memref.subview %633[672] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 672>>
    %expand_shape_733 = memref.expand_shape %subview_732 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 672>> into memref<1x48xf32, strided<[48, 1], offset: 672>>
    %742 = cinm.compute_block (%arg16 = %expand_shape_733 : memref<1x48xf32, strided<[48, 1], offset: 672>>, %arg17 = %expand_shape_730 : memref<1x1024xf32>, %arg18 = %741 : f32, %arg19 = %subview_731 : memref<1024x48xf32, strided<[768, 1], offset: 3932832>>) -> memref<1x48xf32, strided<[48, 1], offset: 672>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932832>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 672>>
    }
    %collapse_shape_734 = memref.collapse_shape %742 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 672>> into memref<48xf32, strided<[1], offset: 672>>
    memref.copy %collapse_shape_734, %subview_732 : memref<48xf32, strided<[1], offset: 672>> to memref<48xf32, strided<[1], offset: 672>>
    %subview_735 = memref.subview %637#0[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %subview_736 = memref.subview %subview_629[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932880>>
    %743 = cinm.compute_block (%arg16 = %alloc_11 : memref<1024xf32>, %arg17 = %subview_736 : memref<1024x48xf32, strided<[768, 1], offset: 3932880>>, %arg18 = %subview_735 : memref<48xf32, strided<[1], offset: 720>>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<1024x48xf32, strided<[768, 1], offset: 3932880>>, memref<48xf32, strided<[1], offset: 720>>) outs(%arg16 : memref<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %744 = cinm.compute_block (%arg16 = %743 : memref<1024xf32>, %arg17 = %alloc_11 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_1 : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %760 : memref<1024xf32>, memref<f32>) outs(%arg17 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %745 = cinm.compute_block (%arg16 = %7 : index, %arg17 = %744 : memref<1024xf32>) -> memref<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 0xFF800000 : f32
      %c1024 = arith.constant 1024 : index
      %c1 = arith.constant 1 : index
      scf.for %arg18 = %arg16 to %c1024 step %c1 {
        memref.store %cst, %arg17[%arg18] : memref<1024xf32>
      }
      cinm.yield %arg17 : memref<1024xf32>
    }
    %746 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %745 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0xFFC00000 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.maxnumf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %747 = cinm.compute_block (%arg16 = %745 : memref<1024xf32>, %arg17 = %746 : f32) -> memref<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17 : memref<1024xf32>, f32) outs(%arg16 : memref<1024xf32>) {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.subf %in, %in_751 : f32
        %761 = math.exp %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg16 : memref<1024xf32>
    }
    %748 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %747 : memref<1024xf32>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<1024xf32>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.addf %in, %out : f32
        linalg.yield %761 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %expand_shape_737 = memref.expand_shape %747 [[0, 1]] output_shape [1, 1024] : memref<1024xf32> into memref<1x1024xf32>
    %subview_738 = memref.subview %subview_630[0, 720] [1024, 48] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: 3932160>> to memref<1024x48xf32, strided<[768, 1], offset: 3932880>>
    %subview_739 = memref.subview %633[720] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: 720>>
    %expand_shape_740 = memref.expand_shape %subview_739 [[0, 1]] output_shape [1, 48] : memref<48xf32, strided<[1], offset: 720>> into memref<1x48xf32, strided<[48, 1], offset: 720>>
    %749 = cinm.compute_block (%arg16 = %expand_shape_740 : memref<1x48xf32, strided<[48, 1], offset: 720>>, %arg17 = %expand_shape_737 : memref<1x1024xf32>, %arg18 = %748 : f32, %arg19 = %subview_738 : memref<1024x48xf32, strided<[768, 1], offset: 3932880>>) -> memref<1x48xf32, strided<[48, 1], offset: 720>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg17, %arg18, %arg19 : memref<1x1024xf32>, f32, memref<1024x48xf32, strided<[768, 1], offset: 3932880>>) outs(%arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %761 = arith.divf %in, %in_751 : f32
        %762 = arith.mulf %761, %in_752 : f32
        %763 = arith.addf %out, %762 : f32
        linalg.yield %763 : f32
      }
      cinm.yield %arg16 : memref<1x48xf32, strided<[48, 1], offset: 720>>
    }
    %collapse_shape_741 = memref.collapse_shape %749 [[0, 1]] : memref<1x48xf32, strided<[48, 1], offset: 720>> into memref<48xf32, strided<[1], offset: 720>>
    memref.copy %collapse_shape_741, %subview_739 : memref<48xf32, strided<[1], offset: 720>> to memref<48xf32, strided<[1], offset: 720>>
    %subview_742 = memref.subview %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : memref<6x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: 2949120>>
    %750 = cinm.compute_block (%arg16 = %subview_742 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, %arg17 = %633 : memref<768xf32>, %arg18 = %630#1 : memref<768xf32, strided<[1], offset: ?>>) -> memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17 : memref<768x768xf32, strided<[768, 1], offset: 2949120>>, memref<768xf32>) outs(%arg18 : memref<768xf32, strided<[1], offset: ?>>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.addf %out, %760 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg18 : memref<768xf32, strided<[1], offset: ?>>
    }
    %subview_743 = memref.subview %arg13[5, 0] [1, 768] [1, 1] : memref<6x768xf32> to memref<768xf32, strided<[1], offset: 3840>>
    %751 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %750 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %752 = cinm.compute_block (%arg16 = %751 : f32) -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 7.680000e+02 : f32
      %cst_751 = arith.constant 9.99999974E-6 : f32
      %760 = arith.divf %arg16, %cst : f32
      %761 = arith.addf %760, %cst_751 : f32
      %762 = math.rsqrt %761 : f32
      cinm.yield %762 : f32
    }
    %753 = cinm.compute_block (%arg16 = %750 : memref<768xf32, strided<[1], offset: ?>>, %arg17 = %752 : f32, %arg18 = %subview_743 : memref<768xf32, strided<[1], offset: 3840>>, %arg19 = %633 : memref<768xf32>) -> memref<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      linalg.generic {indexing_maps = [#map, #map1, #map, #map], iterator_types = ["parallel"]} ins(%arg16, %arg17, %arg18 : memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32, strided<[1], offset: 3840>>) outs(%arg19 : memref<768xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32):
        %760 = arith.mulf %in, %in_751 : f32
        %761 = arith.mulf %760, %in_752 : f32
        linalg.yield %761 : f32
      }
      cinm.yield %arg19 : memref<768xf32>
    }
    memref.copy %753, %633 : memref<768xf32> to memref<768xf32>
    %subview_744 = memref.subview %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
    %subview_745 = memref.subview %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<6x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: 7864320>>
    %754 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_744 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, %arg18 = %633 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %755 = cinm.compute_block (%arg16 = %alloc_125 : memref<2048xf32>, %arg17 = %subview_745 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, %arg18 = %633 : memref<768xf32>) -> memref<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18 : memref<2048x768xf32, strided<[768, 1], offset: 7864320>>, memref<768xf32>) outs(%arg16 : memref<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map2, #map3, #map4]} {
      ^bb0(%in: f32, %in_751: f32, %out: f32):
        %761 = arith.mulf %in, %in_751 : f32
        %762 = arith.addf %out, %761 : f32
        linalg.yield %762 : f32
      }
      cinm.yield %arg16 : memref<2048xf32>
    }
    %subview_746 = memref.subview %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<6x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>
    %756:2 = cinm.compute_block (%arg16 = %subview_746 : memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>, %arg17 = %755 : memref<2048xf32>, %arg18 = %754 : memref<2048xf32>, %arg19 = %750 : memref<768xf32, strided<[1], offset: ?>>) -> memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32_2 : memref<f32>
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg16, %arg17, %760 : memref<768x2048xf32, strided<[2048, 1], offset: 7864320>>, memref<2048xf32>, memref<f32>) outs(%arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %out: f32, %out_753: f32):
        %761 = arith.negf %out : f32
        %762 = math.exp %761 : f32
        %763 = arith.addf %762, %in_752 : f32
        %764 = arith.divf %in_752, %763 : f32
        %765 = arith.mulf %out, %764 : f32
        %766 = arith.mulf %765, %in_751 : f32
        %767 = arith.mulf %in, %766 : f32
        %768 = arith.addf %out_753, %767 : f32
        linalg.yield %766, %768 : f32, f32
      }
      cinm.yield %arg18, %arg19 : memref<2048xf32>, memref<768xf32, strided<[1], offset: ?>>
    }
    %757 = cinm.compute_block (%arg16 = %alloc : memref<f32>, %arg17 = %756#1 : memref<768xf32, strided<[1], offset: ?>>) -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_751 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst, %alloc_751[] : memref<f32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg17 : memref<768xf32, strided<[1], offset: ?>>) outs(%alloc_751 : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %761 = arith.mulf %in, %in : f32
        %762 = arith.addf %761, %out : f32
        linalg.yield %762 : f32
      }
      %760 = memref.load %alloc_751[] : memref<f32>
      cinm.yield %760 : f32
    }
    %alloc_747 = memref.alloc() {alignment = 64 : i64} : memref<34048x768xf32>
    %758:2 = cinm.compute_block (%arg16 = %757 : f32, %arg17 = %alloc_747 : memref<34048x768xf32>) -> f32, memref<34048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %cst = arith.constant 9.99999974E-6 : f32
      %cst_751 = arith.constant 7.680000e+02 : f32
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      %761 = arith.divf %arg16, %cst_751 : f32
      %762 = arith.addf %761, %cst : f32
      %763 = math.rsqrt %762 : f32
      linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%760 : memref<f32>) outs(%arg17 : memref<34048x768xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      cinm.yield %763, %arg17 : f32, memref<34048x768xf32>
    }
    %subview_748 = memref.subview %758#1[0, 0] [32000, 768] [1, 1] : memref<34048x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
    memref.copy %arg15, %subview_748 : memref<32000x768xf32> to memref<32000x768xf32, strided<[768, 1]>>
    %alloc_749 = memref.alloc() {alignment = 64 : i64} : memref<34048xf32>
    %759 = cinm.compute_block (%arg16 = %alloc_749 : memref<34048xf32>, %arg17 = %758#1 : memref<34048x768xf32>, %arg18 = %756#1 : memref<768xf32, strided<[1], offset: ?>>, %arg19 = %758#0 : f32, %arg20 = %arg14 : memref<768xf32>) -> memref<34048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %760 = memref.get_global @__constant_xf32 : memref<f32>
      linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%760 : memref<f32>) outs(%arg16 : memref<34048xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      linalg.generic {indexing_maps = [#map2, #map3, #map5, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg17, %arg18, %arg19, %arg20 : memref<34048x768xf32>, memref<768xf32, strided<[1], offset: ?>>, f32, memref<768xf32>) outs(%arg16 : memref<34048xf32>) {
      ^bb0(%in: f32, %in_751: f32, %in_752: f32, %in_753: f32, %out: f32):
        %761 = arith.mulf %in_751, %in_752 : f32
        %762 = arith.mulf %761, %in_753 : f32
        %763 = arith.mulf %in, %762 : f32
        %764 = arith.addf %out, %763 : f32
        linalg.yield %764 : f32
      }
      cinm.yield %arg16 : memref<34048xf32>
    }
    %subview_750 = memref.subview %759[0] [32000] [1] : memref<34048xf32> to memref<32000xf32, strided<[1]>>
    %cast = memref.cast %subview_750 : memref<32000xf32, strided<[1]>> to memref<32000xf32>
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
