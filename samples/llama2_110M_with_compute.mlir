#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> ()>
#map8 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> ()>
#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = #map3, cinm.static}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = #map3, cinm.static}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, cinm.static}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = #map3, cinm.static}, %arg14: tensor<768xf32> {bufferization.buffer_layout = #map1, cinm.static}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = #map3, cinm.static}) -> tensor<32000xf32> attributes {cinm.available_platforms = [#upmem]} {
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant dense<[1, 48]> : tensor<2xindex>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 0xFF800000 : f32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %cst_4 = arith.constant 7.680000e+02 : f32
    %cst_5 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 4.800000e+01 : f32
    %cst_8 = arith.constant 1.000000e+04 : f32
    %0:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg5[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<768xf32>
    }
    %1 = tensor.empty() : tensor<f32>
    %2 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%0#0 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %3 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %2[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %4 = tensor.empty() : tensor<768xf32>
    %5 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%0#0, %3, %0#1 : tensor<768xf32>, f32, tensor<768xf32>) outs(%4 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %6:3 = cinm.compute on platform #cinm.host_platform -> tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9, %extracted_slice_10 : tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>
    }
    %7 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%6#0, %5 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %8 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %9 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%6#1, %5 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %10 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %11 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%6#2, %5 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %12:9 = cinm.compute on platform #cinm.host_platform -> tensor<6x1024x768xf32>, f32, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %inserted_slice = tensor.insert_slice %11 into %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %1096 = arith.index_cast %arg1 : index to i64
      %1097 = arith.uitofp %1096 : i64 to f32
      %1098:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %7, %arg18 = %9) -> (tensor<768xf32>, tensor<768xf32>) {
        %1100 = arith.remui %arg16, %c48 : index
        %1101 = arith.index_cast %1100 : index to i64
        %1102 = arith.uitofp %1101 : i64 to f32
        %1103 = arith.divf %1102, %cst_7 : f32
        %1104 = math.powf %cst_8, %1103 : f32
        %1105 = arith.divf %cst_6, %1104 : f32
        %1106 = arith.mulf %1097, %1105 : f32
        %1107 = math.cos %1106 : f32
        %1108 = math.sin %1106 : f32
        %1109 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_13 = tensor.extract %arg17[%1109] : tensor<768xf32>
        %1110 = arith.mulf %extracted, %1107 : f32
        %1111 = arith.mulf %extracted_13, %1108 : f32
        %1112 = arith.subf %1110, %1111 : f32
        %inserted = tensor.insert %1112 into %arg17[%arg16] : tensor<768xf32>
        %1113 = arith.mulf %extracted, %1108 : f32
        %1114 = arith.mulf %extracted_13, %1107 : f32
        %1115 = arith.addf %1113, %1114 : f32
        %inserted_14 = tensor.insert %1115 into %inserted[%1109] : tensor<768xf32>
        %1116 = bufferization.materialize_in_destination %inserted_14 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %1117 = arith.cmpi ult, %arg16, %c768 : index
        %1118 = scf.if %1117 -> (tensor<768xf32>) {
          %extracted_15 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_16 = tensor.extract %arg18[%1109] : tensor<768xf32>
          %1119 = arith.mulf %extracted_15, %1107 : f32
          %1120 = arith.mulf %extracted_16, %1108 : f32
          %1121 = arith.subf %1119, %1120 : f32
          %inserted_17 = tensor.insert %1121 into %arg18[%arg16] : tensor<768xf32>
          %1122 = arith.mulf %extracted_15, %1108 : f32
          %1123 = arith.mulf %extracted_16, %1107 : f32
          %1124 = arith.addf %1122, %1123 : f32
          %inserted_18 = tensor.insert %1124 into %inserted_17[%1109] : tensor<768xf32>
          %1125 = bufferization.materialize_in_destination %inserted_18 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %1125 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %1116, %1118 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %1098#1 into %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice_9[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %1099 = arith.addi %arg1, %c1 : index
      %extracted_slice_11 = tensor.extract_slice %1098#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %1097, %1098#0, %inserted_slice_9, %extracted_slice, %extracted_slice_10, %1099, %extracted_slice_11, %extracted_slice_12 : tensor<6x1024x768xf32>, f32, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index, tensor<48xf32>, tensor<1024x48xf32>
    }
    %13 = tensor.empty() : tensor<1024xf32>
    %14 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%12#8, %12#7 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %15 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%14 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %16 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %15) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %17 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%16 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %18 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %17[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %19 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%16, %18 : tensor<1024xf32>, f32) outs(%16 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %20 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%19 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %21:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %20[] : tensor<f32>
      %expanded = tensor.expand_shape %19 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %4[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %22 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%21#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21#1, %21#0, %21#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %23:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %22 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %5[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %24 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%23#2, %23#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %25 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%24 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %26 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %25) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %27 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%26 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %28 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %27[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %29 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%26, %28 : tensor<1024xf32>, f32) outs(%26 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %30 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%29 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %31:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %30[] : tensor<f32>
      %expanded = tensor.expand_shape %29 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %23#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %32 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%31#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%31#1, %31#0, %31#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %33:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %32 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %23#0[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %34 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%33#2, %33#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %35 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%34 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %36 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %35) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %37 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%36 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %38 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %37[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %39 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%36, %38 : tensor<1024xf32>, f32) outs(%36 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %40 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%39 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %41:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %40[] : tensor<f32>
      %expanded = tensor.expand_shape %39 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %33#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %42 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%41#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41#1, %41#0, %41#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %43:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %42 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %33#0[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %44 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%43#2, %43#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %45 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%44 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %46 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %45) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %47 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%46 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %48 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %47[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %49 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%46, %48 : tensor<1024xf32>, f32) outs(%46 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %50 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%49 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %51:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %50[] : tensor<f32>
      %expanded = tensor.expand_shape %49 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %43#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %52 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%51#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51#1, %51#0, %51#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %53:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %52 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %43#0[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %54 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%53#2, %53#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %55 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%54 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %56 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %55) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %57 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%56 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %58 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %57[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %59 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%56, %58 : tensor<1024xf32>, f32) outs(%56 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %60 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%59 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %61:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %60[] : tensor<f32>
      %expanded = tensor.expand_shape %59 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %53#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %62 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%61#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61#1, %61#0, %61#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %63:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %62 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %53#0[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %64 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%63#2, %63#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %65 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%64 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %66 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %65) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %67 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%66 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %68 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %67[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %69 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%66, %68 : tensor<1024xf32>, f32) outs(%66 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %70 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%69 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %71:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %70[] : tensor<f32>
      %expanded = tensor.expand_shape %69 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %63#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %72 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%71#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%71#1, %71#0, %71#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %73:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %72 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %63#0[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %74 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%73#2, %73#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %75 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%74 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %76 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %75) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %77 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%76 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %78 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %77[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %79 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%76, %78 : tensor<1024xf32>, f32) outs(%76 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %80 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%79 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %81:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %80[] : tensor<f32>
      %expanded = tensor.expand_shape %79 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %73#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %82 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%81#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81#1, %81#0, %81#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %83:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %82 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %73#0[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %84 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%83#2, %83#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %85 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%84 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %86 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %85) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %87 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%86 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %88 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %87[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %89 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%86, %88 : tensor<1024xf32>, f32) outs(%86 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %90 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%89 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %91:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %90[] : tensor<f32>
      %expanded = tensor.expand_shape %89 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %83#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %92 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%91#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%91#1, %91#0, %91#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %93:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %92 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %83#0[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %94 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%93#2, %93#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %95 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%94 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %96 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %95) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %97 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%96 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %98 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %97[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %99 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%96, %98 : tensor<1024xf32>, f32) outs(%96 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %100 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%99 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %101:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %100[] : tensor<f32>
      %expanded = tensor.expand_shape %99 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %93#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %102 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%101#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%101#1, %101#0, %101#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %103:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %102 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %93#0[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %104 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%103#2, %103#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %105 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%104 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %106 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %105) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %107 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%106 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %108 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %107[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %109 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%106, %108 : tensor<1024xf32>, f32) outs(%106 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %110 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%109 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %111:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %110[] : tensor<f32>
      %expanded = tensor.expand_shape %109 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %103#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %112 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%111#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111#1, %111#0, %111#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %113:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %112 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %103#0[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %114 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%113#2, %113#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %115 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%114 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %116 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %115) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %117 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%116 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %118 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %117[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %119 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%116, %118 : tensor<1024xf32>, f32) outs(%116 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %120 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%119 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %121:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %120[] : tensor<f32>
      %expanded = tensor.expand_shape %119 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %113#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %122 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%121#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121#1, %121#0, %121#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %123:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %122 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %113#0[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %124 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%123#2, %123#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %125 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%124 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %126 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %125) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %127 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%126 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %128 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %127[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %129 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%126, %128 : tensor<1024xf32>, f32) outs(%126 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %130 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%129 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %131:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %130[] : tensor<f32>
      %expanded = tensor.expand_shape %129 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %123#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %132 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%131#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%131#1, %131#0, %131#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %133:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %132 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %123#0[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %134 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%133#2, %133#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %135 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%134 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %136 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %135) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %137 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%136 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %138 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %137[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %139 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%136, %138 : tensor<1024xf32>, f32) outs(%136 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %140 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%139 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %141:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %140[] : tensor<f32>
      %expanded = tensor.expand_shape %139 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %133#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %142 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%141#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%141#1, %141#0, %141#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %143:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %142 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %133#0[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %144 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%143#2, %143#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %145 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%144 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %146 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %145) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %147 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%146 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %148 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %147[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %149 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%146, %148 : tensor<1024xf32>, f32) outs(%146 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %150 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%149 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %151:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %150[] : tensor<f32>
      %expanded = tensor.expand_shape %149 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %143#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %152 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%151#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%151#1, %151#0, %151#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %153:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %152 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %143#0[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %154 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%153#2, %153#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %155 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%154 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %156 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %155) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %157 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%156 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %158 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %157[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %159 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%156, %158 : tensor<1024xf32>, f32) outs(%156 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %160 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%159 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %161:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %160[] : tensor<f32>
      %expanded = tensor.expand_shape %159 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %153#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %162 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%161#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%161#1, %161#0, %161#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %163:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %162 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %153#0[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %12#2[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %12#4[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %164 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%163#2, %163#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %165 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%164 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %166 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %165) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %167 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%166 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %168 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %167[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %169 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%166, %168 : tensor<1024xf32>, f32) outs(%166 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %170 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%169 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %171:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %170[] : tensor<f32>
      %expanded = tensor.expand_shape %169 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %12#5[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %163#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %172 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%171#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%171#1, %171#0, %171#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %173:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %172 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %163#0[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %1096 = bufferization.materialize_in_destination %inserted_slice in %5 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %1096, %extracted_slice : tensor<768xf32>, tensor<768x768xf32>
    }
    %174 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%173#1, %173#0 : tensor<768x768xf32>, tensor<768xf32>) outs(%0#0 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.addf %out, %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %175 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg13[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %176 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%174 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %177 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %176[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %178 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%174, %177, %175 : tensor<768xf32>, f32, tensor<768xf32>) outs(%173#0 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %179:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = bufferization.materialize_in_destination %178 in %173#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      cinm.yield %1096, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32>
    }
    %180 = tensor.empty() : tensor<2048xf32>
    %181 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%179#1, %179#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %182 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%179#2, %179#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %183 = cinm.compute on platform #cinm.host_platform -> tensor<768x2048xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      cinm.yield %extracted_slice : tensor<768x2048xf32>
    }
    %184:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%183, %182 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%181, %174 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %1097 = arith.negf %out : f32
        %1098 = math.exp %1097 : f32
        %1099 = arith.addf %1098, %cst_6 : f32
        %1100 = arith.divf %cst_6, %1099 : f32
        %1101 = arith.mulf %out, %1100 : f32
        %1102 = arith.mulf %1101, %in_9 : f32
        %1103 = arith.mulf %in, %1102 : f32
        %1104 = arith.addf %out_10, %1103 : f32
        linalg.yield %1102, %1104 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %1096#0, %1096#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %185 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg5[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %186 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%184#1 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %187 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %186[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %188 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%184#1, %187, %185 : tensor<768xf32>, f32, tensor<768xf32>) outs(%4 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %189:3 = cinm.compute on platform #cinm.host_platform -> tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9, %extracted_slice_10 : tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>
    }
    %190 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%189#0, %188 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %191 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %12#3[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %192 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%191 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%189#1, %188 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %193 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %12#0[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %194 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%193 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%189#2, %188 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %195:7 = cinm.compute on platform #cinm.host_platform -> tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %inserted_slice = tensor.insert_slice %194 into %12#0[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %1096:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %190, %arg18 = %192) -> (tensor<768xf32>, tensor<768xf32>) {
        %1097 = arith.remui %arg16, %c48 : index
        %1098 = arith.index_cast %1097 : index to i64
        %1099 = arith.uitofp %1098 : i64 to f32
        %1100 = arith.divf %1099, %cst_7 : f32
        %1101 = math.powf %cst_8, %1100 : f32
        %1102 = arith.divf %cst_6, %1101 : f32
        %1103 = arith.mulf %12#1, %1102 : f32
        %1104 = math.cos %1103 : f32
        %1105 = math.sin %1103 : f32
        %1106 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_13 = tensor.extract %arg17[%1106] : tensor<768xf32>
        %1107 = arith.mulf %extracted, %1104 : f32
        %1108 = arith.mulf %extracted_13, %1105 : f32
        %1109 = arith.subf %1107, %1108 : f32
        %inserted = tensor.insert %1109 into %arg17[%arg16] : tensor<768xf32>
        %1110 = arith.mulf %extracted, %1105 : f32
        %1111 = arith.mulf %extracted_13, %1104 : f32
        %1112 = arith.addf %1110, %1111 : f32
        %inserted_14 = tensor.insert %1112 into %inserted[%1106] : tensor<768xf32>
        %1113 = bufferization.materialize_in_destination %inserted_14 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %1114 = arith.cmpi ult, %arg16, %c768 : index
        %1115 = scf.if %1114 -> (tensor<768xf32>) {
          %extracted_15 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_16 = tensor.extract %arg18[%1106] : tensor<768xf32>
          %1116 = arith.mulf %extracted_15, %1104 : f32
          %1117 = arith.mulf %extracted_16, %1105 : f32
          %1118 = arith.subf %1116, %1117 : f32
          %inserted_17 = tensor.insert %1118 into %arg18[%arg16] : tensor<768xf32>
          %1119 = arith.mulf %extracted_15, %1105 : f32
          %1120 = arith.mulf %extracted_16, %1104 : f32
          %1121 = arith.addf %1119, %1120 : f32
          %inserted_18 = tensor.insert %1121 into %inserted_17[%1106] : tensor<768xf32>
          %1122 = bufferization.materialize_in_destination %inserted_18 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %1122 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %1113, %1115 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %1096#1 into %12#3[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice_9[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_11 = tensor.extract_slice %1096#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %1096#0, %inserted_slice_9, %extracted_slice, %extracted_slice_10, %extracted_slice_11, %extracted_slice_12 : tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %196 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%195#6, %195#5 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %197 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%196 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %198 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %197) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %199 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%198 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %200 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %199[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %201 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%198, %200 : tensor<1024xf32>, f32) outs(%198 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %202 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%201 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %203:3 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %202[] : tensor<f32>
      %expanded = tensor.expand_shape %201 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice : f32, tensor<1x1024xf32>, tensor<1024x48xf32>
    }
    %204 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%21#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%203#1, %203#0, %203#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %205:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %204 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %188[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %206 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%205#2, %205#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %207 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%206 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %208 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %207) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %209 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%208 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %210 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %209[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %211 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%208, %210 : tensor<1024xf32>, f32) outs(%208 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %212 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%211 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %213:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %212[] : tensor<f32>
      %expanded = tensor.expand_shape %211 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %205#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %214 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%213#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%213#1, %213#0, %213#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %215:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %214 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %205#0[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %216 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%215#2, %215#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %217 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%216 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %218 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %217) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %219 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%218 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %220 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %219[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %221 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%218, %220 : tensor<1024xf32>, f32) outs(%218 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %222 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%221 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %223:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %222[] : tensor<f32>
      %expanded = tensor.expand_shape %221 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %215#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %224 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%223#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%223#1, %223#0, %223#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %225:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %224 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %215#0[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %226 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%225#2, %225#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %227 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%226 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %228 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %227) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %229 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%228 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %230 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %229[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %231 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%228, %230 : tensor<1024xf32>, f32) outs(%228 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %232 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%231 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %233:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %232[] : tensor<f32>
      %expanded = tensor.expand_shape %231 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %225#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %234 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%233#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%233#1, %233#0, %233#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %235:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %234 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %225#0[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %236 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%235#2, %235#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %237 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%236 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %238 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %237) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %239 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%238 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %240 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %239[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %241 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%238, %240 : tensor<1024xf32>, f32) outs(%238 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %242 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%241 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %243:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %242[] : tensor<f32>
      %expanded = tensor.expand_shape %241 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %235#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %244 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%243#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%243#1, %243#0, %243#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %245:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %244 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %235#0[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %246 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%245#2, %245#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %247 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%246 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %248 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %247) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %249 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%248 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %250 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %249[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %251 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%248, %250 : tensor<1024xf32>, f32) outs(%248 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %252 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%251 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %253:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %252[] : tensor<f32>
      %expanded = tensor.expand_shape %251 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %245#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %254 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%253#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%253#1, %253#0, %253#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %255:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %254 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %245#0[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %256 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%255#2, %255#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %257 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%256 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %258 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %257) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %259 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%258 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %260 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %259[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %261 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%258, %260 : tensor<1024xf32>, f32) outs(%258 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %262 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%261 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %263:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %262[] : tensor<f32>
      %expanded = tensor.expand_shape %261 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %255#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %264 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%263#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%263#1, %263#0, %263#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %265:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %264 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %255#0[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %266 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%265#2, %265#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %267 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%266 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %268 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %267) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %269 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%268 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %270 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %269[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %271 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%268, %270 : tensor<1024xf32>, f32) outs(%268 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %272 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%271 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %273:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %272[] : tensor<f32>
      %expanded = tensor.expand_shape %271 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %265#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %274 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%273#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%273#1, %273#0, %273#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %275:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %274 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %265#0[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %276 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%275#2, %275#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %277 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%276 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %278 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %277) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %279 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%278 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %280 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %279[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %281 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%278, %280 : tensor<1024xf32>, f32) outs(%278 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %282 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%281 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %283:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %282[] : tensor<f32>
      %expanded = tensor.expand_shape %281 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %275#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %284 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%283#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%283#1, %283#0, %283#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %285:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %284 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %275#0[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %286 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%285#2, %285#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %287 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%286 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %288 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %287) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %289 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%288 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %290 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %289[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %291 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%288, %290 : tensor<1024xf32>, f32) outs(%288 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %292 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%291 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %293:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %292[] : tensor<f32>
      %expanded = tensor.expand_shape %291 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %285#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %294 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%293#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%293#1, %293#0, %293#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %295:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %294 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %285#0[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %296 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%295#2, %295#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %297 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%296 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %298 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %297) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %299 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%298 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %300 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %299[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %301 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%298, %300 : tensor<1024xf32>, f32) outs(%298 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %302 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%301 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %303:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %302[] : tensor<f32>
      %expanded = tensor.expand_shape %301 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %295#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %304 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%303#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%303#1, %303#0, %303#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %305:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %304 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %295#0[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %306 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%305#2, %305#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %307 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%306 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %308 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %307) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %309 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%308 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %310 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %309[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %311 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%308, %310 : tensor<1024xf32>, f32) outs(%308 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %312 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%311 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %313:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %312[] : tensor<f32>
      %expanded = tensor.expand_shape %311 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %305#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %314 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%313#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%313#1, %313#0, %313#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %315:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %314 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %305#0[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %316 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%315#2, %315#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %317 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%316 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %318 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %317) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %319 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%318 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %320 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %319[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %321 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%318, %320 : tensor<1024xf32>, f32) outs(%318 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %322 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%321 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %323:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %322[] : tensor<f32>
      %expanded = tensor.expand_shape %321 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %315#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %324 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%323#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%323#1, %323#0, %323#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %325:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %324 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %315#0[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %326 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%325#2, %325#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %327 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%326 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %328 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %327) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %329 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%328 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %330 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %329[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %331 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%328, %330 : tensor<1024xf32>, f32) outs(%328 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %332 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%331 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %333:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %332[] : tensor<f32>
      %expanded = tensor.expand_shape %331 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %325#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %334 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%333#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%333#1, %333#0, %333#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %335:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %334 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %325#0[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %336 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%335#2, %335#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %337 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%336 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %338 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %337) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %339 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%338 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %340 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %339[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %341 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%338, %340 : tensor<1024xf32>, f32) outs(%338 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %342 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%341 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %343:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %342[] : tensor<f32>
      %expanded = tensor.expand_shape %341 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %335#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %344 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%343#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%343#1, %343#0, %343#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %345:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %344 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %335#0[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %195#1[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %195#3[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %346 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%345#2, %345#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %347 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%346 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %348 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %347) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %349 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%348 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %350 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %349[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %351 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%348, %350 : tensor<1024xf32>, f32) outs(%348 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %352 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%351 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %353:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %352[] : tensor<f32>
      %expanded = tensor.expand_shape %351 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195#4[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %345#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %354 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%353#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%353#1, %353#0, %353#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %355:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %354 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %345#0[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %1096 = bufferization.materialize_in_destination %inserted_slice in %188 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %1096, %extracted_slice : tensor<768xf32>, tensor<768x768xf32>
    }
    %356 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%355#1, %355#0 : tensor<768x768xf32>, tensor<768xf32>) outs(%184#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.addf %out, %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %357 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg13[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %358 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%356 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %359 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %358[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %360 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%356, %359, %357 : tensor<768xf32>, f32, tensor<768xf32>) outs(%355#0 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %361:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = bufferization.materialize_in_destination %360 in %355#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      cinm.yield %1096, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32>
    }
    %362 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%361#1, %361#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %363 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%361#2, %361#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %364 = cinm.compute on platform #cinm.host_platform -> tensor<768x2048xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      cinm.yield %extracted_slice : tensor<768x2048xf32>
    }
    %365:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%364, %363 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%362, %356 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %1097 = arith.negf %out : f32
        %1098 = math.exp %1097 : f32
        %1099 = arith.addf %1098, %cst_6 : f32
        %1100 = arith.divf %cst_6, %1099 : f32
        %1101 = arith.mulf %out, %1100 : f32
        %1102 = arith.mulf %1101, %in_9 : f32
        %1103 = arith.mulf %in, %1102 : f32
        %1104 = arith.addf %out_10, %1103 : f32
        linalg.yield %1102, %1104 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %1096#0, %1096#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %366 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg5[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %367 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%365#1 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %368 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %367[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %369 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%365#1, %368, %366 : tensor<768xf32>, f32, tensor<768xf32>) outs(%4 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %370:3 = cinm.compute on platform #cinm.host_platform -> tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9, %extracted_slice_10 : tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>
    }
    %371 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%370#0, %369 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %372 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %195#2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %373 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%372 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%370#1, %369 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %374 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %195#0[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %375 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%374 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%370#2, %369 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %376:7 = cinm.compute on platform #cinm.host_platform -> tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %inserted_slice = tensor.insert_slice %375 into %195#0[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %1096:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %371, %arg18 = %373) -> (tensor<768xf32>, tensor<768xf32>) {
        %1097 = arith.remui %arg16, %c48 : index
        %1098 = arith.index_cast %1097 : index to i64
        %1099 = arith.uitofp %1098 : i64 to f32
        %1100 = arith.divf %1099, %cst_7 : f32
        %1101 = math.powf %cst_8, %1100 : f32
        %1102 = arith.divf %cst_6, %1101 : f32
        %1103 = arith.mulf %12#1, %1102 : f32
        %1104 = math.cos %1103 : f32
        %1105 = math.sin %1103 : f32
        %1106 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_13 = tensor.extract %arg17[%1106] : tensor<768xf32>
        %1107 = arith.mulf %extracted, %1104 : f32
        %1108 = arith.mulf %extracted_13, %1105 : f32
        %1109 = arith.subf %1107, %1108 : f32
        %inserted = tensor.insert %1109 into %arg17[%arg16] : tensor<768xf32>
        %1110 = arith.mulf %extracted, %1105 : f32
        %1111 = arith.mulf %extracted_13, %1104 : f32
        %1112 = arith.addf %1110, %1111 : f32
        %inserted_14 = tensor.insert %1112 into %inserted[%1106] : tensor<768xf32>
        %1113 = bufferization.materialize_in_destination %inserted_14 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %1114 = arith.cmpi ult, %arg16, %c768 : index
        %1115 = scf.if %1114 -> (tensor<768xf32>) {
          %extracted_15 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_16 = tensor.extract %arg18[%1106] : tensor<768xf32>
          %1116 = arith.mulf %extracted_15, %1104 : f32
          %1117 = arith.mulf %extracted_16, %1105 : f32
          %1118 = arith.subf %1116, %1117 : f32
          %inserted_17 = tensor.insert %1118 into %arg18[%arg16] : tensor<768xf32>
          %1119 = arith.mulf %extracted_15, %1105 : f32
          %1120 = arith.mulf %extracted_16, %1104 : f32
          %1121 = arith.addf %1119, %1120 : f32
          %inserted_18 = tensor.insert %1121 into %inserted_17[%1106] : tensor<768xf32>
          %1122 = bufferization.materialize_in_destination %inserted_18 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %1122 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %1113, %1115 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %1096#1 into %195#2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice_9[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_11 = tensor.extract_slice %1096#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %1096#0, %inserted_slice_9, %extracted_slice, %extracted_slice_10, %extracted_slice_11, %extracted_slice_12 : tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %377 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%376#6, %376#5 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %378 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%377 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %379 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %378) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %380 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%379 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %381 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %380[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %382 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%379, %381 : tensor<1024xf32>, f32) outs(%379 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %383 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%382 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %384:3 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %383[] : tensor<f32>
      %expanded = tensor.expand_shape %382 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice : f32, tensor<1x1024xf32>, tensor<1024x48xf32>
    }
    %385 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%21#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%384#1, %384#0, %384#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %386:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %385 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %369[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %387 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%386#2, %386#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %388 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%387 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %389 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %388) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %390 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%389 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %391 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %390[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %392 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%389, %391 : tensor<1024xf32>, f32) outs(%389 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %393 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%392 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %394:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %393[] : tensor<f32>
      %expanded = tensor.expand_shape %392 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %386#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %395 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%394#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%394#1, %394#0, %394#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %396:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %395 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %386#0[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %397 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%396#2, %396#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %398 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%397 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %399 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %398) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %400 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%399 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %401 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %400[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %402 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%399, %401 : tensor<1024xf32>, f32) outs(%399 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %403 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%402 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %404:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %403[] : tensor<f32>
      %expanded = tensor.expand_shape %402 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %396#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %405 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%404#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%404#1, %404#0, %404#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %406:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %405 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %396#0[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %407 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%406#2, %406#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %408 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%407 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %409 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %408) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %410 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%409 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %411 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %410[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %412 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%409, %411 : tensor<1024xf32>, f32) outs(%409 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %413 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%412 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %414:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %413[] : tensor<f32>
      %expanded = tensor.expand_shape %412 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %406#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %415 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%414#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%414#1, %414#0, %414#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %416:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %415 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %406#0[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %417 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%416#2, %416#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %418 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%417 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %419 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %418) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %420 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%419 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %421 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %420[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %422 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%419, %421 : tensor<1024xf32>, f32) outs(%419 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %423 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%422 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %424:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %423[] : tensor<f32>
      %expanded = tensor.expand_shape %422 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %416#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %425 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%424#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%424#1, %424#0, %424#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %426:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %425 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %416#0[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %427 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%426#2, %426#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %428 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%427 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %429 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %428) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %430 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%429 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %431 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %430[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %432 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%429, %431 : tensor<1024xf32>, f32) outs(%429 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %433 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%432 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %434:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %433[] : tensor<f32>
      %expanded = tensor.expand_shape %432 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %426#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %435 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%434#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%434#1, %434#0, %434#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %436:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %435 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %426#0[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %437 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%436#2, %436#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %438 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%437 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %439 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %438) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %440 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%439 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %441 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %440[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %442 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%439, %441 : tensor<1024xf32>, f32) outs(%439 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %443 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%442 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %444:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %443[] : tensor<f32>
      %expanded = tensor.expand_shape %442 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %436#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %445 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%444#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%444#1, %444#0, %444#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %446:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %445 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %436#0[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %447 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%446#2, %446#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %448 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%447 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %449 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %448) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %450 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%449 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %451 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %450[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %452 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%449, %451 : tensor<1024xf32>, f32) outs(%449 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %453 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%452 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %454:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %453[] : tensor<f32>
      %expanded = tensor.expand_shape %452 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %446#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %455 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%454#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%454#1, %454#0, %454#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %456:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %455 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %446#0[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %457 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%456#2, %456#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %458 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%457 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %459 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %458) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %460 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%459 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %461 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %460[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %462 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%459, %461 : tensor<1024xf32>, f32) outs(%459 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %463 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%462 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %464:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %463[] : tensor<f32>
      %expanded = tensor.expand_shape %462 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %456#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %465 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%464#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%464#1, %464#0, %464#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %466:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %465 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %456#0[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %467 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%466#2, %466#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %468 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%467 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %469 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %468) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %470 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%469 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %471 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %470[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %472 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%469, %471 : tensor<1024xf32>, f32) outs(%469 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %473 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%472 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %474:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %473[] : tensor<f32>
      %expanded = tensor.expand_shape %472 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %466#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %475 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%474#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%474#1, %474#0, %474#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %476:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %475 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %466#0[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %477 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%476#2, %476#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %478 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%477 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %479 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %478) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %480 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%479 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %481 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %480[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %482 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%479, %481 : tensor<1024xf32>, f32) outs(%479 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %483 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%482 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %484:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %483[] : tensor<f32>
      %expanded = tensor.expand_shape %482 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %476#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %485 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%484#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%484#1, %484#0, %484#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %486:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %485 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %476#0[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %487 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%486#2, %486#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %488 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%487 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %489 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %488) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %490 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%489 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %491 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %490[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %492 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%489, %491 : tensor<1024xf32>, f32) outs(%489 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %493 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%492 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %494:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %493[] : tensor<f32>
      %expanded = tensor.expand_shape %492 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %486#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %495 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%494#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%494#1, %494#0, %494#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %496:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %495 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %486#0[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %497 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%496#2, %496#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %498 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%497 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %499 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %498) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %500 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%499 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %501 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %500[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %502 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%499, %501 : tensor<1024xf32>, f32) outs(%499 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %503 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%502 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %504:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %503[] : tensor<f32>
      %expanded = tensor.expand_shape %502 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %496#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %505 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%504#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%504#1, %504#0, %504#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %506:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %505 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %496#0[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %507 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%506#2, %506#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %508 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%507 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %509 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %508) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %510 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%509 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %511 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %510[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %512 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%509, %511 : tensor<1024xf32>, f32) outs(%509 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %513 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%512 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %514:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %513[] : tensor<f32>
      %expanded = tensor.expand_shape %512 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %506#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %515 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%514#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%514#1, %514#0, %514#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %516:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %515 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %506#0[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %517 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%516#2, %516#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %518 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%517 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %519 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %518) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %520 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%519 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %521 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %520[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %522 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%519, %521 : tensor<1024xf32>, f32) outs(%519 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %523 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%522 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %524:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %523[] : tensor<f32>
      %expanded = tensor.expand_shape %522 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %516#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %525 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%524#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%524#1, %524#0, %524#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %526:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %525 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %516#0[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %376#1[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %376#3[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %527 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%526#2, %526#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %528 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%527 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %529 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %528) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %530 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%529 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %531 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %530[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %532 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%529, %531 : tensor<1024xf32>, f32) outs(%529 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %533 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%532 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %534:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %533[] : tensor<f32>
      %expanded = tensor.expand_shape %532 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %376#4[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %526#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %535 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%534#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%534#1, %534#0, %534#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %536:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %535 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %526#0[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %1096 = bufferization.materialize_in_destination %inserted_slice in %369 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %1096, %extracted_slice : tensor<768xf32>, tensor<768x768xf32>
    }
    %537 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%536#1, %536#0 : tensor<768x768xf32>, tensor<768xf32>) outs(%365#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.addf %out, %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %538 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg13[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %539 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%537 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %540 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %539[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %541 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%537, %540, %538 : tensor<768xf32>, f32, tensor<768xf32>) outs(%536#0 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %542:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = bufferization.materialize_in_destination %541 in %536#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      cinm.yield %1096, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32>
    }
    %543 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%542#1, %542#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %544 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%542#2, %542#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %545 = cinm.compute on platform #cinm.host_platform -> tensor<768x2048xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      cinm.yield %extracted_slice : tensor<768x2048xf32>
    }
    %546:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%545, %544 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%543, %537 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %1097 = arith.negf %out : f32
        %1098 = math.exp %1097 : f32
        %1099 = arith.addf %1098, %cst_6 : f32
        %1100 = arith.divf %cst_6, %1099 : f32
        %1101 = arith.mulf %out, %1100 : f32
        %1102 = arith.mulf %1101, %in_9 : f32
        %1103 = arith.mulf %in, %1102 : f32
        %1104 = arith.addf %out_10, %1103 : f32
        linalg.yield %1102, %1104 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %1096#0, %1096#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %547 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg5[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %548 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%546#1 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %549 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %548[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %550 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%546#1, %549, %547 : tensor<768xf32>, f32, tensor<768xf32>) outs(%4 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %551:3 = cinm.compute on platform #cinm.host_platform -> tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9, %extracted_slice_10 : tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>
    }
    %552 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%551#0, %550 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %553 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %376#2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %554 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%553 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%551#1, %550 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %555 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %376#0[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %556 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%555 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%551#2, %550 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %557:7 = cinm.compute on platform #cinm.host_platform -> tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %inserted_slice = tensor.insert_slice %556 into %376#0[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %1096:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %552, %arg18 = %554) -> (tensor<768xf32>, tensor<768xf32>) {
        %1097 = arith.remui %arg16, %c48 : index
        %1098 = arith.index_cast %1097 : index to i64
        %1099 = arith.uitofp %1098 : i64 to f32
        %1100 = arith.divf %1099, %cst_7 : f32
        %1101 = math.powf %cst_8, %1100 : f32
        %1102 = arith.divf %cst_6, %1101 : f32
        %1103 = arith.mulf %12#1, %1102 : f32
        %1104 = math.cos %1103 : f32
        %1105 = math.sin %1103 : f32
        %1106 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_13 = tensor.extract %arg17[%1106] : tensor<768xf32>
        %1107 = arith.mulf %extracted, %1104 : f32
        %1108 = arith.mulf %extracted_13, %1105 : f32
        %1109 = arith.subf %1107, %1108 : f32
        %inserted = tensor.insert %1109 into %arg17[%arg16] : tensor<768xf32>
        %1110 = arith.mulf %extracted, %1105 : f32
        %1111 = arith.mulf %extracted_13, %1104 : f32
        %1112 = arith.addf %1110, %1111 : f32
        %inserted_14 = tensor.insert %1112 into %inserted[%1106] : tensor<768xf32>
        %1113 = bufferization.materialize_in_destination %inserted_14 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %1114 = arith.cmpi ult, %arg16, %c768 : index
        %1115 = scf.if %1114 -> (tensor<768xf32>) {
          %extracted_15 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_16 = tensor.extract %arg18[%1106] : tensor<768xf32>
          %1116 = arith.mulf %extracted_15, %1104 : f32
          %1117 = arith.mulf %extracted_16, %1105 : f32
          %1118 = arith.subf %1116, %1117 : f32
          %inserted_17 = tensor.insert %1118 into %arg18[%arg16] : tensor<768xf32>
          %1119 = arith.mulf %extracted_15, %1105 : f32
          %1120 = arith.mulf %extracted_16, %1104 : f32
          %1121 = arith.addf %1119, %1120 : f32
          %inserted_18 = tensor.insert %1121 into %inserted_17[%1106] : tensor<768xf32>
          %1122 = bufferization.materialize_in_destination %inserted_18 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %1122 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %1113, %1115 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %1096#1 into %376#2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice_9[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_11 = tensor.extract_slice %1096#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %1096#0, %inserted_slice_9, %extracted_slice, %extracted_slice_10, %extracted_slice_11, %extracted_slice_12 : tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %558 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%557#6, %557#5 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %559 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%558 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %560 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %559) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %561 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%560 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %562 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %561[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %563 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%560, %562 : tensor<1024xf32>, f32) outs(%560 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %564 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%563 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %565:3 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %564[] : tensor<f32>
      %expanded = tensor.expand_shape %563 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice : f32, tensor<1x1024xf32>, tensor<1024x48xf32>
    }
    %566 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%21#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%565#1, %565#0, %565#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %567:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %566 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %550[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %568 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%567#2, %567#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %569 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%568 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %570 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %569) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %571 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%570 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %572 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %571[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %573 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%570, %572 : tensor<1024xf32>, f32) outs(%570 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %574 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%573 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %575:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %574[] : tensor<f32>
      %expanded = tensor.expand_shape %573 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %567#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %576 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%575#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%575#1, %575#0, %575#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %577:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %576 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %567#0[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %578 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%577#2, %577#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %579 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%578 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %580 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %579) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %581 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%580 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %582 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %581[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %583 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%580, %582 : tensor<1024xf32>, f32) outs(%580 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %584 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%583 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %585:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %584[] : tensor<f32>
      %expanded = tensor.expand_shape %583 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %577#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %586 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%585#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%585#1, %585#0, %585#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %587:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %586 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %577#0[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %588 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%587#2, %587#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %589 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%588 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %590 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %589) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %591 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%590 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %592 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %591[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %593 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%590, %592 : tensor<1024xf32>, f32) outs(%590 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %594 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%593 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %595:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %594[] : tensor<f32>
      %expanded = tensor.expand_shape %593 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %587#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %596 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%595#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%595#1, %595#0, %595#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %597:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %596 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %587#0[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %598 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%597#2, %597#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %599 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%598 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %600 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %599) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %601 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%600 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %602 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %601[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %603 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%600, %602 : tensor<1024xf32>, f32) outs(%600 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %604 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%603 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %605:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %604[] : tensor<f32>
      %expanded = tensor.expand_shape %603 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %597#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %606 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%605#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%605#1, %605#0, %605#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %607:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %606 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %597#0[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %608 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%607#2, %607#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %609 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%608 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %610 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %609) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %611 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%610 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %612 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %611[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %613 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%610, %612 : tensor<1024xf32>, f32) outs(%610 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %614 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%613 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %615:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %614[] : tensor<f32>
      %expanded = tensor.expand_shape %613 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %607#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %616 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%615#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%615#1, %615#0, %615#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %617:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %616 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %607#0[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %618 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%617#2, %617#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %619 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%618 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %620 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %619) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %621 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%620 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %622 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %621[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %623 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%620, %622 : tensor<1024xf32>, f32) outs(%620 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %624 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%623 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %625:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %624[] : tensor<f32>
      %expanded = tensor.expand_shape %623 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %617#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %626 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%625#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%625#1, %625#0, %625#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %627:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %626 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %617#0[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %628 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%627#2, %627#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %629 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%628 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %630 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %629) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %631 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%630 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %632 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %631[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %633 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%630, %632 : tensor<1024xf32>, f32) outs(%630 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %634 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%633 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %635:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %634[] : tensor<f32>
      %expanded = tensor.expand_shape %633 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %627#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %636 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%635#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%635#1, %635#0, %635#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %637:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %636 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %627#0[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %638 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%637#2, %637#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %639 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%638 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %640 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %639) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %641 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%640 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %642 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %641[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %643 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%640, %642 : tensor<1024xf32>, f32) outs(%640 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %644 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%643 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %645:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %644[] : tensor<f32>
      %expanded = tensor.expand_shape %643 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %637#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %646 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%645#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%645#1, %645#0, %645#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %647:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %646 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %637#0[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %648 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%647#2, %647#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %649 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%648 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %650 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %649) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %651 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%650 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %652 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %651[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %653 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%650, %652 : tensor<1024xf32>, f32) outs(%650 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %654 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%653 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %655:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %654[] : tensor<f32>
      %expanded = tensor.expand_shape %653 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %647#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %656 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%655#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%655#1, %655#0, %655#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %657:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %656 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %647#0[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %658 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%657#2, %657#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %659 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%658 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %660 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %659) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %661 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%660 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %662 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %661[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %663 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%660, %662 : tensor<1024xf32>, f32) outs(%660 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %664 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%663 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %665:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %664[] : tensor<f32>
      %expanded = tensor.expand_shape %663 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %657#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %666 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%665#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%665#1, %665#0, %665#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %667:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %666 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %657#0[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %668 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%667#2, %667#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %669 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%668 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %670 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %669) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %671 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%670 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %672 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %671[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %673 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%670, %672 : tensor<1024xf32>, f32) outs(%670 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %674 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%673 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %675:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %674[] : tensor<f32>
      %expanded = tensor.expand_shape %673 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %667#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %676 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%675#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%675#1, %675#0, %675#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %677:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %676 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %667#0[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %678 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%677#2, %677#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %679 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%678 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %680 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %679) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %681 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%680 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %682 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %681[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %683 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%680, %682 : tensor<1024xf32>, f32) outs(%680 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %684 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%683 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %685:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %684[] : tensor<f32>
      %expanded = tensor.expand_shape %683 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %677#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %686 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%685#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%685#1, %685#0, %685#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %687:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %686 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %677#0[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %688 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%687#2, %687#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %689 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%688 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %690 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %689) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %691 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%690 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %692 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %691[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %693 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%690, %692 : tensor<1024xf32>, f32) outs(%690 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %694 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%693 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %695:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %694[] : tensor<f32>
      %expanded = tensor.expand_shape %693 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %687#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %696 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%695#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%695#1, %695#0, %695#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %697:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %696 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %687#0[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %698 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%697#2, %697#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %699 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%698 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %700 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %699) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %701 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%700 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %702 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %701[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %703 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%700, %702 : tensor<1024xf32>, f32) outs(%700 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %704 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%703 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %705:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %704[] : tensor<f32>
      %expanded = tensor.expand_shape %703 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %697#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %706 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%705#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%705#1, %705#0, %705#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %707:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %706 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %697#0[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %557#1[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %557#3[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %708 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%707#2, %707#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %709 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%708 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %710 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %709) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %711 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%710 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %712 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %711[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %713 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%710, %712 : tensor<1024xf32>, f32) outs(%710 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %714 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%713 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %715:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %714[] : tensor<f32>
      %expanded = tensor.expand_shape %713 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %557#4[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %707#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %716 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%715#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%715#1, %715#0, %715#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %717:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %716 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %707#0[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %1096 = bufferization.materialize_in_destination %inserted_slice in %550 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %1096, %extracted_slice : tensor<768xf32>, tensor<768x768xf32>
    }
    %718 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%717#1, %717#0 : tensor<768x768xf32>, tensor<768xf32>) outs(%546#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.addf %out, %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %719 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg13[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %720 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%718 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %721 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %720[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %722 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%718, %721, %719 : tensor<768xf32>, f32, tensor<768xf32>) outs(%717#0 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %723:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = bufferization.materialize_in_destination %722 in %717#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      cinm.yield %1096, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32>
    }
    %724 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%723#1, %723#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %725 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%723#2, %723#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %726 = cinm.compute on platform #cinm.host_platform -> tensor<768x2048xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      cinm.yield %extracted_slice : tensor<768x2048xf32>
    }
    %727:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%726, %725 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%724, %718 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %1097 = arith.negf %out : f32
        %1098 = math.exp %1097 : f32
        %1099 = arith.addf %1098, %cst_6 : f32
        %1100 = arith.divf %cst_6, %1099 : f32
        %1101 = arith.mulf %out, %1100 : f32
        %1102 = arith.mulf %1101, %in_9 : f32
        %1103 = arith.mulf %in, %1102 : f32
        %1104 = arith.addf %out_10, %1103 : f32
        linalg.yield %1102, %1104 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %1096#0, %1096#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %728 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg5[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %729 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%727#1 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %730 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %729[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %731 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%727#1, %730, %728 : tensor<768xf32>, f32, tensor<768xf32>) outs(%4 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %732:3 = cinm.compute on platform #cinm.host_platform -> tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9, %extracted_slice_10 : tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>
    }
    %733 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%732#0, %731 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %734 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %557#2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %735 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%734 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%732#1, %731 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %736 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %557#0[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %737 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%736 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%732#2, %731 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %738:7 = cinm.compute on platform #cinm.host_platform -> tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %inserted_slice = tensor.insert_slice %737 into %557#0[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %1096:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %733, %arg18 = %735) -> (tensor<768xf32>, tensor<768xf32>) {
        %1097 = arith.remui %arg16, %c48 : index
        %1098 = arith.index_cast %1097 : index to i64
        %1099 = arith.uitofp %1098 : i64 to f32
        %1100 = arith.divf %1099, %cst_7 : f32
        %1101 = math.powf %cst_8, %1100 : f32
        %1102 = arith.divf %cst_6, %1101 : f32
        %1103 = arith.mulf %12#1, %1102 : f32
        %1104 = math.cos %1103 : f32
        %1105 = math.sin %1103 : f32
        %1106 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_13 = tensor.extract %arg17[%1106] : tensor<768xf32>
        %1107 = arith.mulf %extracted, %1104 : f32
        %1108 = arith.mulf %extracted_13, %1105 : f32
        %1109 = arith.subf %1107, %1108 : f32
        %inserted = tensor.insert %1109 into %arg17[%arg16] : tensor<768xf32>
        %1110 = arith.mulf %extracted, %1105 : f32
        %1111 = arith.mulf %extracted_13, %1104 : f32
        %1112 = arith.addf %1110, %1111 : f32
        %inserted_14 = tensor.insert %1112 into %inserted[%1106] : tensor<768xf32>
        %1113 = bufferization.materialize_in_destination %inserted_14 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %1114 = arith.cmpi ult, %arg16, %c768 : index
        %1115 = scf.if %1114 -> (tensor<768xf32>) {
          %extracted_15 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_16 = tensor.extract %arg18[%1106] : tensor<768xf32>
          %1116 = arith.mulf %extracted_15, %1104 : f32
          %1117 = arith.mulf %extracted_16, %1105 : f32
          %1118 = arith.subf %1116, %1117 : f32
          %inserted_17 = tensor.insert %1118 into %arg18[%arg16] : tensor<768xf32>
          %1119 = arith.mulf %extracted_15, %1105 : f32
          %1120 = arith.mulf %extracted_16, %1104 : f32
          %1121 = arith.addf %1119, %1120 : f32
          %inserted_18 = tensor.insert %1121 into %inserted_17[%1106] : tensor<768xf32>
          %1122 = bufferization.materialize_in_destination %inserted_18 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %1122 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %1113, %1115 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %1096#1 into %557#2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice_9[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_11 = tensor.extract_slice %1096#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %1096#0, %inserted_slice_9, %extracted_slice, %extracted_slice_10, %extracted_slice_11, %extracted_slice_12 : tensor<6x1024x768xf32>, tensor<768xf32>, tensor<6x1024x768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %739 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%738#6, %738#5 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %740 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%739 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %741 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %740) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %742 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%741 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %743 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %742[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %744 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%741, %743 : tensor<1024xf32>, f32) outs(%741 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %745 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%744 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %746:3 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %745[] : tensor<f32>
      %expanded = tensor.expand_shape %744 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice : f32, tensor<1x1024xf32>, tensor<1024x48xf32>
    }
    %747 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%21#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%746#1, %746#0, %746#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %748:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %747 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %731[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %749 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%748#2, %748#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %750 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%749 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %751 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %750) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %752 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%751 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %753 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %752[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %754 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%751, %753 : tensor<1024xf32>, f32) outs(%751 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %755 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%754 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %756:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %755[] : tensor<f32>
      %expanded = tensor.expand_shape %754 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %748#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %757 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%756#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%756#1, %756#0, %756#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %758:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %757 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %748#0[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %759 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%758#2, %758#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %760 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%759 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %761 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %760) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %762 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%761 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %763 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %762[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %764 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%761, %763 : tensor<1024xf32>, f32) outs(%761 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %765 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%764 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %766:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %765[] : tensor<f32>
      %expanded = tensor.expand_shape %764 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %758#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %767 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%766#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%766#1, %766#0, %766#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %768:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %767 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %758#0[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %769 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%768#2, %768#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %770 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%769 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %771 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %770) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %772 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%771 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %773 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %772[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %774 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%771, %773 : tensor<1024xf32>, f32) outs(%771 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %775 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%774 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %776:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %775[] : tensor<f32>
      %expanded = tensor.expand_shape %774 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %768#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %777 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%776#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%776#1, %776#0, %776#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %778:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %777 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %768#0[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %779 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%778#2, %778#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %780 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%779 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %781 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %780) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %782 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%781 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %783 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %782[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %784 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%781, %783 : tensor<1024xf32>, f32) outs(%781 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %785 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%784 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %786:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %785[] : tensor<f32>
      %expanded = tensor.expand_shape %784 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %778#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %787 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%786#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%786#1, %786#0, %786#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %788:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %787 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %778#0[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %789 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%788#2, %788#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %790 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%789 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %791 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %790) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %792 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%791 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %793 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %792[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %794 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%791, %793 : tensor<1024xf32>, f32) outs(%791 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %795 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%794 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %796:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %795[] : tensor<f32>
      %expanded = tensor.expand_shape %794 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %788#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %797 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%796#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%796#1, %796#0, %796#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %798:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %797 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %788#0[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %799 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%798#2, %798#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %800 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%799 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %801 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %800) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %802 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%801 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %803 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %802[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %804 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%801, %803 : tensor<1024xf32>, f32) outs(%801 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %805 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%804 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %806:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %805[] : tensor<f32>
      %expanded = tensor.expand_shape %804 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %798#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %807 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%806#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%806#1, %806#0, %806#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %808:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %807 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %798#0[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %809 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%808#2, %808#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %810 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%809 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %811 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %810) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %812 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%811 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %813 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %812[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %814 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%811, %813 : tensor<1024xf32>, f32) outs(%811 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %815 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%814 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %816:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %815[] : tensor<f32>
      %expanded = tensor.expand_shape %814 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %808#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %817 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%816#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%816#1, %816#0, %816#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %818:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %817 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %808#0[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %819 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%818#2, %818#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %820 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%819 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %821 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %820) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %822 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%821 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %823 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %822[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %824 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%821, %823 : tensor<1024xf32>, f32) outs(%821 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %825 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%824 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %826:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %825[] : tensor<f32>
      %expanded = tensor.expand_shape %824 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %818#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %827 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%826#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%826#1, %826#0, %826#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %828:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %827 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %818#0[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %829 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%828#2, %828#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %830 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%829 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %831 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %830) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %832 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%831 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %833 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %832[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %834 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%831, %833 : tensor<1024xf32>, f32) outs(%831 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %835 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%834 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %836:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %835[] : tensor<f32>
      %expanded = tensor.expand_shape %834 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %828#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %837 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%836#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%836#1, %836#0, %836#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %838:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %837 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %828#0[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %839 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%838#2, %838#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %840 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%839 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %841 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %840) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %842 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%841 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %843 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %842[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %844 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%841, %843 : tensor<1024xf32>, f32) outs(%841 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %845 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%844 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %846:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %845[] : tensor<f32>
      %expanded = tensor.expand_shape %844 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %838#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %847 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%846#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%846#1, %846#0, %846#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %848:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %847 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %838#0[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %849 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%848#2, %848#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %850 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%849 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %851 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %850) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %852 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%851 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %853 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %852[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %854 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%851, %853 : tensor<1024xf32>, f32) outs(%851 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %855 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%854 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %856:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %855[] : tensor<f32>
      %expanded = tensor.expand_shape %854 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %848#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %857 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%856#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%856#1, %856#0, %856#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %858:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %857 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %848#0[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %859 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%858#2, %858#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %860 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%859 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %861 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %860) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %862 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%861 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %863 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %862[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %864 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%861, %863 : tensor<1024xf32>, f32) outs(%861 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %865 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%864 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %866:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %865[] : tensor<f32>
      %expanded = tensor.expand_shape %864 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %858#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %867 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%866#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%866#1, %866#0, %866#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %868:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %867 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %858#0[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %869 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%868#2, %868#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %870 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%869 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %871 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %870) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %872 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%871 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %873 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %872[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %874 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%871, %873 : tensor<1024xf32>, f32) outs(%871 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %875 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%874 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %876:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %875[] : tensor<f32>
      %expanded = tensor.expand_shape %874 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %868#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %877 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%876#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%876#1, %876#0, %876#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %878:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %877 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %868#0[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %879 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%878#2, %878#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %880 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%879 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %881 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %880) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %882 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%881 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %883 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %882[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %884 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%881, %883 : tensor<1024xf32>, f32) outs(%881 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %885 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%884 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %886:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %885[] : tensor<f32>
      %expanded = tensor.expand_shape %884 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %878#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %887 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%886#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%886#1, %886#0, %886#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %888:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %887 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %878#0[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %738#1[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %738#3[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %889 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%888#2, %888#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %890 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%889 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %891 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %890) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %892 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%891 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %893 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %892[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %894 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%891, %893 : tensor<1024xf32>, f32) outs(%891 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %895 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%894 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %896:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %895[] : tensor<f32>
      %expanded = tensor.expand_shape %894 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %738#4[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %888#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %897 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%896#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%896#1, %896#0, %896#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %898:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %897 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %888#0[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %1096 = bufferization.materialize_in_destination %inserted_slice in %731 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %1096, %extracted_slice : tensor<768xf32>, tensor<768x768xf32>
    }
    %899 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%898#1, %898#0 : tensor<768x768xf32>, tensor<768xf32>) outs(%727#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.addf %out, %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %900 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg13[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %901 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%899 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %902 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %901[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %903 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%899, %902, %900 : tensor<768xf32>, f32, tensor<768xf32>) outs(%898#0 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %904:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = bufferization.materialize_in_destination %903 in %898#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      cinm.yield %1096, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32>
    }
    %905 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%904#1, %904#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %906 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%904#2, %904#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %907 = cinm.compute on platform #cinm.host_platform -> tensor<768x2048xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      cinm.yield %extracted_slice : tensor<768x2048xf32>
    }
    %908:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%907, %906 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%905, %899 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %1097 = arith.negf %out : f32
        %1098 = math.exp %1097 : f32
        %1099 = arith.addf %1098, %cst_6 : f32
        %1100 = arith.divf %cst_6, %1099 : f32
        %1101 = arith.mulf %out, %1100 : f32
        %1102 = arith.mulf %1101, %in_9 : f32
        %1103 = arith.mulf %in, %1102 : f32
        %1104 = arith.addf %out_10, %1103 : f32
        linalg.yield %1102, %1104 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %1096#0, %1096#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %909 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg5[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %910 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%908#1 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %911 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %910[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %912 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%908#1, %911, %909 : tensor<768xf32>, f32, tensor<768xf32>) outs(%4 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %913:3 = cinm.compute on platform #cinm.host_platform -> tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_10 = tensor.extract_slice %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %extracted_slice, %extracted_slice_9, %extracted_slice_10 : tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>
    }
    %914 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%4 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%913#0, %912 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %915 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %738#2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %916 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%915 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%913#1, %912 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %917 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %738#0[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %918 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%917 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%913#2, %912 : tensor<768x768xf32>, tensor<768xf32>) outs(%1096 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<768xf32>
      cinm.yield %1097 : tensor<768xf32>
    }
    %919:5 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %inserted_slice = tensor.insert_slice %918 into %738#0[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %1096:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %914, %arg18 = %916) -> (tensor<768xf32>, tensor<768xf32>) {
        %1097 = arith.remui %arg16, %c48 : index
        %1098 = arith.index_cast %1097 : index to i64
        %1099 = arith.uitofp %1098 : i64 to f32
        %1100 = arith.divf %1099, %cst_7 : f32
        %1101 = math.powf %cst_8, %1100 : f32
        %1102 = arith.divf %cst_6, %1101 : f32
        %1103 = arith.mulf %12#1, %1102 : f32
        %1104 = math.cos %1103 : f32
        %1105 = math.sin %1103 : f32
        %1106 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_13 = tensor.extract %arg17[%1106] : tensor<768xf32>
        %1107 = arith.mulf %extracted, %1104 : f32
        %1108 = arith.mulf %extracted_13, %1105 : f32
        %1109 = arith.subf %1107, %1108 : f32
        %inserted = tensor.insert %1109 into %arg17[%arg16] : tensor<768xf32>
        %1110 = arith.mulf %extracted, %1105 : f32
        %1111 = arith.mulf %extracted_13, %1104 : f32
        %1112 = arith.addf %1110, %1111 : f32
        %inserted_14 = tensor.insert %1112 into %inserted[%1106] : tensor<768xf32>
        %1113 = bufferization.materialize_in_destination %inserted_14 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %1114 = arith.cmpi ult, %arg16, %c768 : index
        %1115 = scf.if %1114 -> (tensor<768xf32>) {
          %extracted_15 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_16 = tensor.extract %arg18[%1106] : tensor<768xf32>
          %1116 = arith.mulf %extracted_15, %1104 : f32
          %1117 = arith.mulf %extracted_16, %1105 : f32
          %1118 = arith.subf %1116, %1117 : f32
          %inserted_17 = tensor.insert %1118 into %arg18[%arg16] : tensor<768xf32>
          %1119 = arith.mulf %extracted_15, %1105 : f32
          %1120 = arith.mulf %extracted_16, %1104 : f32
          %1121 = arith.addf %1119, %1120 : f32
          %inserted_18 = tensor.insert %1121 into %inserted_17[%1106] : tensor<768xf32>
          %1122 = bufferization.materialize_in_destination %inserted_18 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %1122 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %1113, %1115 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %1096#1 into %738#2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %extracted_slice = tensor.extract_slice %inserted_slice_9[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_10 = tensor.extract_slice %inserted_slice[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_11 = tensor.extract_slice %1096#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %1096#0, %extracted_slice, %extracted_slice_10, %extracted_slice_11, %extracted_slice_12 : tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %920 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%919#4, %919#3 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %921 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%920 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %922 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %921) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %923 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%922 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %924 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %923[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %925 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%922, %924 : tensor<1024xf32>, f32) outs(%922 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %926 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%925 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %927:3 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %926[] : tensor<f32>
      %expanded = tensor.expand_shape %925 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice : f32, tensor<1x1024xf32>, tensor<1024x48xf32>
    }
    %928 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%21#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%927#1, %927#0, %927#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %929:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %928 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %912[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %930 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%929#2, %929#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %931 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%930 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %932 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %931) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %933 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%932 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %934 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %933[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %935 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%932, %934 : tensor<1024xf32>, f32) outs(%932 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %936 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%935 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %937:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %936[] : tensor<f32>
      %expanded = tensor.expand_shape %935 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %929#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %938 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%937#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%937#1, %937#0, %937#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %939:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %938 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %929#0[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %940 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%939#2, %939#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %941 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%940 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %942 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %941) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %943 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%942 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %944 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %943[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %945 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%942, %944 : tensor<1024xf32>, f32) outs(%942 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %946 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%945 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %947:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %946[] : tensor<f32>
      %expanded = tensor.expand_shape %945 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %939#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %948 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%947#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%947#1, %947#0, %947#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %949:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %948 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %939#0[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %950 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%949#2, %949#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %951 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%950 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %952 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %951) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %953 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%952 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %954 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %953[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %955 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%952, %954 : tensor<1024xf32>, f32) outs(%952 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %956 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%955 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %957:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %956[] : tensor<f32>
      %expanded = tensor.expand_shape %955 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %949#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %958 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%957#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%957#1, %957#0, %957#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %959:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %958 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %949#0[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %960 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%959#2, %959#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %961 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%960 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %962 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %961) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %963 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%962 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %964 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %963[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %965 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%962, %964 : tensor<1024xf32>, f32) outs(%962 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %966 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%965 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %967:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %966[] : tensor<f32>
      %expanded = tensor.expand_shape %965 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %959#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %968 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%967#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%967#1, %967#0, %967#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %969:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %968 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %959#0[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %970 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%969#2, %969#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %971 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%970 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %972 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %971) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %973 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%972 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %974 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %973[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %975 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%972, %974 : tensor<1024xf32>, f32) outs(%972 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %976 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%975 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %977:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %976[] : tensor<f32>
      %expanded = tensor.expand_shape %975 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %969#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %978 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%977#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%977#1, %977#0, %977#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %979:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %978 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %969#0[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %980 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%979#2, %979#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %981 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%980 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %982 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %981) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %983 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%982 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %984 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %983[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %985 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%982, %984 : tensor<1024xf32>, f32) outs(%982 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %986 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%985 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %987:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %986[] : tensor<f32>
      %expanded = tensor.expand_shape %985 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %979#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %988 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%987#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%987#1, %987#0, %987#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %989:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %988 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %979#0[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %990 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%989#2, %989#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %991 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%990 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %992 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %991) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %993 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%992 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %994 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %993[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %995 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%992, %994 : tensor<1024xf32>, f32) outs(%992 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %996 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%995 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %997:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %996[] : tensor<f32>
      %expanded = tensor.expand_shape %995 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %989#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %998 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%997#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%997#1, %997#0, %997#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %999:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %998 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %989#0[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1000 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%999#2, %999#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1001 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1000 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1002 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1001) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1003 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1002 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1004 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1003[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1005 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1002, %1004 : tensor<1024xf32>, f32) outs(%1002 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1006 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1005 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1007:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1006[] : tensor<f32>
      %expanded = tensor.expand_shape %1005 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %999#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1008 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1007#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1007#1, %1007#0, %1007#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1009:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1008 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %999#0[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1010 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1009#2, %1009#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1011 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1010 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1012 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1011) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1013 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1012 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1014 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1013[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1015 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1012, %1014 : tensor<1024xf32>, f32) outs(%1012 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1016 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1015 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1017:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1016[] : tensor<f32>
      %expanded = tensor.expand_shape %1015 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1009#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1018 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1017#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1017#1, %1017#0, %1017#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1019:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1018 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1009#0[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1020 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1019#2, %1019#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1021 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1020 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1022 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1021) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1023 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1022 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1024 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1023[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1025 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1022, %1024 : tensor<1024xf32>, f32) outs(%1022 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1026 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1025 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1027:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1026[] : tensor<f32>
      %expanded = tensor.expand_shape %1025 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1019#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1028 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1027#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1027#1, %1027#0, %1027#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1029:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1028 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1019#0[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1030 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1029#2, %1029#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1031 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1030 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1032 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1031) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1033 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1032 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1034 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1033[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1035 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1032, %1034 : tensor<1024xf32>, f32) outs(%1032 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1036 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1035 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1037:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1036[] : tensor<f32>
      %expanded = tensor.expand_shape %1035 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1029#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1038 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1037#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1037#1, %1037#0, %1037#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1039:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1038 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1029#0[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1040 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1039#2, %1039#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1041 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1040 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1042 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1041) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1043 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1042 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1044 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1043[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1045 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1042, %1044 : tensor<1024xf32>, f32) outs(%1042 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1046 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1045 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1047:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1046[] : tensor<f32>
      %expanded = tensor.expand_shape %1045 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1039#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1048 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1047#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1047#1, %1047#0, %1047#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1049:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1048 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1039#0[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1050 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1049#2, %1049#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1051 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1050 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1052 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1051) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1053 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1052 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1054 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1053[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1055 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1052, %1054 : tensor<1024xf32>, f32) outs(%1052 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1056 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1055 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1057:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1056[] : tensor<f32>
      %expanded = tensor.expand_shape %1055 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1049#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1058 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1057#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1057#1, %1057#0, %1057#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1059:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1058 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1049#0[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1060 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1059#2, %1059#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1061 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1060 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1062 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1061) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1063 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1062 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1064 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1063[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1065 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1062, %1064 : tensor<1024xf32>, f32) outs(%1062 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1066 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1065 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1067:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1066[] : tensor<f32>
      %expanded = tensor.expand_shape %1065 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1059#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1068 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1067#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1067#1, %1067#0, %1067#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1069:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1068 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1059#0[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %extracted_slice = tensor.extract_slice %919#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %extracted_slice_9 = tensor.extract_slice %919#1[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      cinm.yield %inserted_slice, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<48xf32>, tensor<1024x48xf32>
    }
    %1070 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%13 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1069#2, %1069#1 : tensor<1024x48xf32>, tensor<48xf32>) outs(%1096 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<1024xf32>
      cinm.yield %1097 : tensor<1024xf32>
    }
    %1071 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1070 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1097 = arith.divf %in, %cst_0 : f32
        linalg.yield %1097 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1072 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = scf.for %arg16 = %12#6 to %c1024 step %c1 iter_args(%arg17 = %1071) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1073 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1072 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.maxnumf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1074 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1073[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1075 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%1072, %1074 : tensor<1024xf32>, f32) outs(%1072 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.subf %in, %in_9 : f32
        %1098 = math.exp %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<1024xf32>
      cinm.yield %1096 : tensor<1024xf32>
    }
    %1076 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1075 : tensor<1024xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.addf %in, %out : f32
        linalg.yield %1098 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1077:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1076[] : tensor<f32>
      %expanded = tensor.expand_shape %1075 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %919#2[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_9 = tensor.extract_slice %1069#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice_9(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %extracted, %expanded, %extracted_slice, %reshape : f32, tensor<1x1024xf32>, tensor<1024x48xf32>, tensor<1x48xf32>
    }
    %1078 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1077#3 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %1097 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1077#1, %1077#0, %1077#2 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%1096 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1098 = arith.divf %in, %in_9 : f32
        %1099 = arith.mulf %1098, %in_10 : f32
        %1100 = arith.addf %out, %1099 : f32
        linalg.yield %1100 : f32
      } -> tensor<1x48xf32>
      cinm.yield %1097 : tensor<1x48xf32>
    }
    %1079:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %1078 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %1069#0[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %1096 = bufferization.materialize_in_destination %inserted_slice in %912 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      cinm.yield %1096, %extracted_slice : tensor<768xf32>, tensor<768x768xf32>
    }
    %1080 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1079#1, %1079#0 : tensor<768x768xf32>, tensor<768xf32>) outs(%908#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.addf %out, %1097 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %1081 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg13[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      cinm.yield %extracted_slice : tensor<768xf32>
    }
    %1082 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1080 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1083 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1082[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      cinm.yield %1098 : f32
    }
    %1084 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%1080, %1083, %1081 : tensor<768xf32>, f32, tensor<768xf32>) outs(%1079#0 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %1097 = arith.mulf %in, %in_9 : f32
        %1098 = arith.mulf %1097, %in_10 : f32
        linalg.yield %1098 : f32
      } -> tensor<768xf32>
      cinm.yield %1096 : tensor<768xf32>
    }
    %1085:3 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %1096 = bufferization.materialize_in_destination %1084 in %1079#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %extracted_slice = tensor.extract_slice %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      cinm.yield %1096, %extracted_slice, %extracted_slice_9 : tensor<768xf32>, tensor<2048x768xf32>, tensor<2048x768xf32>
    }
    %1086 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1085#1, %1085#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %1087 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%180 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1085#2, %1085#0 : tensor<2048x768xf32>, tensor<768xf32>) outs(%1096 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %1098 = arith.mulf %in, %in_9 : f32
        %1099 = arith.addf %out, %1098 : f32
        linalg.yield %1099 : f32
      } -> tensor<2048xf32>
      cinm.yield %1097 : tensor<2048xf32>
    }
    %1088 = cinm.compute on platform #cinm.host_platform -> tensor<768x2048xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      cinm.yield %extracted_slice : tensor<768x2048xf32>
    }
    %1089:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1088, %1087 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%1086, %1080 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %1097 = arith.negf %out : f32
        %1098 = math.exp %1097 : f32
        %1099 = arith.addf %1098, %cst_6 : f32
        %1100 = arith.divf %cst_6, %1099 : f32
        %1101 = arith.mulf %out, %1100 : f32
        %1102 = arith.mulf %1101, %in_9 : f32
        %1103 = arith.mulf %in, %1102 : f32
        %1104 = arith.addf %out_10, %1103 : f32
        linalg.yield %1102, %1104 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %1096#0, %1096#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %1090 = cinm.compute -> tensor<f32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %1097 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%1089#1 : tensor<768xf32>) outs(%1096 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %1098 = arith.mulf %in, %in : f32
        %1099 = arith.addf %1098, %out : f32
        linalg.yield %1099 : f32
      } -> tensor<f32>
      cinm.yield %1097 : tensor<f32>
    }
    %1091 = tensor.empty() : tensor<34048x768xf32>
    %1092:2 = cinm.compute on platform #cinm.host_platform -> f32, tensor<34048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted = tensor.extract %1090[] : tensor<f32>
      %1096 = arith.divf %extracted, %cst_4 : f32
      %1097 = arith.addf %1096, %cst_5 : f32
      %1098 = math.rsqrt %1097 : f32
      %1099 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1091 : tensor<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<34048x768xf32>
      %inserted_slice = tensor.insert_slice %arg15 into %1099[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
      cinm.yield %1098, %inserted_slice : f32, tensor<34048x768xf32>
    }
    %1093 = tensor.empty() : tensor<34048xf32>
    %1094 = cinm.compute -> tensor<34048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1096 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%1093 : tensor<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<34048xf32>
      %1097 = linalg.generic {indexing_maps = [#map3, #map4, #map10, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%1092#1, %1089#1, %1092#0, %arg14 : tensor<34048x768xf32>, tensor<768xf32>, f32, tensor<768xf32>) outs(%1096 : tensor<34048xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %out: f32):
        %1098 = arith.mulf %in_9, %in_10 : f32
        %1099 = arith.mulf %1098, %in_11 : f32
        %1100 = arith.mulf %in, %1099 : f32
        %1101 = arith.addf %out, %1100 : f32
        linalg.yield %1101 : f32
      } -> tensor<34048xf32>
      cinm.yield %1097 : tensor<34048xf32>
    }
    %1095 = cinm.compute on platform #cinm.host_platform -> tensor<32000xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %1094[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
      cinm.yield %extracted_slice : tensor<32000xf32>
    }
    return %1095 : tensor<32000xf32>
  }
  func.func @rot(%arg0: tensor<768xf32> {bufferization.writable = true}, %arg1: index, %arg2: f32, %arg3: f32) -> tensor<768xf32> {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %extracted = tensor.extract %arg0[%arg1] : tensor<768xf32>
    %extracted_0 = tensor.extract %arg0[%0] : tensor<768xf32>
    %1 = arith.mulf %extracted, %arg2 : f32
    %2 = arith.mulf %extracted_0, %arg3 : f32
    %3 = arith.subf %1, %2 : f32
    %inserted = tensor.insert %3 into %arg0[%arg1] : tensor<768xf32>
    %4 = arith.mulf %extracted, %arg3 : f32
    %5 = arith.mulf %extracted_0, %arg2 : f32
    %6 = arith.addf %4, %5 : f32
    %inserted_1 = tensor.insert %6 into %inserted[%0] : tensor<768xf32>
    %7 = bufferization.materialize_in_destination %inserted_1 in %arg0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    return %7 : tensor<768xf32>
  }
  func.func @mha(%arg0: tensor<768xf32>, %arg1: tensor<1024x768xf32>, %arg2: tensor<1024x768xf32>, %arg3: index) -> tensor<768xf32> {
    %cst = arith.constant dense<[1, 48]> : tensor<2xindex>
    %cst_0 = arith.constant 0xFFC00000 : f32
    %cst_1 = arith.constant 6.92820311 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_3 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %extracted_slice = tensor.extract_slice %arg0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_4 = tensor.extract_slice %arg1[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %2 = tensor.empty() : tensor<1024xf32>
    %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<1024xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1024xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_4, %extracted_slice : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %6 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %5) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %7 = tensor.empty() : tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%7 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<f32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%6 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %9[] : tensor<f32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%6, %extracted : tensor<1024xf32>, f32) outs(%6 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %11 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%7 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<f32>
    %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%10 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_5 = tensor.extract %12[] : tensor<f32>
    %expanded = tensor.expand_shape %10 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_6 = tensor.extract_slice %arg2[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_7 = tensor.extract_slice %1[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape = tensor.reshape %extracted_slice_7(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %13 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %14 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded, %extracted_5, %extracted_slice_6 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%13 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed = tensor.collapse_shape %14 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %1[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_8 = tensor.extract_slice %arg0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_9 = tensor.extract_slice %arg1[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_8 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%15 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %17 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %16) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%17 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_10 = tensor.extract %18[] : tensor<f32>
    %19 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%17, %extracted_10 : tensor<1024xf32>, f32) outs(%17 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%19 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_11 = tensor.extract %20[] : tensor<f32>
    %expanded_12 = tensor.expand_shape %19 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_13 = tensor.extract_slice %arg2[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_14 = tensor.extract_slice %inserted_slice[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_15 = tensor.reshape %extracted_slice_14(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %21 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_15 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %22 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_12, %extracted_11, %extracted_slice_13 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%21 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_16 = tensor.collapse_shape %22 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_17 = tensor.insert_slice %collapsed_16 into %inserted_slice[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_18 = tensor.extract_slice %arg0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_19 = tensor.extract_slice %arg1[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %23 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_19, %extracted_slice_18 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%23 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %25 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %24) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %26 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%25 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_20 = tensor.extract %26[] : tensor<f32>
    %27 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%25, %extracted_20 : tensor<1024xf32>, f32) outs(%25 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%27 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_21 = tensor.extract %28[] : tensor<f32>
    %expanded_22 = tensor.expand_shape %27 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_23 = tensor.extract_slice %arg2[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_24 = tensor.extract_slice %inserted_slice_17[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_25 = tensor.reshape %extracted_slice_24(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %29 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_25 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %30 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_22, %extracted_21, %extracted_slice_23 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%29 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_26 = tensor.collapse_shape %30 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_27 = tensor.insert_slice %collapsed_26 into %inserted_slice_17[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_28 = tensor.extract_slice %arg0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_29 = tensor.extract_slice %arg1[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %31 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_29, %extracted_slice_28 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %32 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%31 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %33 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %32) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %34 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%33 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_30 = tensor.extract %34[] : tensor<f32>
    %35 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%33, %extracted_30 : tensor<1024xf32>, f32) outs(%33 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %36 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%35 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_31 = tensor.extract %36[] : tensor<f32>
    %expanded_32 = tensor.expand_shape %35 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_33 = tensor.extract_slice %arg2[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_34 = tensor.extract_slice %inserted_slice_27[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_35 = tensor.reshape %extracted_slice_34(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %37 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_35 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %38 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_32, %extracted_31, %extracted_slice_33 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%37 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_36 = tensor.collapse_shape %38 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_37 = tensor.insert_slice %collapsed_36 into %inserted_slice_27[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_38 = tensor.extract_slice %arg0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_39 = tensor.extract_slice %arg1[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %39 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_39, %extracted_slice_38 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %40 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%39 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %41 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %40) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %42 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%41 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_40 = tensor.extract %42[] : tensor<f32>
    %43 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%41, %extracted_40 : tensor<1024xf32>, f32) outs(%41 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%43 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_41 = tensor.extract %44[] : tensor<f32>
    %expanded_42 = tensor.expand_shape %43 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_43 = tensor.extract_slice %arg2[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_44 = tensor.extract_slice %inserted_slice_37[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_45 = tensor.reshape %extracted_slice_44(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %45 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_45 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %46 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_42, %extracted_41, %extracted_slice_43 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%45 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_46 = tensor.collapse_shape %46 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_47 = tensor.insert_slice %collapsed_46 into %inserted_slice_37[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_48 = tensor.extract_slice %arg0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_49 = tensor.extract_slice %arg1[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %47 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_49, %extracted_slice_48 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%47 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %49 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %48) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %50 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%49 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_50 = tensor.extract %50[] : tensor<f32>
    %51 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%49, %extracted_50 : tensor<1024xf32>, f32) outs(%49 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %52 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%51 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_51 = tensor.extract %52[] : tensor<f32>
    %expanded_52 = tensor.expand_shape %51 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_53 = tensor.extract_slice %arg2[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_54 = tensor.extract_slice %inserted_slice_47[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_55 = tensor.reshape %extracted_slice_54(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %53 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_55 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %54 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_52, %extracted_51, %extracted_slice_53 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%53 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_56 = tensor.collapse_shape %54 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_57 = tensor.insert_slice %collapsed_56 into %inserted_slice_47[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_58 = tensor.extract_slice %arg0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_59 = tensor.extract_slice %arg1[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %55 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_59, %extracted_slice_58 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %56 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%55 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %57 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %56) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %58 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%57 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_60 = tensor.extract %58[] : tensor<f32>
    %59 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%57, %extracted_60 : tensor<1024xf32>, f32) outs(%57 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%59 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_61 = tensor.extract %60[] : tensor<f32>
    %expanded_62 = tensor.expand_shape %59 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_63 = tensor.extract_slice %arg2[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_64 = tensor.extract_slice %inserted_slice_57[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_65 = tensor.reshape %extracted_slice_64(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %61 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_65 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %62 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_62, %extracted_61, %extracted_slice_63 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%61 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_66 = tensor.collapse_shape %62 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_67 = tensor.insert_slice %collapsed_66 into %inserted_slice_57[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_68 = tensor.extract_slice %arg0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_69 = tensor.extract_slice %arg1[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_69, %extracted_slice_68 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %65 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %64) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %66 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%65 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_70 = tensor.extract %66[] : tensor<f32>
    %67 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%65, %extracted_70 : tensor<1024xf32>, f32) outs(%65 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %68 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%67 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_71 = tensor.extract %68[] : tensor<f32>
    %expanded_72 = tensor.expand_shape %67 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_73 = tensor.extract_slice %arg2[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_74 = tensor.extract_slice %inserted_slice_67[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_75 = tensor.reshape %extracted_slice_74(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %69 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_75 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %70 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_72, %extracted_71, %extracted_slice_73 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%69 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_76 = tensor.collapse_shape %70 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_77 = tensor.insert_slice %collapsed_76 into %inserted_slice_67[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_78 = tensor.extract_slice %arg0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_79 = tensor.extract_slice %arg1[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %71 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_79, %extracted_slice_78 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %72 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%71 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %73 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %72) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %74 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%73 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_80 = tensor.extract %74[] : tensor<f32>
    %75 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%73, %extracted_80 : tensor<1024xf32>, f32) outs(%73 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %76 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%75 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_81 = tensor.extract %76[] : tensor<f32>
    %expanded_82 = tensor.expand_shape %75 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_83 = tensor.extract_slice %arg2[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_84 = tensor.extract_slice %inserted_slice_77[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_85 = tensor.reshape %extracted_slice_84(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %77 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_85 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %78 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_82, %extracted_81, %extracted_slice_83 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%77 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_86 = tensor.collapse_shape %78 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_87 = tensor.insert_slice %collapsed_86 into %inserted_slice_77[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_88 = tensor.extract_slice %arg0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_89 = tensor.extract_slice %arg1[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %79 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_89, %extracted_slice_88 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %80 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%79 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %81 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %80) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %82 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%81 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_90 = tensor.extract %82[] : tensor<f32>
    %83 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%81, %extracted_90 : tensor<1024xf32>, f32) outs(%81 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %84 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%83 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_91 = tensor.extract %84[] : tensor<f32>
    %expanded_92 = tensor.expand_shape %83 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_93 = tensor.extract_slice %arg2[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_94 = tensor.extract_slice %inserted_slice_87[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_95 = tensor.reshape %extracted_slice_94(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %85 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_95 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %86 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_92, %extracted_91, %extracted_slice_93 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%85 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_96 = tensor.collapse_shape %86 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_97 = tensor.insert_slice %collapsed_96 into %inserted_slice_87[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_98 = tensor.extract_slice %arg0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_99 = tensor.extract_slice %arg1[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %87 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_99, %extracted_slice_98 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %88 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%87 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %89 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %88) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %90 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%89 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_100 = tensor.extract %90[] : tensor<f32>
    %91 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%89, %extracted_100 : tensor<1024xf32>, f32) outs(%89 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %92 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%91 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_101 = tensor.extract %92[] : tensor<f32>
    %expanded_102 = tensor.expand_shape %91 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_103 = tensor.extract_slice %arg2[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_104 = tensor.extract_slice %inserted_slice_97[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_105 = tensor.reshape %extracted_slice_104(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %93 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_105 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %94 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_102, %extracted_101, %extracted_slice_103 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%93 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_106 = tensor.collapse_shape %94 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_107 = tensor.insert_slice %collapsed_106 into %inserted_slice_97[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_108 = tensor.extract_slice %arg0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_109 = tensor.extract_slice %arg1[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %95 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_109, %extracted_slice_108 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %96 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%95 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %97 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %96) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %98 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%97 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_110 = tensor.extract %98[] : tensor<f32>
    %99 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%97, %extracted_110 : tensor<1024xf32>, f32) outs(%97 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %100 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%99 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_111 = tensor.extract %100[] : tensor<f32>
    %expanded_112 = tensor.expand_shape %99 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_113 = tensor.extract_slice %arg2[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_114 = tensor.extract_slice %inserted_slice_107[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_115 = tensor.reshape %extracted_slice_114(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %101 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_115 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %102 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_112, %extracted_111, %extracted_slice_113 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%101 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_116 = tensor.collapse_shape %102 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_117 = tensor.insert_slice %collapsed_116 into %inserted_slice_107[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_118 = tensor.extract_slice %arg0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_119 = tensor.extract_slice %arg1[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %103 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_119, %extracted_slice_118 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %104 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%103 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %105 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %104) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %106 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%105 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_120 = tensor.extract %106[] : tensor<f32>
    %107 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%105, %extracted_120 : tensor<1024xf32>, f32) outs(%105 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %108 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%107 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_121 = tensor.extract %108[] : tensor<f32>
    %expanded_122 = tensor.expand_shape %107 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_123 = tensor.extract_slice %arg2[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_124 = tensor.extract_slice %inserted_slice_117[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_125 = tensor.reshape %extracted_slice_124(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %109 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_125 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %110 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_122, %extracted_121, %extracted_slice_123 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%109 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_126 = tensor.collapse_shape %110 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_127 = tensor.insert_slice %collapsed_126 into %inserted_slice_117[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_128 = tensor.extract_slice %arg0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_129 = tensor.extract_slice %arg1[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %111 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_129, %extracted_slice_128 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %112 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%111 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %113 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %112) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %114 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%113 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_130 = tensor.extract %114[] : tensor<f32>
    %115 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%113, %extracted_130 : tensor<1024xf32>, f32) outs(%113 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %116 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%115 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_131 = tensor.extract %116[] : tensor<f32>
    %expanded_132 = tensor.expand_shape %115 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_133 = tensor.extract_slice %arg2[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_134 = tensor.extract_slice %inserted_slice_127[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_135 = tensor.reshape %extracted_slice_134(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %117 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_135 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %118 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_132, %extracted_131, %extracted_slice_133 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%117 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_136 = tensor.collapse_shape %118 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_137 = tensor.insert_slice %collapsed_136 into %inserted_slice_127[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_138 = tensor.extract_slice %arg0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_139 = tensor.extract_slice %arg1[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %119 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_139, %extracted_slice_138 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %120 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%119 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %121 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %120) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %122 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%121 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_140 = tensor.extract %122[] : tensor<f32>
    %123 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%121, %extracted_140 : tensor<1024xf32>, f32) outs(%121 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %124 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%123 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_141 = tensor.extract %124[] : tensor<f32>
    %expanded_142 = tensor.expand_shape %123 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_143 = tensor.extract_slice %arg2[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_144 = tensor.extract_slice %inserted_slice_137[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_145 = tensor.reshape %extracted_slice_144(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %125 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_145 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %126 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_142, %extracted_141, %extracted_slice_143 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%125 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_146 = tensor.collapse_shape %126 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_147 = tensor.insert_slice %collapsed_146 into %inserted_slice_137[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_148 = tensor.extract_slice %arg0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_149 = tensor.extract_slice %arg1[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %127 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_149, %extracted_slice_148 : tensor<1024x48xf32>, tensor<48xf32>) outs(%3 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.mulf %in, %in_158 : f32
      %136 = arith.addf %out, %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %128 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%127 : tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.divf %in, %cst_1 : f32
      linalg.yield %135 : f32
    } -> tensor<1024xf32>
    %129 = scf.for %arg4 = %0 to %c1024 step %c1 iter_args(%arg5 = %128) -> (tensor<1024xf32>) {
      %inserted = tensor.insert %cst_3 into %arg5[%arg4] : tensor<1024xf32>
      scf.yield %inserted : tensor<1024xf32>
    }
    %130 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%129 : tensor<1024xf32>) outs(%8 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.maxnumf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_150 = tensor.extract %130[] : tensor<f32>
    %131 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%129, %extracted_150 : tensor<1024xf32>, f32) outs(%129 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_158: f32, %out: f32):
      %135 = arith.subf %in, %in_158 : f32
      %136 = math.exp %135 : f32
      linalg.yield %136 : f32
    } -> tensor<1024xf32>
    %132 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%131 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %135 = arith.addf %in, %out : f32
      linalg.yield %135 : f32
    } -> tensor<f32>
    %extracted_151 = tensor.extract %132[] : tensor<f32>
    %expanded_152 = tensor.expand_shape %131 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_153 = tensor.extract_slice %arg2[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_154 = tensor.extract_slice %inserted_slice_147[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_155 = tensor.reshape %extracted_slice_154(%cst) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %133 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_155 : tensor<1x48xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    } -> tensor<1x48xf32>
    %134 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_152, %extracted_151, %extracted_slice_153 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%133 : tensor<1x48xf32>) {
    ^bb0(%in: f32, %in_158: f32, %in_159: f32, %out: f32):
      %135 = arith.divf %in, %in_158 : f32
      %136 = arith.mulf %135, %in_159 : f32
      %137 = arith.addf %out, %136 : f32
      linalg.yield %137 : f32
    } -> tensor<1x48xf32>
    %collapsed_156 = tensor.collapse_shape %134 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_157 = tensor.insert_slice %collapsed_156 into %inserted_slice_147[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    return %inserted_slice_157 : tensor<768xf32>
  }
  func.func @rmsnorm(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<768xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %in : f32
      %9 = arith.addf %8, %out : f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %2[] : tensor<f32>
    %3 = arith.divf %extracted, %cst_1 : f32
    %4 = arith.addf %3, %cst_0 : f32
    %5 = math.rsqrt %4 : f32
    %6 = tensor.empty() : tensor<768xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %5, %arg1 : tensor<768xf32>, f32, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in, %in_2 : f32
      %9 = arith.mulf %8, %in_3 : f32
      linalg.yield %9 : f32
    } -> tensor<768xf32>
    return %7 : tensor<768xf32>
  }
  func.func @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.maxnumf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %2[] : tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg0, %extracted : tensor<1024xf32>, f32) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.subf %in, %in_2 : f32
      %8 = math.exp %7 : f32
      linalg.yield %8 : f32
    } -> tensor<1024xf32>
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%3 : tensor<1024xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.addf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<f32>
    %extracted_1 = tensor.extract %5[] : tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%3, %extracted_1 : tensor<1024xf32>, f32) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %7 = arith.divf %in, %in_2 : f32
      linalg.yield %7 : f32
    } -> tensor<1024xf32>
    return %6 : tensor<1024xf32>
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
