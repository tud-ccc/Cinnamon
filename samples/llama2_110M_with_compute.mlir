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
    %0 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %1 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %0, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %2 = tensor.empty() : tensor<768xf32>
    %3 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg5[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %1, %extracted_slice_9 : tensor<768xf32>, f32, tensor<768xf32>) outs(%2 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %964 = arith.mulf %in, %in_10 : f32
        %965 = arith.mulf %964, %in_11 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %4 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %5 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %6 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      %inserted_slice = tensor.insert_slice %964 into %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %inserted_slice : tensor<6x1024x768xf32>
    }
    %7:4 = cinm.compute on platform #cinm.host_platform -> f32, tensor<768xf32>, tensor<6x1024x768xf32>, index attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.index_cast %arg1 : index to i64
      %964 = arith.uitofp %963 : i64 to f32
      %965:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %4, %arg18 = %5) -> (tensor<768xf32>, tensor<768xf32>) {
        %967 = arith.remui %arg16, %c48 : index
        %968 = arith.index_cast %967 : index to i64
        %969 = arith.uitofp %968 : i64 to f32
        %970 = arith.divf %969, %cst_7 : f32
        %971 = math.powf %cst_8, %970 : f32
        %972 = arith.divf %cst_6, %971 : f32
        %973 = arith.mulf %964, %972 : f32
        %974 = math.cos %973 : f32
        %975 = math.sin %973 : f32
        %976 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_9 = tensor.extract %arg17[%976] : tensor<768xf32>
        %977 = arith.mulf %extracted, %974 : f32
        %978 = arith.mulf %extracted_9, %975 : f32
        %979 = arith.subf %977, %978 : f32
        %inserted = tensor.insert %979 into %arg17[%arg16] : tensor<768xf32>
        %980 = arith.mulf %extracted, %975 : f32
        %981 = arith.mulf %extracted_9, %974 : f32
        %982 = arith.addf %980, %981 : f32
        %inserted_10 = tensor.insert %982 into %inserted[%976] : tensor<768xf32>
        %983 = bufferization.materialize_in_destination %inserted_10 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %984 = arith.cmpi ult, %arg16, %c768 : index
        %985 = scf.if %984 -> (tensor<768xf32>) {
          %extracted_11 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_12 = tensor.extract %arg18[%976] : tensor<768xf32>
          %986 = arith.mulf %extracted_11, %974 : f32
          %987 = arith.mulf %extracted_12, %975 : f32
          %988 = arith.subf %986, %987 : f32
          %inserted_13 = tensor.insert %988 into %arg18[%arg16] : tensor<768xf32>
          %989 = arith.mulf %extracted_11, %975 : f32
          %990 = arith.mulf %extracted_12, %974 : f32
          %991 = arith.addf %989, %990 : f32
          %inserted_14 = tensor.insert %991 into %inserted_13[%976] : tensor<768xf32>
          %992 = bufferization.materialize_in_destination %inserted_14 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %992 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %983, %985 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %965#1 into %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %966 = arith.addi %arg1, %c1 : index
      cinm.yield %964, %965#0, %inserted_slice, %966 : f32, tensor<768xf32>, tensor<6x1024x768xf32>, index
    }
    %8 = tensor.empty() : tensor<1024xf32>
    %9 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %10 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%9 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %11 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %10) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %12 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%11 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %13 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%11, %12 : tensor<1024xf32>, f32) outs(%11 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %14 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%13 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %15:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %13 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %2[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %16 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%15#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15#0, %14, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %17 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %16 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %3[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %18 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %19 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%18 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %20 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %19) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %21 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%20 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %22 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%20, %21 : tensor<1024xf32>, f32) outs(%20 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %23 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%22 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %24:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %22 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %17[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %25 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%24#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%24#0, %23, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %26 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %25 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %17[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %27 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %28 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%27 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %29 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %28) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %30 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%29 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %31 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%29, %30 : tensor<1024xf32>, f32) outs(%29 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %32 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%31 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %33:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %31 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %26[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %34 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%33#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%33#0, %32, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %35 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %34 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %26[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %36 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %37 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%36 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %38 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %37) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %39 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%38 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %40 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%38, %39 : tensor<1024xf32>, f32) outs(%38 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %41 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%40 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %42:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %40 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %35[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %43 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%42#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%42#0, %41, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %44 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %43 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %35[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %45 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %46 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%45 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %47 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %46) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %48 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%47 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %49 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%47, %48 : tensor<1024xf32>, f32) outs(%47 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %50 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%49 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %51:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %49 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %44[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %52 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%51#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51#0, %50, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %53 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %52 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %44[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %54 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %55 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%54 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %56 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %55) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %57 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%56 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %58 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%56, %57 : tensor<1024xf32>, f32) outs(%56 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %59 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%58 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %60:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %58 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %53[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %61 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%60#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%60#0, %59, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %62 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %61 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %53[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %63 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %64 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %65 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %64) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %66 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%65 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %67 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%65, %66 : tensor<1024xf32>, f32) outs(%65 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %68 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%67 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %69:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %67 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %62[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %70 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%69#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69#0, %68, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %71 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %70 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %62[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %72 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %73 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%72 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %74 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %73) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %75 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%74 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %76 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%74, %75 : tensor<1024xf32>, f32) outs(%74 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %77 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%76 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %78:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %76 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %71[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %79 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%78#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78#0, %77, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %80 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %79 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %71[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %81 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %82 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%81 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %83 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %82) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %84 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%83 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %85 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%83, %84 : tensor<1024xf32>, f32) outs(%83 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %86 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%85 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %87:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %85 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %80[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %88 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%87#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87#0, %86, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %89 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %88 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %80[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %90 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %91 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%90 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %92 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %91) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %93 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%92 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %94 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%92, %93 : tensor<1024xf32>, f32) outs(%92 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %95 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%94 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %96:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %94 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %89[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %97 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%96#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%96#0, %95, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %98 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %97 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %89[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %99 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %100 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%99 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %101 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %100) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %102 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%101 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %103 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%101, %102 : tensor<1024xf32>, f32) outs(%101 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %104 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%103 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %105:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %103 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %98[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %106 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%105#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%105#0, %104, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %107 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %106 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %98[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %108 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %109 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%108 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %110 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %109) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %111 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%110 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %112 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%110, %111 : tensor<1024xf32>, f32) outs(%110 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %113 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%112 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %114:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %112 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %107[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %115 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%114#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114#0, %113, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %116 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %115 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %107[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %117 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %118 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%117 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %119 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %118) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %120 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%119 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %121 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%119, %120 : tensor<1024xf32>, f32) outs(%119 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %122 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%121 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %123:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %121 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %116[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %124 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%123#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%123#0, %122, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %125 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %124 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %116[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %126 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %127 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%126 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %128 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %127) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %129 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%128 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %130 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%128, %129 : tensor<1024xf32>, f32) outs(%128 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %131 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%130 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %132:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %130 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %125[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %133 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%132#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%132#0, %131, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %134 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %133 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %125[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %135 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %136 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%135 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %137 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %136) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %138 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%137 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %139 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%137, %138 : tensor<1024xf32>, f32) outs(%137 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %140 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%139 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %141:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %139 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %134[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %142 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%141#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%141#0, %140, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %143 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %142 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %134[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %144 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %7#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %7#1[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %145 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%144 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %146 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %145) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %147 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%146 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %148 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%146, %147 : tensor<1024xf32>, f32) outs(%146 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %149 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%148 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %150:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %148 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %143[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %151 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %6[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%150#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%150#0, %149, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %152 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %151 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %143[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %963 = bufferization.materialize_in_destination %inserted_slice in %3 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %153 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %152 : tensor<768x768xf32>, tensor<768xf32>) outs(%extracted_slice_9 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_10 : f32
        %965 = arith.addf %out, %964 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %154 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%153 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %155 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %154, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %156 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg13[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%153, %155, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%152 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %157 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = bufferization.materialize_in_destination %156 in %152 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %158 = tensor.empty() : tensor<2048xf32>
    %159 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %157 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %160 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %157 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %161:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %963:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %160 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%159, %153 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %964 = arith.negf %out : f32
        %965 = math.exp %964 : f32
        %966 = arith.addf %965, %cst_6 : f32
        %967 = arith.divf %cst_6, %966 : f32
        %968 = arith.mulf %out, %967 : f32
        %969 = arith.mulf %968, %in_9 : f32
        %970 = arith.mulf %in, %969 : f32
        %971 = arith.addf %out_10, %970 : f32
        linalg.yield %969, %971 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %963#0, %963#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %162 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%161#1 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %163 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %162, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %164 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg5[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%161#1, %163, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%2 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %165 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %164 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %166 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %7#2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %164 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %167 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %6[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %164 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      %inserted_slice = tensor.insert_slice %964 into %6[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %inserted_slice : tensor<6x1024x768xf32>
    }
    %168:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %165, %arg18 = %166) -> (tensor<768xf32>, tensor<768xf32>) {
        %964 = arith.remui %arg16, %c48 : index
        %965 = arith.index_cast %964 : index to i64
        %966 = arith.uitofp %965 : i64 to f32
        %967 = arith.divf %966, %cst_7 : f32
        %968 = math.powf %cst_8, %967 : f32
        %969 = arith.divf %cst_6, %968 : f32
        %970 = arith.mulf %7#0, %969 : f32
        %971 = math.cos %970 : f32
        %972 = math.sin %970 : f32
        %973 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_9 = tensor.extract %arg17[%973] : tensor<768xf32>
        %974 = arith.mulf %extracted, %971 : f32
        %975 = arith.mulf %extracted_9, %972 : f32
        %976 = arith.subf %974, %975 : f32
        %inserted = tensor.insert %976 into %arg17[%arg16] : tensor<768xf32>
        %977 = arith.mulf %extracted, %972 : f32
        %978 = arith.mulf %extracted_9, %971 : f32
        %979 = arith.addf %977, %978 : f32
        %inserted_10 = tensor.insert %979 into %inserted[%973] : tensor<768xf32>
        %980 = bufferization.materialize_in_destination %inserted_10 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %981 = arith.cmpi ult, %arg16, %c768 : index
        %982 = scf.if %981 -> (tensor<768xf32>) {
          %extracted_11 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_12 = tensor.extract %arg18[%973] : tensor<768xf32>
          %983 = arith.mulf %extracted_11, %971 : f32
          %984 = arith.mulf %extracted_12, %972 : f32
          %985 = arith.subf %983, %984 : f32
          %inserted_13 = tensor.insert %985 into %arg18[%arg16] : tensor<768xf32>
          %986 = arith.mulf %extracted_11, %972 : f32
          %987 = arith.mulf %extracted_12, %971 : f32
          %988 = arith.addf %986, %987 : f32
          %inserted_14 = tensor.insert %988 into %inserted_13[%973] : tensor<768xf32>
          %989 = bufferization.materialize_in_destination %inserted_14 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %989 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %980, %982 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %963#1 into %7#2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %963#0, %inserted_slice : tensor<768xf32>, tensor<6x1024x768xf32>
    }
    %169 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %170 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%169 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %171 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %170) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %172 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%171 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %173 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%171, %172 : tensor<1024xf32>, f32) outs(%171 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %174 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%173 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %175 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %173 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      cinm.yield %expanded : tensor<1x1024xf32>
    }
    %176 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%15#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%175, %174, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %177 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %176 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %164[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %178 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %179 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%178 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %180 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %179) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %181 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%180 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %182 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%180, %181 : tensor<1024xf32>, f32) outs(%180 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %183 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%182 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %184:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %182 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %177[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %185 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%184#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%184#0, %183, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %186 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %185 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %177[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %187 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %188 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%187 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %189 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %188) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %190 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%189 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %191 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%189, %190 : tensor<1024xf32>, f32) outs(%189 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %192 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%191 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %193:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %191 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %186[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %194 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%193#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%193#0, %192, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %195 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %194 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %186[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %196 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %197 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%196 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %198 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %197) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %199 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%198 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %200 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%198, %199 : tensor<1024xf32>, f32) outs(%198 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %201 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%200 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %202:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %200 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %195[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %203 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%202#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%202#0, %201, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %204 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %203 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %195[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %205 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %206 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%205 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %207 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %206) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %208 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%207 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %209 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%207, %208 : tensor<1024xf32>, f32) outs(%207 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %210 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%209 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %211:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %209 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %204[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %212 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%211#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%211#0, %210, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %213 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %212 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %204[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %214 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %215 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%214 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %216 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %215) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %217 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%216 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %218 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%216, %217 : tensor<1024xf32>, f32) outs(%216 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %219 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%218 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %220:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %218 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %213[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %221 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%220#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%220#0, %219, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %222 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %221 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %213[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %223 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %224 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%223 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %225 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %224) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %226 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%225 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %227 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%225, %226 : tensor<1024xf32>, f32) outs(%225 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %228 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%227 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %229:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %227 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %222[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %230 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%229#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%229#0, %228, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %231 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %230 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %222[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %232 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %233 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%232 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %234 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %233) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %235 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%234 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %236 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%234, %235 : tensor<1024xf32>, f32) outs(%234 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %237 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%236 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %238:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %236 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %231[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %239 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%238#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%238#0, %237, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %240 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %239 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %231[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %241 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %242 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%241 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %243 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %242) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %244 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%243 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %245 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%243, %244 : tensor<1024xf32>, f32) outs(%243 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %246 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%245 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %247:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %245 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %240[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %248 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%247#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%247#0, %246, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %249 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %248 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %240[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %250 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %251 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%250 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %252 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %251) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %253 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%252 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %254 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%252, %253 : tensor<1024xf32>, f32) outs(%252 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %255 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%254 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %256:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %254 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %249[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %257 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%256#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%256#0, %255, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %258 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %257 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %249[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %259 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %260 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%259 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %261 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %260) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %262 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%261 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %263 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%261, %262 : tensor<1024xf32>, f32) outs(%261 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %264 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%263 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %265:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %263 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %258[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %266 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%265#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%265#0, %264, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %267 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %266 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %258[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %268 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %269 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%268 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %270 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %269) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %271 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%270 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %272 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%270, %271 : tensor<1024xf32>, f32) outs(%270 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %273 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%272 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %274:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %272 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %267[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %275 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%274#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%274#0, %273, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %276 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %275 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %267[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %277 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %278 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%277 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %279 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %278) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %280 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%279 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %281 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%279, %280 : tensor<1024xf32>, f32) outs(%279 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %282 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%281 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %283:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %281 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %276[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %284 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%283#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%283#0, %282, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %285 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %284 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %276[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %286 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %287 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%286 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %288 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %287) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %289 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%288 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %290 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%288, %289 : tensor<1024xf32>, f32) outs(%288 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %291 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%290 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %292:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %290 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %285[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %293 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%292#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%292#0, %291, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %294 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %293 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %285[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %295 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %296 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%295 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %297 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %296) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %298 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%297 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %299 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%297, %298 : tensor<1024xf32>, f32) outs(%297 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %300 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%299 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %301:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %299 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %294[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %302 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%301#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%301#0, %300, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %303 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %302 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %294[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %304 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %168#1[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %168#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %305 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%304 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %306 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %305) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %307 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%306 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %308 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%306, %307 : tensor<1024xf32>, f32) outs(%306 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %309 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%308 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %310:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %308 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %303[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %311 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %167[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%310#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%310#0, %309, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %312 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %311 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %303[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %963 = bufferization.materialize_in_destination %inserted_slice in %164 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %313 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %312 : tensor<768x768xf32>, tensor<768xf32>) outs(%161#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.addf %out, %964 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %314 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%313 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %315 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %314, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %316 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg13[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%313, %315, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%312 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %317 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = bufferization.materialize_in_destination %316 in %312 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %318 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %317 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %319 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %317 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %320:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %963:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %319 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%318, %313 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %964 = arith.negf %out : f32
        %965 = math.exp %964 : f32
        %966 = arith.addf %965, %cst_6 : f32
        %967 = arith.divf %cst_6, %966 : f32
        %968 = arith.mulf %out, %967 : f32
        %969 = arith.mulf %968, %in_9 : f32
        %970 = arith.mulf %in, %969 : f32
        %971 = arith.addf %out_10, %970 : f32
        linalg.yield %969, %971 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %963#0, %963#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %321 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%320#1 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %322 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %321, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %323 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg5[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%320#1, %322, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%2 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %324 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %323 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %325 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %168#1[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %323 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %326 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %167[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %323 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      %inserted_slice = tensor.insert_slice %964 into %167[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %inserted_slice : tensor<6x1024x768xf32>
    }
    %327:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %324, %arg18 = %325) -> (tensor<768xf32>, tensor<768xf32>) {
        %964 = arith.remui %arg16, %c48 : index
        %965 = arith.index_cast %964 : index to i64
        %966 = arith.uitofp %965 : i64 to f32
        %967 = arith.divf %966, %cst_7 : f32
        %968 = math.powf %cst_8, %967 : f32
        %969 = arith.divf %cst_6, %968 : f32
        %970 = arith.mulf %7#0, %969 : f32
        %971 = math.cos %970 : f32
        %972 = math.sin %970 : f32
        %973 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_9 = tensor.extract %arg17[%973] : tensor<768xf32>
        %974 = arith.mulf %extracted, %971 : f32
        %975 = arith.mulf %extracted_9, %972 : f32
        %976 = arith.subf %974, %975 : f32
        %inserted = tensor.insert %976 into %arg17[%arg16] : tensor<768xf32>
        %977 = arith.mulf %extracted, %972 : f32
        %978 = arith.mulf %extracted_9, %971 : f32
        %979 = arith.addf %977, %978 : f32
        %inserted_10 = tensor.insert %979 into %inserted[%973] : tensor<768xf32>
        %980 = bufferization.materialize_in_destination %inserted_10 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %981 = arith.cmpi ult, %arg16, %c768 : index
        %982 = scf.if %981 -> (tensor<768xf32>) {
          %extracted_11 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_12 = tensor.extract %arg18[%973] : tensor<768xf32>
          %983 = arith.mulf %extracted_11, %971 : f32
          %984 = arith.mulf %extracted_12, %972 : f32
          %985 = arith.subf %983, %984 : f32
          %inserted_13 = tensor.insert %985 into %arg18[%arg16] : tensor<768xf32>
          %986 = arith.mulf %extracted_11, %972 : f32
          %987 = arith.mulf %extracted_12, %971 : f32
          %988 = arith.addf %986, %987 : f32
          %inserted_14 = tensor.insert %988 into %inserted_13[%973] : tensor<768xf32>
          %989 = bufferization.materialize_in_destination %inserted_14 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %989 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %980, %982 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %963#1 into %168#1[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %963#0, %inserted_slice : tensor<768xf32>, tensor<6x1024x768xf32>
    }
    %328 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %329 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%328 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %330 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %329) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %331 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%330 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %332 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%330, %331 : tensor<1024xf32>, f32) outs(%330 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %333 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%332 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %334 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %332 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      cinm.yield %expanded : tensor<1x1024xf32>
    }
    %335 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%15#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%334, %333, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %336 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %335 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %323[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %337 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %338 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%337 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %339 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %338) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %340 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%339 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %341 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%339, %340 : tensor<1024xf32>, f32) outs(%339 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %342 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%341 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %343:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %341 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %336[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %344 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%343#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%343#0, %342, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %345 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %344 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %336[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %346 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %347 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%346 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %348 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %347) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %349 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%348 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %350 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%348, %349 : tensor<1024xf32>, f32) outs(%348 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %351 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%350 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %352:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %350 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %345[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %353 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%352#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%352#0, %351, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %354 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %353 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %345[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %355 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %356 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%355 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %357 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %356) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %358 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%357 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %359 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%357, %358 : tensor<1024xf32>, f32) outs(%357 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %360 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%359 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %361:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %359 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %354[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %362 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%361#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%361#0, %360, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %363 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %362 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %354[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %364 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %365 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%364 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %366 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %365) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %367 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%366 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %368 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%366, %367 : tensor<1024xf32>, f32) outs(%366 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %369 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%368 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %370:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %368 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %363[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %371 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%370#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%370#0, %369, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %372 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %371 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %363[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %373 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %374 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%373 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %375 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %374) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %376 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%375 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %377 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%375, %376 : tensor<1024xf32>, f32) outs(%375 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %378 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%377 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %379:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %377 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %372[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %380 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%379#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%379#0, %378, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %381 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %380 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %372[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %382 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %383 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%382 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %384 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %383) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %385 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%384 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %386 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%384, %385 : tensor<1024xf32>, f32) outs(%384 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %387 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%386 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %388:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %386 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %381[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %389 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%388#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%388#0, %387, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %390 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %389 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %381[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %391 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %392 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%391 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %393 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %392) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %394 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%393 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %395 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%393, %394 : tensor<1024xf32>, f32) outs(%393 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %396 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%395 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %397:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %395 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %390[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %398 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%397#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%397#0, %396, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %399 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %398 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %390[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %400 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %401 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%400 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %402 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %401) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %403 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%402 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %404 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%402, %403 : tensor<1024xf32>, f32) outs(%402 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %405 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%404 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %406:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %404 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %399[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %407 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%406#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%406#0, %405, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %408 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %407 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %399[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %409 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %410 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%409 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %411 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %410) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %412 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%411 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %413 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%411, %412 : tensor<1024xf32>, f32) outs(%411 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %414 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%413 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %415:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %413 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %408[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %416 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%415#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%415#0, %414, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %417 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %416 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %408[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %418 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %419 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%418 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %420 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %419) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %421 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%420 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %422 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%420, %421 : tensor<1024xf32>, f32) outs(%420 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %423 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%422 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %424:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %422 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %417[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %425 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%424#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%424#0, %423, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %426 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %425 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %417[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %427 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %428 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%427 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %429 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %428) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %430 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%429 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %431 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%429, %430 : tensor<1024xf32>, f32) outs(%429 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %432 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%431 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %433:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %431 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %426[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %434 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%433#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%433#0, %432, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %435 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %434 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %426[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %436 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %437 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%436 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %438 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %437) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %439 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%438 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %440 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%438, %439 : tensor<1024xf32>, f32) outs(%438 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %441 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%440 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %442:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %440 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %435[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %443 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%442#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%442#0, %441, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %444 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %443 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %435[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %445 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %446 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%445 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %447 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %446) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %448 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%447 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %449 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%447, %448 : tensor<1024xf32>, f32) outs(%447 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %450 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%449 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %451:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %449 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %444[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %452 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%451#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%451#0, %450, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %453 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %452 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %444[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %454 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %455 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%454 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %456 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %455) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %457 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%456 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %458 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%456, %457 : tensor<1024xf32>, f32) outs(%456 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %459 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%458 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %460:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %458 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %453[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %461 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%460#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%460#0, %459, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %462 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %461 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %453[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %463 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %327#1[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %327#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %464 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%463 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %465 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %464) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %466 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%465 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %467 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%465, %466 : tensor<1024xf32>, f32) outs(%465 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %468 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%467 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %469:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %467 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %462[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %470 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %326[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%469#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%469#0, %468, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %471 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %470 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %462[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %963 = bufferization.materialize_in_destination %inserted_slice in %323 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %472 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %471 : tensor<768x768xf32>, tensor<768xf32>) outs(%320#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.addf %out, %964 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %473 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%472 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %474 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %473, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %475 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg13[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%472, %474, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%471 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %476 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = bufferization.materialize_in_destination %475 in %471 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %477 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %476 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %478 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %476 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %479:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %963:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %478 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%477, %472 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %964 = arith.negf %out : f32
        %965 = math.exp %964 : f32
        %966 = arith.addf %965, %cst_6 : f32
        %967 = arith.divf %cst_6, %966 : f32
        %968 = arith.mulf %out, %967 : f32
        %969 = arith.mulf %968, %in_9 : f32
        %970 = arith.mulf %in, %969 : f32
        %971 = arith.addf %out_10, %970 : f32
        linalg.yield %969, %971 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %963#0, %963#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %480 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%479#1 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %481 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %480, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %482 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg5[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%479#1, %481, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%2 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %483 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %482 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %484 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %327#1[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %482 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %485 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %326[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %482 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      %inserted_slice = tensor.insert_slice %964 into %326[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %inserted_slice : tensor<6x1024x768xf32>
    }
    %486:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %483, %arg18 = %484) -> (tensor<768xf32>, tensor<768xf32>) {
        %964 = arith.remui %arg16, %c48 : index
        %965 = arith.index_cast %964 : index to i64
        %966 = arith.uitofp %965 : i64 to f32
        %967 = arith.divf %966, %cst_7 : f32
        %968 = math.powf %cst_8, %967 : f32
        %969 = arith.divf %cst_6, %968 : f32
        %970 = arith.mulf %7#0, %969 : f32
        %971 = math.cos %970 : f32
        %972 = math.sin %970 : f32
        %973 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_9 = tensor.extract %arg17[%973] : tensor<768xf32>
        %974 = arith.mulf %extracted, %971 : f32
        %975 = arith.mulf %extracted_9, %972 : f32
        %976 = arith.subf %974, %975 : f32
        %inserted = tensor.insert %976 into %arg17[%arg16] : tensor<768xf32>
        %977 = arith.mulf %extracted, %972 : f32
        %978 = arith.mulf %extracted_9, %971 : f32
        %979 = arith.addf %977, %978 : f32
        %inserted_10 = tensor.insert %979 into %inserted[%973] : tensor<768xf32>
        %980 = bufferization.materialize_in_destination %inserted_10 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %981 = arith.cmpi ult, %arg16, %c768 : index
        %982 = scf.if %981 -> (tensor<768xf32>) {
          %extracted_11 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_12 = tensor.extract %arg18[%973] : tensor<768xf32>
          %983 = arith.mulf %extracted_11, %971 : f32
          %984 = arith.mulf %extracted_12, %972 : f32
          %985 = arith.subf %983, %984 : f32
          %inserted_13 = tensor.insert %985 into %arg18[%arg16] : tensor<768xf32>
          %986 = arith.mulf %extracted_11, %972 : f32
          %987 = arith.mulf %extracted_12, %971 : f32
          %988 = arith.addf %986, %987 : f32
          %inserted_14 = tensor.insert %988 into %inserted_13[%973] : tensor<768xf32>
          %989 = bufferization.materialize_in_destination %inserted_14 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %989 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %980, %982 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %963#1 into %327#1[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %963#0, %inserted_slice : tensor<768xf32>, tensor<6x1024x768xf32>
    }
    %487 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %488 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%487 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %489 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %488) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %490 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%489 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %491 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%489, %490 : tensor<1024xf32>, f32) outs(%489 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %492 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%491 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %493 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %491 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      cinm.yield %expanded : tensor<1x1024xf32>
    }
    %494 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%15#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%493, %492, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %495 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %494 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %482[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %496 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %497 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%496 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %498 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %497) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %499 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%498 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %500 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%498, %499 : tensor<1024xf32>, f32) outs(%498 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %501 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%500 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %502:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %500 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %495[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %503 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%502#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%502#0, %501, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %504 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %503 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %495[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %505 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %506 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%505 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %507 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %506) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %508 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%507 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %509 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%507, %508 : tensor<1024xf32>, f32) outs(%507 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %510 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%509 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %511:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %509 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %504[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %512 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%511#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%511#0, %510, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %513 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %512 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %504[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %514 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %515 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%514 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %516 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %515) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %517 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%516 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %518 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%516, %517 : tensor<1024xf32>, f32) outs(%516 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %519 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%518 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %520:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %518 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %513[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %521 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%520#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%520#0, %519, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %522 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %521 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %513[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %523 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %524 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%523 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %525 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %524) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %526 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%525 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %527 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%525, %526 : tensor<1024xf32>, f32) outs(%525 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %528 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%527 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %529:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %527 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %522[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %530 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%529#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%529#0, %528, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %531 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %530 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %522[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %532 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %533 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%532 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %534 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %533) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %535 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%534 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %536 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%534, %535 : tensor<1024xf32>, f32) outs(%534 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %537 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%536 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %538:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %536 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %531[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %539 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%538#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%538#0, %537, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %540 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %539 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %531[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %541 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %542 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%541 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %543 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %542) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %544 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%543 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %545 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%543, %544 : tensor<1024xf32>, f32) outs(%543 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %546 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%545 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %547:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %545 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %540[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %548 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%547#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%547#0, %546, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %549 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %548 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %540[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %550 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %551 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%550 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %552 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %551) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %553 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%552 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %554 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%552, %553 : tensor<1024xf32>, f32) outs(%552 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %555 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%554 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %556:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %554 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %549[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %557 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%556#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%556#0, %555, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %558 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %557 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %549[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %559 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %560 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%559 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %561 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %560) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %562 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%561 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %563 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%561, %562 : tensor<1024xf32>, f32) outs(%561 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %564 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%563 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %565:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %563 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %558[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %566 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%565#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%565#0, %564, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %567 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %566 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %558[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %568 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %569 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%568 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %570 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %569) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %571 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%570 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %572 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%570, %571 : tensor<1024xf32>, f32) outs(%570 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %573 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%572 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %574:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %572 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %567[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %575 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%574#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%574#0, %573, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %576 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %575 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %567[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %577 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %578 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%577 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %579 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %578) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %580 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%579 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %581 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%579, %580 : tensor<1024xf32>, f32) outs(%579 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %582 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%581 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %583:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %581 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %576[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %584 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%583#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%583#0, %582, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %585 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %584 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %576[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %586 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %587 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%586 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %588 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %587) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %589 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%588 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %590 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%588, %589 : tensor<1024xf32>, f32) outs(%588 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %591 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%590 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %592:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %590 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %585[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %593 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%592#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%592#0, %591, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %594 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %593 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %585[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %595 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %596 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%595 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %597 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %596) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %598 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%597 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %599 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%597, %598 : tensor<1024xf32>, f32) outs(%597 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %600 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%599 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %601:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %599 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %594[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %602 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%601#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%601#0, %600, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %603 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %602 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %594[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %604 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %605 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%604 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %606 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %605) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %607 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%606 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %608 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%606, %607 : tensor<1024xf32>, f32) outs(%606 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %609 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%608 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %610:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %608 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %603[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %611 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%610#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%610#0, %609, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %612 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %611 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %603[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %613 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %614 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%613 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %615 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %614) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %616 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%615 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %617 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%615, %616 : tensor<1024xf32>, f32) outs(%615 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %618 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%617 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %619:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %617 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %612[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %620 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%619#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%619#0, %618, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %621 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %620 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %612[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %622 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %486#1[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %486#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %623 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%622 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %624 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %623) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %625 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%624 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %626 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%624, %625 : tensor<1024xf32>, f32) outs(%624 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %627 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%626 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %628:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %626 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %621[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %629 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %485[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%628#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%628#0, %627, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %630 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %629 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %621[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %963 = bufferization.materialize_in_destination %inserted_slice in %482 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %631 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %630 : tensor<768x768xf32>, tensor<768xf32>) outs(%479#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.addf %out, %964 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %632 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%631 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %633 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %632, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %634 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg13[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%631, %633, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%630 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %635 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = bufferization.materialize_in_destination %634 in %630 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %636 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %635 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %637 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %635 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %638:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %963:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %637 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%636, %631 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %964 = arith.negf %out : f32
        %965 = math.exp %964 : f32
        %966 = arith.addf %965, %cst_6 : f32
        %967 = arith.divf %cst_6, %966 : f32
        %968 = arith.mulf %out, %967 : f32
        %969 = arith.mulf %968, %in_9 : f32
        %970 = arith.mulf %in, %969 : f32
        %971 = arith.addf %out_10, %970 : f32
        linalg.yield %969, %971 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %963#0, %963#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %639 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%638#1 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %640 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %639, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %641 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg5[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%638#1, %640, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%2 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %642 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %641 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %643 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %486#1[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %641 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %644 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %485[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %641 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      %inserted_slice = tensor.insert_slice %964 into %485[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %inserted_slice : tensor<6x1024x768xf32>
    }
    %645:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %642, %arg18 = %643) -> (tensor<768xf32>, tensor<768xf32>) {
        %964 = arith.remui %arg16, %c48 : index
        %965 = arith.index_cast %964 : index to i64
        %966 = arith.uitofp %965 : i64 to f32
        %967 = arith.divf %966, %cst_7 : f32
        %968 = math.powf %cst_8, %967 : f32
        %969 = arith.divf %cst_6, %968 : f32
        %970 = arith.mulf %7#0, %969 : f32
        %971 = math.cos %970 : f32
        %972 = math.sin %970 : f32
        %973 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_9 = tensor.extract %arg17[%973] : tensor<768xf32>
        %974 = arith.mulf %extracted, %971 : f32
        %975 = arith.mulf %extracted_9, %972 : f32
        %976 = arith.subf %974, %975 : f32
        %inserted = tensor.insert %976 into %arg17[%arg16] : tensor<768xf32>
        %977 = arith.mulf %extracted, %972 : f32
        %978 = arith.mulf %extracted_9, %971 : f32
        %979 = arith.addf %977, %978 : f32
        %inserted_10 = tensor.insert %979 into %inserted[%973] : tensor<768xf32>
        %980 = bufferization.materialize_in_destination %inserted_10 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %981 = arith.cmpi ult, %arg16, %c768 : index
        %982 = scf.if %981 -> (tensor<768xf32>) {
          %extracted_11 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_12 = tensor.extract %arg18[%973] : tensor<768xf32>
          %983 = arith.mulf %extracted_11, %971 : f32
          %984 = arith.mulf %extracted_12, %972 : f32
          %985 = arith.subf %983, %984 : f32
          %inserted_13 = tensor.insert %985 into %arg18[%arg16] : tensor<768xf32>
          %986 = arith.mulf %extracted_11, %972 : f32
          %987 = arith.mulf %extracted_12, %971 : f32
          %988 = arith.addf %986, %987 : f32
          %inserted_14 = tensor.insert %988 into %inserted_13[%973] : tensor<768xf32>
          %989 = bufferization.materialize_in_destination %inserted_14 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %989 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %980, %982 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %963#1 into %486#1[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %963#0, %inserted_slice : tensor<768xf32>, tensor<6x1024x768xf32>
    }
    %646 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %647 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%646 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %648 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %647) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %649 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%648 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %650 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%648, %649 : tensor<1024xf32>, f32) outs(%648 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %651 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%650 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %652 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %650 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      cinm.yield %expanded : tensor<1x1024xf32>
    }
    %653 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%15#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%652, %651, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %654 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %653 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %641[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %655 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %656 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%655 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %657 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %656) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %658 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%657 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %659 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%657, %658 : tensor<1024xf32>, f32) outs(%657 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %660 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%659 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %661:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %659 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %654[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %662 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%661#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%661#0, %660, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %663 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %662 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %654[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %664 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %665 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%664 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %666 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %665) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %667 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%666 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %668 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%666, %667 : tensor<1024xf32>, f32) outs(%666 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %669 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%668 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %670:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %668 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %663[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %671 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%670#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%670#0, %669, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %672 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %671 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %663[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %673 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %674 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%673 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %675 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %674) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %676 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%675 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %677 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%675, %676 : tensor<1024xf32>, f32) outs(%675 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %678 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%677 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %679:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %677 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %672[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %680 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%679#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%679#0, %678, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %681 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %680 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %672[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %682 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %683 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%682 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %684 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %683) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %685 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%684 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %686 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%684, %685 : tensor<1024xf32>, f32) outs(%684 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %687 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%686 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %688:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %686 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %681[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %689 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%688#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%688#0, %687, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %690 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %689 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %681[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %691 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %692 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%691 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %693 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %692) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %694 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%693 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %695 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%693, %694 : tensor<1024xf32>, f32) outs(%693 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %696 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%695 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %697:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %695 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %690[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %698 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%697#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%697#0, %696, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %699 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %698 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %690[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %700 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %701 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%700 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %702 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %701) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %703 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%702 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %704 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%702, %703 : tensor<1024xf32>, f32) outs(%702 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %705 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%704 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %706:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %704 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %699[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %707 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%706#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%706#0, %705, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %708 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %707 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %699[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %709 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %710 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%709 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %711 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %710) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %712 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%711 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %713 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%711, %712 : tensor<1024xf32>, f32) outs(%711 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %714 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%713 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %715:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %713 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %708[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %716 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%715#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%715#0, %714, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %717 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %716 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %708[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %718 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %719 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%718 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %720 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %719) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %721 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%720 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %722 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%720, %721 : tensor<1024xf32>, f32) outs(%720 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %723 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%722 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %724:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %722 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %717[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %725 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%724#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%724#0, %723, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %726 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %725 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %717[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %727 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %728 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%727 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %729 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %728) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %730 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%729 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %731 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%729, %730 : tensor<1024xf32>, f32) outs(%729 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %732 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%731 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %733:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %731 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %726[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %734 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%733#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%733#0, %732, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %735 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %734 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %726[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %736 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %737 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%736 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %738 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %737) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %739 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%738 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %740 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%738, %739 : tensor<1024xf32>, f32) outs(%738 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %741 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%740 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %742:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %740 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %735[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %743 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%742#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%742#0, %741, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %744 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %743 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %735[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %745 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %746 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%745 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %747 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %746) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %748 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%747 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %749 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%747, %748 : tensor<1024xf32>, f32) outs(%747 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %750 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%749 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %751:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %749 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %744[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %752 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%751#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%751#0, %750, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %753 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %752 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %744[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %754 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %755 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%754 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %756 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %755) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %757 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%756 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %758 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%756, %757 : tensor<1024xf32>, f32) outs(%756 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %759 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%758 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %760:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %758 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %753[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %761 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%760#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%760#0, %759, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %762 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %761 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %753[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %763 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %764 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%763 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %765 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %764) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %766 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%765 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %767 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%765, %766 : tensor<1024xf32>, f32) outs(%765 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %768 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%767 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %769:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %767 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %762[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %770 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%769#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%769#0, %768, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %771 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %770 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %762[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %772 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %773 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%772 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %774 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %773) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %775 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%774 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %776 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%774, %775 : tensor<1024xf32>, f32) outs(%774 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %777 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%776 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %778:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %776 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %771[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %779 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%778#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%778#0, %777, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %780 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %779 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %771[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %781 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %645#1[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %645#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %782 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%781 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %783 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %782) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %784 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%783 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %785 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%783, %784 : tensor<1024xf32>, f32) outs(%783 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %786 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%785 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %787:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %785 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %780[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %788 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %644[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%787#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%787#0, %786, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %789 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %788 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %780[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %963 = bufferization.materialize_in_destination %inserted_slice in %641 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %790 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %789 : tensor<768x768xf32>, tensor<768xf32>) outs(%638#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.addf %out, %964 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %791 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%790 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %792 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %791, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %793 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg13[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%790, %792, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%789 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %794 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = bufferization.materialize_in_destination %793 in %789 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %795 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %794 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %796 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %794 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %797:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %963:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %796 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%795, %790 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %964 = arith.negf %out : f32
        %965 = math.exp %964 : f32
        %966 = arith.addf %965, %cst_6 : f32
        %967 = arith.divf %cst_6, %966 : f32
        %968 = arith.mulf %out, %967 : f32
        %969 = arith.mulf %968, %in_9 : f32
        %970 = arith.mulf %in, %969 : f32
        %971 = arith.addf %out_10, %970 : f32
        linalg.yield %969, %971 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %963#0, %963#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %798 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%797#1 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %799 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %798, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %800 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg5[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%797#1, %799, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%2 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %801 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%2 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %800 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %802 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %645#1[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %800 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      cinm.yield %964 : tensor<768xf32>
    }
    %803 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %extracted_slice_9 = tensor.extract_slice %644[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_9 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %800 : tensor<768x768xf32>, tensor<768xf32>) outs(%963 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %965 = arith.mulf %in, %in_10 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<768xf32>
      %inserted_slice = tensor.insert_slice %964 into %644[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %inserted_slice : tensor<6x1024x768xf32>
    }
    %804:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %801, %arg18 = %802) -> (tensor<768xf32>, tensor<768xf32>) {
        %964 = arith.remui %arg16, %c48 : index
        %965 = arith.index_cast %964 : index to i64
        %966 = arith.uitofp %965 : i64 to f32
        %967 = arith.divf %966, %cst_7 : f32
        %968 = math.powf %cst_8, %967 : f32
        %969 = arith.divf %cst_6, %968 : f32
        %970 = arith.mulf %7#0, %969 : f32
        %971 = math.cos %970 : f32
        %972 = math.sin %970 : f32
        %973 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_9 = tensor.extract %arg17[%973] : tensor<768xf32>
        %974 = arith.mulf %extracted, %971 : f32
        %975 = arith.mulf %extracted_9, %972 : f32
        %976 = arith.subf %974, %975 : f32
        %inserted = tensor.insert %976 into %arg17[%arg16] : tensor<768xf32>
        %977 = arith.mulf %extracted, %972 : f32
        %978 = arith.mulf %extracted_9, %971 : f32
        %979 = arith.addf %977, %978 : f32
        %inserted_10 = tensor.insert %979 into %inserted[%973] : tensor<768xf32>
        %980 = bufferization.materialize_in_destination %inserted_10 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %981 = arith.cmpi ult, %arg16, %c768 : index
        %982 = scf.if %981 -> (tensor<768xf32>) {
          %extracted_11 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_12 = tensor.extract %arg18[%973] : tensor<768xf32>
          %983 = arith.mulf %extracted_11, %971 : f32
          %984 = arith.mulf %extracted_12, %972 : f32
          %985 = arith.subf %983, %984 : f32
          %inserted_13 = tensor.insert %985 into %arg18[%arg16] : tensor<768xf32>
          %986 = arith.mulf %extracted_11, %972 : f32
          %987 = arith.mulf %extracted_12, %971 : f32
          %988 = arith.addf %986, %987 : f32
          %inserted_14 = tensor.insert %988 into %inserted_13[%973] : tensor<768xf32>
          %989 = bufferization.materialize_in_destination %inserted_14 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %989 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %980, %982 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice = tensor.insert_slice %963#1 into %645#1[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %963#0, %inserted_slice : tensor<768xf32>, tensor<6x1024x768xf32>
    }
    %805 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %806 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%805 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %807 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %806) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %808 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%807 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %809 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%807, %808 : tensor<1024xf32>, f32) outs(%807 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %810 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%809 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %811 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %809 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      cinm.yield %expanded : tensor<1x1024xf32>
    }
    %812 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%15#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%811, %810, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %813 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %812 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %800[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %814 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %815 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%814 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %816 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %815) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %817 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%816 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %818 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%816, %817 : tensor<1024xf32>, f32) outs(%816 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %819 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%818 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %820:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %818 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %813[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %821 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%820#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%820#0, %819, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %822 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %821 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %813[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %823 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %824 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%823 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %825 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %824) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %826 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%825 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %827 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%825, %826 : tensor<1024xf32>, f32) outs(%825 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %828 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%827 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %829:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %827 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %822[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %830 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%829#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%829#0, %828, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %831 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %830 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %822[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %832 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %833 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%832 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %834 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %833) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %835 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%834 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %836 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%834, %835 : tensor<1024xf32>, f32) outs(%834 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %837 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%836 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %838:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %836 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %831[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %839 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%838#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%838#0, %837, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %840 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %839 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %831[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %841 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %842 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%841 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %843 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %842) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %844 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%843 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %845 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%843, %844 : tensor<1024xf32>, f32) outs(%843 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %846 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%845 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %847:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %845 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %840[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %848 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%847#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%847#0, %846, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %849 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %848 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %840[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %850 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %851 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%850 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %852 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %851) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %853 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%852 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %854 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%852, %853 : tensor<1024xf32>, f32) outs(%852 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %855 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%854 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %856:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %854 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %849[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %857 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%856#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%856#0, %855, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %858 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %857 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %849[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %859 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %860 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%859 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %861 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %860) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %862 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%861 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %863 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%861, %862 : tensor<1024xf32>, f32) outs(%861 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %864 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%863 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %865:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %863 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %858[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %866 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%865#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%865#0, %864, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %867 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %866 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %858[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %868 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %869 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%868 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %870 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %869) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %871 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%870 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %872 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%870, %871 : tensor<1024xf32>, f32) outs(%870 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %873 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%872 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %874:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %872 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %867[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %875 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%874#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%874#0, %873, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %876 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %875 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %867[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %877 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %878 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%877 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %879 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %878) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %880 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%879 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %881 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%879, %880 : tensor<1024xf32>, f32) outs(%879 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %882 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%881 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %883:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %881 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %876[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %884 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%883#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%883#0, %882, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %885 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %884 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %876[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %886 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %887 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%886 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %888 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %887) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %889 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%888 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %890 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%888, %889 : tensor<1024xf32>, f32) outs(%888 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %891 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%890 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %892:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %890 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %885[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %893 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%892#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%892#0, %891, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %894 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %893 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %885[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %895 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %896 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%895 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %897 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %896) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %898 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%897 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %899 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%897, %898 : tensor<1024xf32>, f32) outs(%897 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %900 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%899 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %901:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %899 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %894[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %902 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%901#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%901#0, %900, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %903 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %902 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %894[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %904 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %905 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%904 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %906 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %905) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %907 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%906 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %908 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%906, %907 : tensor<1024xf32>, f32) outs(%906 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %909 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%908 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %910:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %908 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %903[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %911 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%910#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%910#0, %909, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %912 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %911 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %903[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %913 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %914 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%913 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %915 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %914) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %916 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%915 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %917 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%915, %916 : tensor<1024xf32>, f32) outs(%915 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %918 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%917 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %919:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %917 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %912[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %920 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%919#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%919#0, %918, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %921 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %920 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %912[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %922 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %923 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%922 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %924 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %923) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %925 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%924 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %926 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%924, %925 : tensor<1024xf32>, f32) outs(%924 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %927 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%926 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %928:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %926 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %921[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %929 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%928#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%928#0, %927, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %930 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %929 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %921[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %931 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %932 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%931 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %933 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %932) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %934 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%933 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %935 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%933, %934 : tensor<1024xf32>, f32) outs(%933 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %936 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%935 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %937:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %935 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %930[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %938 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%937#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%937#0, %936, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %939 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %938 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %930[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
      cinm.yield %inserted_slice : tensor<768xf32>
    }
    %940 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %804#1[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %extracted_slice_10 = tensor.extract_slice %804#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %extracted_slice_10 : tensor<1024x48xf32>, tensor<48xf32>) outs(%963 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in, %in_11 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<1024xf32>
      cinm.yield %964 : tensor<1024xf32>
    }
    %941 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%940 : tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %964 = arith.divf %in, %cst_0 : f32
        linalg.yield %964 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %942 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = scf.for %arg16 = %7#3 to %c1024 step %c1 iter_args(%arg17 = %941) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %963 : tensor<1024xf32>
    }
    %943 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%942 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.maxnumf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %944 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%942, %943 : tensor<1024xf32>, f32) outs(%942 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.subf %in, %in_9 : f32
        %965 = math.exp %964 : f32
        linalg.yield %965 : f32
      } -> tensor<1024xf32>
      cinm.yield %963 : tensor<1024xf32>
    }
    %945 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%944 : tensor<1024xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.addf %in, %out : f32
        linalg.yield %966 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %946:2 = cinm.compute on platform #cinm.host_platform -> tensor<1x1024xf32>, tensor<1x48xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %expanded = tensor.expand_shape %944 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      %extracted_slice = tensor.extract_slice %939[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %reshape = tensor.reshape %extracted_slice(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
      cinm.yield %expanded, %reshape : tensor<1x1024xf32>, tensor<1x48xf32>
    }
    %947 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %803[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
      %extracted_slice_9 = tensor.extract_slice %extracted_slice[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
      %963 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%946#1 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %964 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%946#0, %945, %extracted_slice_9 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%963 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.divf %in, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.addf %out, %966 : f32
        linalg.yield %967 : f32
      } -> tensor<1x48xf32>
      cinm.yield %964 : tensor<1x48xf32>
    }
    %948 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %collapsed = tensor.collapse_shape %947 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
      %inserted_slice = tensor.insert_slice %collapsed into %939[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %963 = bufferization.materialize_in_destination %inserted_slice in %800 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %949 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
      %963 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %948 : tensor<768x768xf32>, tensor<768xf32>) outs(%797#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.addf %out, %964 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %950 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%949 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %951 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %950, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      cinm.yield %965 : f32
    }
    %952 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg13[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
      %963 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%949, %951, %extracted_slice : tensor<768xf32>, f32, tensor<768xf32>) outs(%948 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
        %964 = arith.mulf %in, %in_9 : f32
        %965 = arith.mulf %964, %in_10 : f32
        linalg.yield %965 : f32
      } -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %953 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = bufferization.materialize_in_destination %952 in %948 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %963 : tensor<768xf32>
    }
    %954 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %953 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %955 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%158 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %953 : tensor<2048x768xf32>, tensor<768xf32>) outs(%963 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %965 = arith.mulf %in, %in_9 : f32
        %966 = arith.addf %out, %965 : f32
        linalg.yield %966 : f32
      } -> tensor<2048xf32>
      cinm.yield %964 : tensor<2048xf32>
    }
    %956:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %extracted_slice = tensor.extract_slice %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
      %963:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %955 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%954, %949 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32, %out_10: f32):
        %964 = arith.negf %out : f32
        %965 = math.exp %964 : f32
        %966 = arith.addf %965, %cst_6 : f32
        %967 = arith.divf %cst_6, %966 : f32
        %968 = arith.mulf %out, %967 : f32
        %969 = arith.mulf %968, %in_9 : f32
        %970 = arith.mulf %in, %969 : f32
        %971 = arith.addf %out_10, %970 : f32
        linalg.yield %969, %971 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %963#0, %963#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %957 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %963 = tensor.empty() : tensor<f32>
      %964 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%963 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %965 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%956#1 : tensor<768xf32>) outs(%964 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %966 = arith.mulf %in, %in : f32
        %967 = arith.addf %966, %out : f32
        linalg.yield %967 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %965[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %958 = tensor.empty() : tensor<34048x768xf32>
    %959:2 = cinm.compute on platform #cinm.host_platform -> f32, tensor<34048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %963 = arith.divf %957, %cst_4 : f32
      %964 = arith.addf %963, %cst_5 : f32
      %965 = math.rsqrt %964 : f32
      %966 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%958 : tensor<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<34048x768xf32>
      %inserted_slice = tensor.insert_slice %arg15 into %966[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
      cinm.yield %965, %inserted_slice : f32, tensor<34048x768xf32>
    }
    %960 = tensor.empty() : tensor<34048xf32>
    %961 = cinm.compute -> tensor<34048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %963 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%960 : tensor<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<34048xf32>
      %964 = linalg.generic {indexing_maps = [#map3, #map4, #map10, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%959#1, %956#1, %959#0, %arg14 : tensor<34048x768xf32>, tensor<768xf32>, f32, tensor<768xf32>) outs(%963 : tensor<34048xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %out: f32):
        %965 = arith.mulf %in_9, %in_10 : f32
        %966 = arith.mulf %965, %in_11 : f32
        %967 = arith.mulf %in, %966 : f32
        %968 = arith.addf %out, %967 : f32
        linalg.yield %968 : f32
      } -> tensor<34048xf32>
      cinm.yield %964 : tensor<34048xf32>
    }
    %962 = cinm.compute on platform #cinm.host_platform -> tensor<32000xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %extracted_slice = tensor.extract_slice %961[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
      cinm.yield %extracted_slice : tensor<32000xf32>
    }
    return %962 : tensor<32000xf32>
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
