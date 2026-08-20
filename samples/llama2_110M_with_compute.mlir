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
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %extracted_slice_9 = tensor.extract_slice %arg5[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %0 = tensor.empty() : tensor<f32>
    %1 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %2 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %1, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %3 = tensor.empty() : tensor<768xf32>
    %4 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %2, %extracted_slice_9 : tensor<768xf32>, f32, tensor<768xf32>) outs(%3 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_10 = tensor.extract_slice %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_11 = tensor.extract_slice %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_12 = tensor.extract_slice %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %5 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %4 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_13 = tensor.extract_slice %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %6 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_13 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_11, %4 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_14 = tensor.extract_slice %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %7 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_14 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_12, %4 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %inserted_slice = tensor.insert_slice %7 into %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %8:3 = cinm.compute on platform #cinm.host_platform -> f32, tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.index_cast %arg1 : index to i64
      %779 = arith.uitofp %778 : i64 to f32
      %780:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %5, %arg18 = %6) -> (tensor<768xf32>, tensor<768xf32>) {
        %781 = arith.remui %arg16, %c48 : index
        %782 = arith.index_cast %781 : index to i64
        %783 = arith.uitofp %782 : i64 to f32
        %784 = arith.divf %783, %cst_7 : f32
        %785 = math.powf %cst_8, %784 : f32
        %786 = arith.divf %cst_6, %785 : f32
        %787 = arith.mulf %779, %786 : f32
        %788 = math.cos %787 : f32
        %789 = math.sin %787 : f32
        %790 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_855 = tensor.extract %arg17[%790] : tensor<768xf32>
        %791 = arith.mulf %extracted, %788 : f32
        %792 = arith.mulf %extracted_855, %789 : f32
        %793 = arith.subf %791, %792 : f32
        %inserted = tensor.insert %793 into %arg17[%arg16] : tensor<768xf32>
        %794 = arith.mulf %extracted, %789 : f32
        %795 = arith.mulf %extracted_855, %788 : f32
        %796 = arith.addf %794, %795 : f32
        %inserted_856 = tensor.insert %796 into %inserted[%790] : tensor<768xf32>
        %797 = bufferization.materialize_in_destination %inserted_856 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %798 = arith.cmpi ult, %arg16, %c768 : index
        %799 = scf.if %798 -> (tensor<768xf32>) {
          %extracted_857 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_858 = tensor.extract %arg18[%790] : tensor<768xf32>
          %800 = arith.mulf %extracted_857, %788 : f32
          %801 = arith.mulf %extracted_858, %789 : f32
          %802 = arith.subf %800, %801 : f32
          %inserted_859 = tensor.insert %802 into %arg18[%arg16] : tensor<768xf32>
          %803 = arith.mulf %extracted_857, %789 : f32
          %804 = arith.mulf %extracted_858, %788 : f32
          %805 = arith.addf %803, %804 : f32
          %inserted_860 = tensor.insert %805 into %inserted_859[%790] : tensor<768xf32>
          %806 = bufferization.materialize_in_destination %inserted_860 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %806 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %797, %799 : tensor<768xf32>, tensor<768xf32>
      }
      cinm.yield %779, %780#0, %780#1 : f32, tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_15 = tensor.insert_slice %8#2 into %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_16 = tensor.extract_slice %inserted_slice_15[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_17 = tensor.extract_slice %inserted_slice[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %9 = cinm.compute on platform #cinm.host_platform -> index attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.addi %arg1, %c1 : index
      cinm.yield %778 : index
    }
    %extracted_slice_18 = tensor.extract_slice %8#1[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_19 = tensor.extract_slice %extracted_slice_16[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %10 = tensor.empty() : tensor<1024xf32>
    %11 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_19, %extracted_slice_18 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %12 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%11 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %13 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %12) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %14 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%13 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %15 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%13, %14 : tensor<1024xf32>, f32) outs(%13 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %16 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%15 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded = tensor.expand_shape %15 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_20 = tensor.extract_slice %extracted_slice_17[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_21 = tensor.extract_slice %3[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape = tensor.reshape %extracted_slice_21(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %17 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded, %16, %extracted_slice_20 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed = tensor.collapse_shape %17 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_22 = tensor.insert_slice %collapsed into %4[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_23 = tensor.extract_slice %8#1[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_24 = tensor.extract_slice %extracted_slice_16[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %18 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_24, %extracted_slice_23 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %19 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%18 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %20 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %19) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %21 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%20 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %22 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%20, %21 : tensor<1024xf32>, f32) outs(%20 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %23 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%22 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_25 = tensor.expand_shape %22 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_26 = tensor.extract_slice %extracted_slice_17[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_27 = tensor.extract_slice %inserted_slice_22[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_28 = tensor.reshape %extracted_slice_27(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %24 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_28 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_25, %23, %extracted_slice_26 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_29 = tensor.collapse_shape %24 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_30 = tensor.insert_slice %collapsed_29 into %inserted_slice_22[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_31 = tensor.extract_slice %8#1[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_32 = tensor.extract_slice %extracted_slice_16[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %25 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_32, %extracted_slice_31 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %26 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%25 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %27 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %26) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %28 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%27 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %29 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%27, %28 : tensor<1024xf32>, f32) outs(%27 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %30 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%29 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_33 = tensor.expand_shape %29 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_34 = tensor.extract_slice %extracted_slice_17[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_35 = tensor.extract_slice %inserted_slice_30[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_36 = tensor.reshape %extracted_slice_35(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %31 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_36 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_33, %30, %extracted_slice_34 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_37 = tensor.collapse_shape %31 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_38 = tensor.insert_slice %collapsed_37 into %inserted_slice_30[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_39 = tensor.extract_slice %8#1[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_40 = tensor.extract_slice %extracted_slice_16[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %32 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_40, %extracted_slice_39 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %33 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%32 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %34 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %33) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %35 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%34 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %36 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%34, %35 : tensor<1024xf32>, f32) outs(%34 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %37 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%36 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_41 = tensor.expand_shape %36 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_42 = tensor.extract_slice %extracted_slice_17[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_43 = tensor.extract_slice %inserted_slice_38[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_44 = tensor.reshape %extracted_slice_43(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %38 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_44 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_41, %37, %extracted_slice_42 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_45 = tensor.collapse_shape %38 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_46 = tensor.insert_slice %collapsed_45 into %inserted_slice_38[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_47 = tensor.extract_slice %8#1[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_48 = tensor.extract_slice %extracted_slice_16[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %39 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_48, %extracted_slice_47 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %40 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%39 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %41 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %40) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %42 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%41 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %43 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%41, %42 : tensor<1024xf32>, f32) outs(%41 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %44 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%43 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_49 = tensor.expand_shape %43 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_50 = tensor.extract_slice %extracted_slice_17[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_51 = tensor.extract_slice %inserted_slice_46[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_52 = tensor.reshape %extracted_slice_51(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %45 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_52 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_49, %44, %extracted_slice_50 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_53 = tensor.collapse_shape %45 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_54 = tensor.insert_slice %collapsed_53 into %inserted_slice_46[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_55 = tensor.extract_slice %8#1[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_56 = tensor.extract_slice %extracted_slice_16[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %46 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_56, %extracted_slice_55 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %47 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%46 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %48 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %47) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %49 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%48 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %50 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%48, %49 : tensor<1024xf32>, f32) outs(%48 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %51 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%50 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_57 = tensor.expand_shape %50 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_58 = tensor.extract_slice %extracted_slice_17[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_59 = tensor.extract_slice %inserted_slice_54[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_60 = tensor.reshape %extracted_slice_59(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %52 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_60 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_57, %51, %extracted_slice_58 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_61 = tensor.collapse_shape %52 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_62 = tensor.insert_slice %collapsed_61 into %inserted_slice_54[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_63 = tensor.extract_slice %8#1[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_64 = tensor.extract_slice %extracted_slice_16[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %53 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_64, %extracted_slice_63 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %54 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%53 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %55 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %54) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %56 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%55 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %57 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%55, %56 : tensor<1024xf32>, f32) outs(%55 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %58 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%57 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_65 = tensor.expand_shape %57 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_66 = tensor.extract_slice %extracted_slice_17[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_67 = tensor.extract_slice %inserted_slice_62[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_68 = tensor.reshape %extracted_slice_67(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %59 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_68 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_65, %58, %extracted_slice_66 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_69 = tensor.collapse_shape %59 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_70 = tensor.insert_slice %collapsed_69 into %inserted_slice_62[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_71 = tensor.extract_slice %8#1[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_72 = tensor.extract_slice %extracted_slice_16[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %60 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_72, %extracted_slice_71 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %61 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%60 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %62 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %61) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %63 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%62 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %64 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%62, %63 : tensor<1024xf32>, f32) outs(%62 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %65 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%64 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_73 = tensor.expand_shape %64 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_74 = tensor.extract_slice %extracted_slice_17[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_75 = tensor.extract_slice %inserted_slice_70[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_76 = tensor.reshape %extracted_slice_75(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %66 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_76 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_73, %65, %extracted_slice_74 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_77 = tensor.collapse_shape %66 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_78 = tensor.insert_slice %collapsed_77 into %inserted_slice_70[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_79 = tensor.extract_slice %8#1[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_80 = tensor.extract_slice %extracted_slice_16[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %67 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_80, %extracted_slice_79 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %68 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%67 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %69 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %68) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %70 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%69 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %71 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%69, %70 : tensor<1024xf32>, f32) outs(%69 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %72 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%71 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_81 = tensor.expand_shape %71 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_82 = tensor.extract_slice %extracted_slice_17[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_83 = tensor.extract_slice %inserted_slice_78[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_84 = tensor.reshape %extracted_slice_83(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %73 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_84 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_81, %72, %extracted_slice_82 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_85 = tensor.collapse_shape %73 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_86 = tensor.insert_slice %collapsed_85 into %inserted_slice_78[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_87 = tensor.extract_slice %8#1[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_88 = tensor.extract_slice %extracted_slice_16[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %74 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_88, %extracted_slice_87 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %75 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%74 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %76 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %75) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %77 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%76 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %78 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%76, %77 : tensor<1024xf32>, f32) outs(%76 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %79 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%78 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_89 = tensor.expand_shape %78 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_90 = tensor.extract_slice %extracted_slice_17[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_91 = tensor.extract_slice %inserted_slice_86[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_92 = tensor.reshape %extracted_slice_91(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %80 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_92 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_89, %79, %extracted_slice_90 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_93 = tensor.collapse_shape %80 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_94 = tensor.insert_slice %collapsed_93 into %inserted_slice_86[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_95 = tensor.extract_slice %8#1[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_96 = tensor.extract_slice %extracted_slice_16[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %81 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_96, %extracted_slice_95 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %82 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%81 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %83 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %82) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %84 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%83 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %85 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%83, %84 : tensor<1024xf32>, f32) outs(%83 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %86 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%85 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_97 = tensor.expand_shape %85 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_98 = tensor.extract_slice %extracted_slice_17[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_99 = tensor.extract_slice %inserted_slice_94[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_100 = tensor.reshape %extracted_slice_99(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %87 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_100 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_97, %86, %extracted_slice_98 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_101 = tensor.collapse_shape %87 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_102 = tensor.insert_slice %collapsed_101 into %inserted_slice_94[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_103 = tensor.extract_slice %8#1[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_104 = tensor.extract_slice %extracted_slice_16[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %88 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_104, %extracted_slice_103 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %89 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%88 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %90 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %89) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %91 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%90 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %92 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%90, %91 : tensor<1024xf32>, f32) outs(%90 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %93 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%92 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_105 = tensor.expand_shape %92 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_106 = tensor.extract_slice %extracted_slice_17[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_107 = tensor.extract_slice %inserted_slice_102[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_108 = tensor.reshape %extracted_slice_107(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %94 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_108 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_105, %93, %extracted_slice_106 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_109 = tensor.collapse_shape %94 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_110 = tensor.insert_slice %collapsed_109 into %inserted_slice_102[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_111 = tensor.extract_slice %8#1[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_112 = tensor.extract_slice %extracted_slice_16[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %95 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_112, %extracted_slice_111 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %96 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%95 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %97 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %96) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %98 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%97 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %99 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%97, %98 : tensor<1024xf32>, f32) outs(%97 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %100 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%99 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_113 = tensor.expand_shape %99 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_114 = tensor.extract_slice %extracted_slice_17[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_115 = tensor.extract_slice %inserted_slice_110[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_116 = tensor.reshape %extracted_slice_115(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %101 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_116 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_113, %100, %extracted_slice_114 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_117 = tensor.collapse_shape %101 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_118 = tensor.insert_slice %collapsed_117 into %inserted_slice_110[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_119 = tensor.extract_slice %8#1[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_120 = tensor.extract_slice %extracted_slice_16[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %102 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_120, %extracted_slice_119 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %103 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%102 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %104 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %103) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %105 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%104 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %106 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%104, %105 : tensor<1024xf32>, f32) outs(%104 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %107 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%106 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_121 = tensor.expand_shape %106 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_122 = tensor.extract_slice %extracted_slice_17[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_123 = tensor.extract_slice %inserted_slice_118[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_124 = tensor.reshape %extracted_slice_123(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %108 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_124 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_121, %107, %extracted_slice_122 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_125 = tensor.collapse_shape %108 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_126 = tensor.insert_slice %collapsed_125 into %inserted_slice_118[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_127 = tensor.extract_slice %8#1[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_128 = tensor.extract_slice %extracted_slice_16[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %109 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_128, %extracted_slice_127 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %110 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%109 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %111 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %110) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %112 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%111 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %113 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%111, %112 : tensor<1024xf32>, f32) outs(%111 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %114 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%113 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_129 = tensor.expand_shape %113 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_130 = tensor.extract_slice %extracted_slice_17[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_131 = tensor.extract_slice %inserted_slice_126[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_132 = tensor.reshape %extracted_slice_131(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %115 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_132 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_129, %114, %extracted_slice_130 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_133 = tensor.collapse_shape %115 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_134 = tensor.insert_slice %collapsed_133 into %inserted_slice_126[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_135 = tensor.extract_slice %8#1[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_136 = tensor.extract_slice %extracted_slice_16[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %116 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_136, %extracted_slice_135 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %117 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%116 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %118 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %117) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %119 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%118 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %120 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%118, %119 : tensor<1024xf32>, f32) outs(%118 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %121 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%120 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_137 = tensor.expand_shape %120 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_138 = tensor.extract_slice %extracted_slice_17[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_139 = tensor.extract_slice %inserted_slice_134[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_140 = tensor.reshape %extracted_slice_139(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %122 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_140 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_137, %121, %extracted_slice_138 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_141 = tensor.collapse_shape %122 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_142 = tensor.insert_slice %collapsed_141 into %inserted_slice_134[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %123 = bufferization.materialize_in_destination %inserted_slice_142 in %4 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_143 = tensor.extract_slice %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %124 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_143, %123 : tensor<768x768xf32>, tensor<768xf32>) outs(%extracted_slice : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.addf %out, %779 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_144 = tensor.extract_slice %arg13[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %125 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%124 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %126 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %125, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %127 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%124, %126, %extracted_slice_144 : tensor<768xf32>, f32, tensor<768xf32>) outs(%123 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %128 = bufferization.materialize_in_destination %127 in %123 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_145 = tensor.extract_slice %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_146 = tensor.extract_slice %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %129 = tensor.empty() : tensor<2048xf32>
    %130 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_145, %128 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %131 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_146, %128 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %extracted_slice_147 = tensor.extract_slice %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %132:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_147, %131 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%130, %124 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32, %out_856: f32):
        %779 = arith.negf %out : f32
        %780 = math.exp %779 : f32
        %781 = arith.addf %780, %cst_6 : f32
        %782 = arith.divf %cst_6, %781 : f32
        %783 = arith.mulf %out, %782 : f32
        %784 = arith.mulf %783, %in_855 : f32
        %785 = arith.mulf %in, %784 : f32
        %786 = arith.addf %out_856, %785 : f32
        linalg.yield %784, %786 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %778#0, %778#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %extracted_slice_148 = tensor.extract_slice %arg5[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %133 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%132#1 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %134 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %133, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %135 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%132#1, %134, %extracted_slice_148 : tensor<768xf32>, f32, tensor<768xf32>) outs(%3 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_149 = tensor.extract_slice %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_150 = tensor.extract_slice %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_151 = tensor.extract_slice %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %136 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_149, %135 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_152 = tensor.extract_slice %inserted_slice_15[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %137 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_152 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_150, %135 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_153 = tensor.extract_slice %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %138 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_153 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_151, %135 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %inserted_slice_154 = tensor.insert_slice %138 into %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %139:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %136, %arg18 = %137) -> (tensor<768xf32>, tensor<768xf32>) {
        %779 = arith.remui %arg16, %c48 : index
        %780 = arith.index_cast %779 : index to i64
        %781 = arith.uitofp %780 : i64 to f32
        %782 = arith.divf %781, %cst_7 : f32
        %783 = math.powf %cst_8, %782 : f32
        %784 = arith.divf %cst_6, %783 : f32
        %785 = arith.mulf %8#0, %784 : f32
        %786 = math.cos %785 : f32
        %787 = math.sin %785 : f32
        %788 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_855 = tensor.extract %arg17[%788] : tensor<768xf32>
        %789 = arith.mulf %extracted, %786 : f32
        %790 = arith.mulf %extracted_855, %787 : f32
        %791 = arith.subf %789, %790 : f32
        %inserted = tensor.insert %791 into %arg17[%arg16] : tensor<768xf32>
        %792 = arith.mulf %extracted, %787 : f32
        %793 = arith.mulf %extracted_855, %786 : f32
        %794 = arith.addf %792, %793 : f32
        %inserted_856 = tensor.insert %794 into %inserted[%788] : tensor<768xf32>
        %795 = bufferization.materialize_in_destination %inserted_856 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %796 = arith.cmpi ult, %arg16, %c768 : index
        %797 = scf.if %796 -> (tensor<768xf32>) {
          %extracted_857 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_858 = tensor.extract %arg18[%788] : tensor<768xf32>
          %798 = arith.mulf %extracted_857, %786 : f32
          %799 = arith.mulf %extracted_858, %787 : f32
          %800 = arith.subf %798, %799 : f32
          %inserted_859 = tensor.insert %800 into %arg18[%arg16] : tensor<768xf32>
          %801 = arith.mulf %extracted_857, %787 : f32
          %802 = arith.mulf %extracted_858, %786 : f32
          %803 = arith.addf %801, %802 : f32
          %inserted_860 = tensor.insert %803 into %inserted_859[%788] : tensor<768xf32>
          %804 = bufferization.materialize_in_destination %inserted_860 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %804 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %795, %797 : tensor<768xf32>, tensor<768xf32>
      }
      cinm.yield %778#0, %778#1 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_155 = tensor.insert_slice %139#1 into %inserted_slice_15[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_156 = tensor.extract_slice %inserted_slice_155[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_157 = tensor.extract_slice %inserted_slice_154[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_158 = tensor.extract_slice %139#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_159 = tensor.extract_slice %extracted_slice_156[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %140 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_159, %extracted_slice_158 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %141 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%140 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %142 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %141) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %143 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%142 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %144 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%142, %143 : tensor<1024xf32>, f32) outs(%142 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %145 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%144 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_160 = tensor.expand_shape %144 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_161 = tensor.extract_slice %extracted_slice_157[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %146 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_160, %145, %extracted_slice_161 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_162 = tensor.collapse_shape %146 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_163 = tensor.insert_slice %collapsed_162 into %135[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_164 = tensor.extract_slice %139#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_165 = tensor.extract_slice %extracted_slice_156[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %147 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_165, %extracted_slice_164 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %148 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%147 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %149 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %148) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %150 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%149 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %151 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%149, %150 : tensor<1024xf32>, f32) outs(%149 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %152 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%151 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_166 = tensor.expand_shape %151 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_167 = tensor.extract_slice %extracted_slice_157[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_168 = tensor.extract_slice %inserted_slice_163[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_169 = tensor.reshape %extracted_slice_168(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %153 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_169 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_166, %152, %extracted_slice_167 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_170 = tensor.collapse_shape %153 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_171 = tensor.insert_slice %collapsed_170 into %inserted_slice_163[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_172 = tensor.extract_slice %139#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_173 = tensor.extract_slice %extracted_slice_156[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %154 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_173, %extracted_slice_172 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %155 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%154 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %156 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %155) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %157 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%156 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %158 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%156, %157 : tensor<1024xf32>, f32) outs(%156 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %159 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%158 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_174 = tensor.expand_shape %158 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_175 = tensor.extract_slice %extracted_slice_157[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_176 = tensor.extract_slice %inserted_slice_171[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_177 = tensor.reshape %extracted_slice_176(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %160 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_177 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_174, %159, %extracted_slice_175 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_178 = tensor.collapse_shape %160 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_179 = tensor.insert_slice %collapsed_178 into %inserted_slice_171[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_180 = tensor.extract_slice %139#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_181 = tensor.extract_slice %extracted_slice_156[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %161 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_181, %extracted_slice_180 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %162 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%161 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %163 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %162) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %164 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%163 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %165 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%163, %164 : tensor<1024xf32>, f32) outs(%163 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %166 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%165 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_182 = tensor.expand_shape %165 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_183 = tensor.extract_slice %extracted_slice_157[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_184 = tensor.extract_slice %inserted_slice_179[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_185 = tensor.reshape %extracted_slice_184(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %167 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_185 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_182, %166, %extracted_slice_183 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_186 = tensor.collapse_shape %167 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_187 = tensor.insert_slice %collapsed_186 into %inserted_slice_179[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_188 = tensor.extract_slice %139#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_189 = tensor.extract_slice %extracted_slice_156[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %168 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_189, %extracted_slice_188 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %169 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%168 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %170 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %169) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %171 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%170 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %172 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%170, %171 : tensor<1024xf32>, f32) outs(%170 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %173 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%172 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_190 = tensor.expand_shape %172 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_191 = tensor.extract_slice %extracted_slice_157[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_192 = tensor.extract_slice %inserted_slice_187[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_193 = tensor.reshape %extracted_slice_192(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %174 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_193 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_190, %173, %extracted_slice_191 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_194 = tensor.collapse_shape %174 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_195 = tensor.insert_slice %collapsed_194 into %inserted_slice_187[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_196 = tensor.extract_slice %139#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_197 = tensor.extract_slice %extracted_slice_156[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %175 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_197, %extracted_slice_196 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %176 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%175 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %177 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %176) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %178 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%177 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %179 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%177, %178 : tensor<1024xf32>, f32) outs(%177 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %180 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%179 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_198 = tensor.expand_shape %179 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_199 = tensor.extract_slice %extracted_slice_157[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_200 = tensor.extract_slice %inserted_slice_195[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_201 = tensor.reshape %extracted_slice_200(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %181 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_201 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_198, %180, %extracted_slice_199 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_202 = tensor.collapse_shape %181 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_203 = tensor.insert_slice %collapsed_202 into %inserted_slice_195[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_204 = tensor.extract_slice %139#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_205 = tensor.extract_slice %extracted_slice_156[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %182 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_205, %extracted_slice_204 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %183 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%182 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %184 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %183) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %185 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%184 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %186 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%184, %185 : tensor<1024xf32>, f32) outs(%184 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %187 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%186 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_206 = tensor.expand_shape %186 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_207 = tensor.extract_slice %extracted_slice_157[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_208 = tensor.extract_slice %inserted_slice_203[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_209 = tensor.reshape %extracted_slice_208(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %188 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_209 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_206, %187, %extracted_slice_207 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_210 = tensor.collapse_shape %188 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_211 = tensor.insert_slice %collapsed_210 into %inserted_slice_203[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_212 = tensor.extract_slice %139#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_213 = tensor.extract_slice %extracted_slice_156[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %189 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_213, %extracted_slice_212 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %190 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%189 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %191 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %190) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %192 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%191 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %193 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%191, %192 : tensor<1024xf32>, f32) outs(%191 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %194 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%193 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_214 = tensor.expand_shape %193 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_215 = tensor.extract_slice %extracted_slice_157[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_216 = tensor.extract_slice %inserted_slice_211[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_217 = tensor.reshape %extracted_slice_216(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %195 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_217 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_214, %194, %extracted_slice_215 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_218 = tensor.collapse_shape %195 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_219 = tensor.insert_slice %collapsed_218 into %inserted_slice_211[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_220 = tensor.extract_slice %139#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_221 = tensor.extract_slice %extracted_slice_156[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %196 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_221, %extracted_slice_220 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %197 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%196 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %198 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %197) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %199 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%198 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %200 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%198, %199 : tensor<1024xf32>, f32) outs(%198 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %201 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%200 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_222 = tensor.expand_shape %200 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_223 = tensor.extract_slice %extracted_slice_157[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_224 = tensor.extract_slice %inserted_slice_219[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_225 = tensor.reshape %extracted_slice_224(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %202 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_225 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_222, %201, %extracted_slice_223 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_226 = tensor.collapse_shape %202 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_227 = tensor.insert_slice %collapsed_226 into %inserted_slice_219[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_228 = tensor.extract_slice %139#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_229 = tensor.extract_slice %extracted_slice_156[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %203 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_229, %extracted_slice_228 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %204 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%203 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %205 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %204) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %206 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%205 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %207 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%205, %206 : tensor<1024xf32>, f32) outs(%205 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %208 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%207 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_230 = tensor.expand_shape %207 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_231 = tensor.extract_slice %extracted_slice_157[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_232 = tensor.extract_slice %inserted_slice_227[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_233 = tensor.reshape %extracted_slice_232(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %209 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_233 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_230, %208, %extracted_slice_231 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_234 = tensor.collapse_shape %209 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_235 = tensor.insert_slice %collapsed_234 into %inserted_slice_227[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_236 = tensor.extract_slice %139#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_237 = tensor.extract_slice %extracted_slice_156[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %210 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_237, %extracted_slice_236 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %211 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%210 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %212 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %211) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %213 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%212 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %214 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%212, %213 : tensor<1024xf32>, f32) outs(%212 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %215 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%214 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_238 = tensor.expand_shape %214 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_239 = tensor.extract_slice %extracted_slice_157[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_240 = tensor.extract_slice %inserted_slice_235[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_241 = tensor.reshape %extracted_slice_240(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %216 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_241 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_238, %215, %extracted_slice_239 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_242 = tensor.collapse_shape %216 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_243 = tensor.insert_slice %collapsed_242 into %inserted_slice_235[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_244 = tensor.extract_slice %139#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_245 = tensor.extract_slice %extracted_slice_156[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %217 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_245, %extracted_slice_244 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %218 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%217 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %219 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %218) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %220 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%219 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %221 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%219, %220 : tensor<1024xf32>, f32) outs(%219 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %222 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%221 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_246 = tensor.expand_shape %221 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_247 = tensor.extract_slice %extracted_slice_157[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_248 = tensor.extract_slice %inserted_slice_243[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_249 = tensor.reshape %extracted_slice_248(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %223 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_249 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_246, %222, %extracted_slice_247 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_250 = tensor.collapse_shape %223 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_251 = tensor.insert_slice %collapsed_250 into %inserted_slice_243[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_252 = tensor.extract_slice %139#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_253 = tensor.extract_slice %extracted_slice_156[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %224 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_253, %extracted_slice_252 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %225 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%224 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %226 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %225) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %227 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%226 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %228 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%226, %227 : tensor<1024xf32>, f32) outs(%226 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %229 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%228 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_254 = tensor.expand_shape %228 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_255 = tensor.extract_slice %extracted_slice_157[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_256 = tensor.extract_slice %inserted_slice_251[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_257 = tensor.reshape %extracted_slice_256(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %230 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_257 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_254, %229, %extracted_slice_255 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_258 = tensor.collapse_shape %230 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_259 = tensor.insert_slice %collapsed_258 into %inserted_slice_251[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_260 = tensor.extract_slice %139#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_261 = tensor.extract_slice %extracted_slice_156[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %231 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_261, %extracted_slice_260 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %232 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%231 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %233 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %232) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %234 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%233 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %235 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%233, %234 : tensor<1024xf32>, f32) outs(%233 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %236 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%235 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_262 = tensor.expand_shape %235 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_263 = tensor.extract_slice %extracted_slice_157[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_264 = tensor.extract_slice %inserted_slice_259[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_265 = tensor.reshape %extracted_slice_264(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %237 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_265 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_262, %236, %extracted_slice_263 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_266 = tensor.collapse_shape %237 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_267 = tensor.insert_slice %collapsed_266 into %inserted_slice_259[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_268 = tensor.extract_slice %139#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_269 = tensor.extract_slice %extracted_slice_156[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %238 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_269, %extracted_slice_268 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %239 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%238 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %240 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %239) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %241 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%240 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %242 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%240, %241 : tensor<1024xf32>, f32) outs(%240 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %243 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%242 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_270 = tensor.expand_shape %242 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_271 = tensor.extract_slice %extracted_slice_157[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_272 = tensor.extract_slice %inserted_slice_267[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_273 = tensor.reshape %extracted_slice_272(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %244 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_273 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_270, %243, %extracted_slice_271 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_274 = tensor.collapse_shape %244 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_275 = tensor.insert_slice %collapsed_274 into %inserted_slice_267[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_276 = tensor.extract_slice %139#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_277 = tensor.extract_slice %extracted_slice_156[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %245 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_277, %extracted_slice_276 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %246 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%245 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %247 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %246) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %248 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%247 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %249 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%247, %248 : tensor<1024xf32>, f32) outs(%247 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %250 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%249 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_278 = tensor.expand_shape %249 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_279 = tensor.extract_slice %extracted_slice_157[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_280 = tensor.extract_slice %inserted_slice_275[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_281 = tensor.reshape %extracted_slice_280(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %251 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_281 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_278, %250, %extracted_slice_279 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_282 = tensor.collapse_shape %251 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_283 = tensor.insert_slice %collapsed_282 into %inserted_slice_275[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %252 = bufferization.materialize_in_destination %inserted_slice_283 in %135 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_284 = tensor.extract_slice %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %253 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_284, %252 : tensor<768x768xf32>, tensor<768xf32>) outs(%132#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.addf %out, %779 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_285 = tensor.extract_slice %arg13[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %254 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%253 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %255 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %254, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %256 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%253, %255, %extracted_slice_285 : tensor<768xf32>, f32, tensor<768xf32>) outs(%252 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %257 = bufferization.materialize_in_destination %256 in %252 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_286 = tensor.extract_slice %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_287 = tensor.extract_slice %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %258 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_286, %257 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %259 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_287, %257 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %extracted_slice_288 = tensor.extract_slice %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %260:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_288, %259 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%258, %253 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32, %out_856: f32):
        %779 = arith.negf %out : f32
        %780 = math.exp %779 : f32
        %781 = arith.addf %780, %cst_6 : f32
        %782 = arith.divf %cst_6, %781 : f32
        %783 = arith.mulf %out, %782 : f32
        %784 = arith.mulf %783, %in_855 : f32
        %785 = arith.mulf %in, %784 : f32
        %786 = arith.addf %out_856, %785 : f32
        linalg.yield %784, %786 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %778#0, %778#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %extracted_slice_289 = tensor.extract_slice %arg5[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %261 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%260#1 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %262 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %261, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %263 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%260#1, %262, %extracted_slice_289 : tensor<768xf32>, f32, tensor<768xf32>) outs(%3 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_290 = tensor.extract_slice %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_291 = tensor.extract_slice %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_292 = tensor.extract_slice %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %264 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_290, %263 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_293 = tensor.extract_slice %inserted_slice_155[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %265 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_293 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_291, %263 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_294 = tensor.extract_slice %inserted_slice_154[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %266 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_294 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_292, %263 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %inserted_slice_295 = tensor.insert_slice %266 into %inserted_slice_154[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %267:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %264, %arg18 = %265) -> (tensor<768xf32>, tensor<768xf32>) {
        %779 = arith.remui %arg16, %c48 : index
        %780 = arith.index_cast %779 : index to i64
        %781 = arith.uitofp %780 : i64 to f32
        %782 = arith.divf %781, %cst_7 : f32
        %783 = math.powf %cst_8, %782 : f32
        %784 = arith.divf %cst_6, %783 : f32
        %785 = arith.mulf %8#0, %784 : f32
        %786 = math.cos %785 : f32
        %787 = math.sin %785 : f32
        %788 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_855 = tensor.extract %arg17[%788] : tensor<768xf32>
        %789 = arith.mulf %extracted, %786 : f32
        %790 = arith.mulf %extracted_855, %787 : f32
        %791 = arith.subf %789, %790 : f32
        %inserted = tensor.insert %791 into %arg17[%arg16] : tensor<768xf32>
        %792 = arith.mulf %extracted, %787 : f32
        %793 = arith.mulf %extracted_855, %786 : f32
        %794 = arith.addf %792, %793 : f32
        %inserted_856 = tensor.insert %794 into %inserted[%788] : tensor<768xf32>
        %795 = bufferization.materialize_in_destination %inserted_856 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %796 = arith.cmpi ult, %arg16, %c768 : index
        %797 = scf.if %796 -> (tensor<768xf32>) {
          %extracted_857 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_858 = tensor.extract %arg18[%788] : tensor<768xf32>
          %798 = arith.mulf %extracted_857, %786 : f32
          %799 = arith.mulf %extracted_858, %787 : f32
          %800 = arith.subf %798, %799 : f32
          %inserted_859 = tensor.insert %800 into %arg18[%arg16] : tensor<768xf32>
          %801 = arith.mulf %extracted_857, %787 : f32
          %802 = arith.mulf %extracted_858, %786 : f32
          %803 = arith.addf %801, %802 : f32
          %inserted_860 = tensor.insert %803 into %inserted_859[%788] : tensor<768xf32>
          %804 = bufferization.materialize_in_destination %inserted_860 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %804 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %795, %797 : tensor<768xf32>, tensor<768xf32>
      }
      cinm.yield %778#0, %778#1 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_296 = tensor.insert_slice %267#1 into %inserted_slice_155[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_297 = tensor.extract_slice %inserted_slice_296[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_298 = tensor.extract_slice %inserted_slice_295[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_299 = tensor.extract_slice %267#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_300 = tensor.extract_slice %extracted_slice_297[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %268 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_300, %extracted_slice_299 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %269 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%268 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %270 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %269) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %271 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%270 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %272 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%270, %271 : tensor<1024xf32>, f32) outs(%270 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %273 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%272 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_301 = tensor.expand_shape %272 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_302 = tensor.extract_slice %extracted_slice_298[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %274 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_301, %273, %extracted_slice_302 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_303 = tensor.collapse_shape %274 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_304 = tensor.insert_slice %collapsed_303 into %263[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_305 = tensor.extract_slice %267#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_306 = tensor.extract_slice %extracted_slice_297[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %275 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_306, %extracted_slice_305 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %276 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%275 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %277 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %276) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %278 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%277 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %279 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%277, %278 : tensor<1024xf32>, f32) outs(%277 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %280 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%279 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_307 = tensor.expand_shape %279 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_308 = tensor.extract_slice %extracted_slice_298[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_309 = tensor.extract_slice %inserted_slice_304[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_310 = tensor.reshape %extracted_slice_309(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %281 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_310 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_307, %280, %extracted_slice_308 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_311 = tensor.collapse_shape %281 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_312 = tensor.insert_slice %collapsed_311 into %inserted_slice_304[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_313 = tensor.extract_slice %267#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_314 = tensor.extract_slice %extracted_slice_297[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %282 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_314, %extracted_slice_313 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %283 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%282 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %284 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %283) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %285 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%284 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %286 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%284, %285 : tensor<1024xf32>, f32) outs(%284 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %287 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%286 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_315 = tensor.expand_shape %286 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_316 = tensor.extract_slice %extracted_slice_298[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_317 = tensor.extract_slice %inserted_slice_312[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_318 = tensor.reshape %extracted_slice_317(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %288 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_318 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_315, %287, %extracted_slice_316 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_319 = tensor.collapse_shape %288 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_320 = tensor.insert_slice %collapsed_319 into %inserted_slice_312[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_321 = tensor.extract_slice %267#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_322 = tensor.extract_slice %extracted_slice_297[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %289 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_322, %extracted_slice_321 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %290 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%289 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %291 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %290) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %292 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%291 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %293 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%291, %292 : tensor<1024xf32>, f32) outs(%291 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %294 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%293 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_323 = tensor.expand_shape %293 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_324 = tensor.extract_slice %extracted_slice_298[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_325 = tensor.extract_slice %inserted_slice_320[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_326 = tensor.reshape %extracted_slice_325(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %295 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_326 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_323, %294, %extracted_slice_324 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_327 = tensor.collapse_shape %295 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_328 = tensor.insert_slice %collapsed_327 into %inserted_slice_320[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_329 = tensor.extract_slice %267#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_330 = tensor.extract_slice %extracted_slice_297[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %296 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_330, %extracted_slice_329 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %297 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%296 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %298 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %297) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %299 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%298 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %300 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%298, %299 : tensor<1024xf32>, f32) outs(%298 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %301 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%300 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_331 = tensor.expand_shape %300 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_332 = tensor.extract_slice %extracted_slice_298[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_333 = tensor.extract_slice %inserted_slice_328[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_334 = tensor.reshape %extracted_slice_333(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %302 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_334 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_331, %301, %extracted_slice_332 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_335 = tensor.collapse_shape %302 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_336 = tensor.insert_slice %collapsed_335 into %inserted_slice_328[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_337 = tensor.extract_slice %267#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_338 = tensor.extract_slice %extracted_slice_297[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %303 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_338, %extracted_slice_337 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %304 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%303 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %305 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %304) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %306 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%305 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %307 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%305, %306 : tensor<1024xf32>, f32) outs(%305 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %308 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%307 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_339 = tensor.expand_shape %307 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_340 = tensor.extract_slice %extracted_slice_298[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_341 = tensor.extract_slice %inserted_slice_336[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_342 = tensor.reshape %extracted_slice_341(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %309 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_342 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_339, %308, %extracted_slice_340 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_343 = tensor.collapse_shape %309 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_344 = tensor.insert_slice %collapsed_343 into %inserted_slice_336[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_345 = tensor.extract_slice %267#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_346 = tensor.extract_slice %extracted_slice_297[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %310 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_346, %extracted_slice_345 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %311 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%310 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %312 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %311) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %313 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%312 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %314 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%312, %313 : tensor<1024xf32>, f32) outs(%312 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %315 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%314 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_347 = tensor.expand_shape %314 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_348 = tensor.extract_slice %extracted_slice_298[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_349 = tensor.extract_slice %inserted_slice_344[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_350 = tensor.reshape %extracted_slice_349(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %316 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_350 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_347, %315, %extracted_slice_348 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_351 = tensor.collapse_shape %316 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_352 = tensor.insert_slice %collapsed_351 into %inserted_slice_344[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_353 = tensor.extract_slice %267#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_354 = tensor.extract_slice %extracted_slice_297[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %317 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_354, %extracted_slice_353 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %318 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%317 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %319 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %318) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %320 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%319 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %321 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%319, %320 : tensor<1024xf32>, f32) outs(%319 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %322 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%321 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_355 = tensor.expand_shape %321 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_356 = tensor.extract_slice %extracted_slice_298[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_357 = tensor.extract_slice %inserted_slice_352[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_358 = tensor.reshape %extracted_slice_357(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %323 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_358 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_355, %322, %extracted_slice_356 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_359 = tensor.collapse_shape %323 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_360 = tensor.insert_slice %collapsed_359 into %inserted_slice_352[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_361 = tensor.extract_slice %267#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_362 = tensor.extract_slice %extracted_slice_297[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %324 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_362, %extracted_slice_361 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %325 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%324 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %326 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %325) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %327 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%326 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %328 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%326, %327 : tensor<1024xf32>, f32) outs(%326 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %329 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%328 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_363 = tensor.expand_shape %328 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_364 = tensor.extract_slice %extracted_slice_298[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_365 = tensor.extract_slice %inserted_slice_360[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_366 = tensor.reshape %extracted_slice_365(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %330 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_366 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_363, %329, %extracted_slice_364 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_367 = tensor.collapse_shape %330 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_368 = tensor.insert_slice %collapsed_367 into %inserted_slice_360[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_369 = tensor.extract_slice %267#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_370 = tensor.extract_slice %extracted_slice_297[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %331 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_370, %extracted_slice_369 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %332 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%331 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %333 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %332) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %334 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%333 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %335 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%333, %334 : tensor<1024xf32>, f32) outs(%333 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %336 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%335 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_371 = tensor.expand_shape %335 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_372 = tensor.extract_slice %extracted_slice_298[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_373 = tensor.extract_slice %inserted_slice_368[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_374 = tensor.reshape %extracted_slice_373(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %337 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_374 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_371, %336, %extracted_slice_372 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_375 = tensor.collapse_shape %337 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_376 = tensor.insert_slice %collapsed_375 into %inserted_slice_368[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_377 = tensor.extract_slice %267#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_378 = tensor.extract_slice %extracted_slice_297[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %338 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_378, %extracted_slice_377 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %339 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%338 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %340 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %339) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %341 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%340 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %342 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%340, %341 : tensor<1024xf32>, f32) outs(%340 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %343 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%342 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_379 = tensor.expand_shape %342 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_380 = tensor.extract_slice %extracted_slice_298[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_381 = tensor.extract_slice %inserted_slice_376[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_382 = tensor.reshape %extracted_slice_381(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %344 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_382 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_379, %343, %extracted_slice_380 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_383 = tensor.collapse_shape %344 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_384 = tensor.insert_slice %collapsed_383 into %inserted_slice_376[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_385 = tensor.extract_slice %267#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_386 = tensor.extract_slice %extracted_slice_297[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %345 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_386, %extracted_slice_385 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %346 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%345 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %347 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %346) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %348 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%347 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %349 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%347, %348 : tensor<1024xf32>, f32) outs(%347 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %350 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%349 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_387 = tensor.expand_shape %349 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_388 = tensor.extract_slice %extracted_slice_298[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_389 = tensor.extract_slice %inserted_slice_384[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_390 = tensor.reshape %extracted_slice_389(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %351 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_390 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_387, %350, %extracted_slice_388 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_391 = tensor.collapse_shape %351 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_392 = tensor.insert_slice %collapsed_391 into %inserted_slice_384[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_393 = tensor.extract_slice %267#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_394 = tensor.extract_slice %extracted_slice_297[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %352 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_394, %extracted_slice_393 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %353 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%352 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %354 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %353) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %355 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%354 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %356 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%354, %355 : tensor<1024xf32>, f32) outs(%354 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %357 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%356 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_395 = tensor.expand_shape %356 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_396 = tensor.extract_slice %extracted_slice_298[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_397 = tensor.extract_slice %inserted_slice_392[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_398 = tensor.reshape %extracted_slice_397(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %358 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_398 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_395, %357, %extracted_slice_396 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_399 = tensor.collapse_shape %358 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_400 = tensor.insert_slice %collapsed_399 into %inserted_slice_392[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_401 = tensor.extract_slice %267#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_402 = tensor.extract_slice %extracted_slice_297[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %359 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_402, %extracted_slice_401 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %360 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%359 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %361 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %360) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %362 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%361 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %363 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%361, %362 : tensor<1024xf32>, f32) outs(%361 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %364 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%363 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_403 = tensor.expand_shape %363 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_404 = tensor.extract_slice %extracted_slice_298[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_405 = tensor.extract_slice %inserted_slice_400[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_406 = tensor.reshape %extracted_slice_405(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %365 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_406 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_403, %364, %extracted_slice_404 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_407 = tensor.collapse_shape %365 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_408 = tensor.insert_slice %collapsed_407 into %inserted_slice_400[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_409 = tensor.extract_slice %267#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_410 = tensor.extract_slice %extracted_slice_297[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %366 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_410, %extracted_slice_409 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %367 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%366 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %368 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %367) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %369 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%368 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %370 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%368, %369 : tensor<1024xf32>, f32) outs(%368 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %371 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%370 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_411 = tensor.expand_shape %370 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_412 = tensor.extract_slice %extracted_slice_298[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_413 = tensor.extract_slice %inserted_slice_408[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_414 = tensor.reshape %extracted_slice_413(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %372 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_414 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_411, %371, %extracted_slice_412 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_415 = tensor.collapse_shape %372 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_416 = tensor.insert_slice %collapsed_415 into %inserted_slice_408[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_417 = tensor.extract_slice %267#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_418 = tensor.extract_slice %extracted_slice_297[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %373 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_418, %extracted_slice_417 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %374 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%373 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %375 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %374) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %376 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%375 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %377 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%375, %376 : tensor<1024xf32>, f32) outs(%375 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %378 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%377 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_419 = tensor.expand_shape %377 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_420 = tensor.extract_slice %extracted_slice_298[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_421 = tensor.extract_slice %inserted_slice_416[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_422 = tensor.reshape %extracted_slice_421(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %379 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_422 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_419, %378, %extracted_slice_420 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_423 = tensor.collapse_shape %379 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_424 = tensor.insert_slice %collapsed_423 into %inserted_slice_416[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %380 = bufferization.materialize_in_destination %inserted_slice_424 in %263 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_425 = tensor.extract_slice %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %381 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_425, %380 : tensor<768x768xf32>, tensor<768xf32>) outs(%260#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.addf %out, %779 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_426 = tensor.extract_slice %arg13[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %382 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%381 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %383 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %382, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %384 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%381, %383, %extracted_slice_426 : tensor<768xf32>, f32, tensor<768xf32>) outs(%380 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %385 = bufferization.materialize_in_destination %384 in %380 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_427 = tensor.extract_slice %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_428 = tensor.extract_slice %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %386 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_427, %385 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %387 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_428, %385 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %extracted_slice_429 = tensor.extract_slice %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %388:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_429, %387 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%386, %381 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32, %out_856: f32):
        %779 = arith.negf %out : f32
        %780 = math.exp %779 : f32
        %781 = arith.addf %780, %cst_6 : f32
        %782 = arith.divf %cst_6, %781 : f32
        %783 = arith.mulf %out, %782 : f32
        %784 = arith.mulf %783, %in_855 : f32
        %785 = arith.mulf %in, %784 : f32
        %786 = arith.addf %out_856, %785 : f32
        linalg.yield %784, %786 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %778#0, %778#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %extracted_slice_430 = tensor.extract_slice %arg5[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %389 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%388#1 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %390 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %389, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %391 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%388#1, %390, %extracted_slice_430 : tensor<768xf32>, f32, tensor<768xf32>) outs(%3 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_431 = tensor.extract_slice %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_432 = tensor.extract_slice %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_433 = tensor.extract_slice %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %392 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_431, %391 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_434 = tensor.extract_slice %inserted_slice_296[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %393 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_434 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_432, %391 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_435 = tensor.extract_slice %inserted_slice_295[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %394 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_435 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_433, %391 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %inserted_slice_436 = tensor.insert_slice %394 into %inserted_slice_295[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %395:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %392, %arg18 = %393) -> (tensor<768xf32>, tensor<768xf32>) {
        %779 = arith.remui %arg16, %c48 : index
        %780 = arith.index_cast %779 : index to i64
        %781 = arith.uitofp %780 : i64 to f32
        %782 = arith.divf %781, %cst_7 : f32
        %783 = math.powf %cst_8, %782 : f32
        %784 = arith.divf %cst_6, %783 : f32
        %785 = arith.mulf %8#0, %784 : f32
        %786 = math.cos %785 : f32
        %787 = math.sin %785 : f32
        %788 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_855 = tensor.extract %arg17[%788] : tensor<768xf32>
        %789 = arith.mulf %extracted, %786 : f32
        %790 = arith.mulf %extracted_855, %787 : f32
        %791 = arith.subf %789, %790 : f32
        %inserted = tensor.insert %791 into %arg17[%arg16] : tensor<768xf32>
        %792 = arith.mulf %extracted, %787 : f32
        %793 = arith.mulf %extracted_855, %786 : f32
        %794 = arith.addf %792, %793 : f32
        %inserted_856 = tensor.insert %794 into %inserted[%788] : tensor<768xf32>
        %795 = bufferization.materialize_in_destination %inserted_856 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %796 = arith.cmpi ult, %arg16, %c768 : index
        %797 = scf.if %796 -> (tensor<768xf32>) {
          %extracted_857 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_858 = tensor.extract %arg18[%788] : tensor<768xf32>
          %798 = arith.mulf %extracted_857, %786 : f32
          %799 = arith.mulf %extracted_858, %787 : f32
          %800 = arith.subf %798, %799 : f32
          %inserted_859 = tensor.insert %800 into %arg18[%arg16] : tensor<768xf32>
          %801 = arith.mulf %extracted_857, %787 : f32
          %802 = arith.mulf %extracted_858, %786 : f32
          %803 = arith.addf %801, %802 : f32
          %inserted_860 = tensor.insert %803 into %inserted_859[%788] : tensor<768xf32>
          %804 = bufferization.materialize_in_destination %inserted_860 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %804 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %795, %797 : tensor<768xf32>, tensor<768xf32>
      }
      cinm.yield %778#0, %778#1 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_437 = tensor.insert_slice %395#1 into %inserted_slice_296[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_438 = tensor.extract_slice %inserted_slice_437[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_439 = tensor.extract_slice %inserted_slice_436[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_440 = tensor.extract_slice %395#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_441 = tensor.extract_slice %extracted_slice_438[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %396 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_441, %extracted_slice_440 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %397 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%396 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %398 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %397) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %399 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%398 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %400 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%398, %399 : tensor<1024xf32>, f32) outs(%398 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %401 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%400 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_442 = tensor.expand_shape %400 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_443 = tensor.extract_slice %extracted_slice_439[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %402 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_442, %401, %extracted_slice_443 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_444 = tensor.collapse_shape %402 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_445 = tensor.insert_slice %collapsed_444 into %391[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_446 = tensor.extract_slice %395#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_447 = tensor.extract_slice %extracted_slice_438[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %403 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_447, %extracted_slice_446 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %404 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%403 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %405 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %404) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %406 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%405 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %407 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%405, %406 : tensor<1024xf32>, f32) outs(%405 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %408 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%407 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_448 = tensor.expand_shape %407 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_449 = tensor.extract_slice %extracted_slice_439[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_450 = tensor.extract_slice %inserted_slice_445[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_451 = tensor.reshape %extracted_slice_450(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %409 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_451 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_448, %408, %extracted_slice_449 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_452 = tensor.collapse_shape %409 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_453 = tensor.insert_slice %collapsed_452 into %inserted_slice_445[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_454 = tensor.extract_slice %395#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_455 = tensor.extract_slice %extracted_slice_438[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %410 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_455, %extracted_slice_454 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %411 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%410 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %412 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %411) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %413 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%412 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %414 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%412, %413 : tensor<1024xf32>, f32) outs(%412 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %415 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%414 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_456 = tensor.expand_shape %414 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_457 = tensor.extract_slice %extracted_slice_439[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_458 = tensor.extract_slice %inserted_slice_453[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_459 = tensor.reshape %extracted_slice_458(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %416 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_459 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_456, %415, %extracted_slice_457 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_460 = tensor.collapse_shape %416 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_461 = tensor.insert_slice %collapsed_460 into %inserted_slice_453[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_462 = tensor.extract_slice %395#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_463 = tensor.extract_slice %extracted_slice_438[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %417 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_463, %extracted_slice_462 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %418 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%417 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %419 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %418) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %420 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%419 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %421 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%419, %420 : tensor<1024xf32>, f32) outs(%419 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %422 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%421 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_464 = tensor.expand_shape %421 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_465 = tensor.extract_slice %extracted_slice_439[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_466 = tensor.extract_slice %inserted_slice_461[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_467 = tensor.reshape %extracted_slice_466(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %423 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_467 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_464, %422, %extracted_slice_465 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_468 = tensor.collapse_shape %423 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_469 = tensor.insert_slice %collapsed_468 into %inserted_slice_461[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_470 = tensor.extract_slice %395#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_471 = tensor.extract_slice %extracted_slice_438[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %424 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_471, %extracted_slice_470 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %425 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%424 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %426 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %425) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %427 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%426 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %428 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%426, %427 : tensor<1024xf32>, f32) outs(%426 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %429 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%428 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_472 = tensor.expand_shape %428 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_473 = tensor.extract_slice %extracted_slice_439[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_474 = tensor.extract_slice %inserted_slice_469[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_475 = tensor.reshape %extracted_slice_474(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %430 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_475 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_472, %429, %extracted_slice_473 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_476 = tensor.collapse_shape %430 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_477 = tensor.insert_slice %collapsed_476 into %inserted_slice_469[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_478 = tensor.extract_slice %395#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_479 = tensor.extract_slice %extracted_slice_438[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %431 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_479, %extracted_slice_478 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %432 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%431 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %433 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %432) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %434 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%433 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %435 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%433, %434 : tensor<1024xf32>, f32) outs(%433 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %436 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%435 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_480 = tensor.expand_shape %435 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_481 = tensor.extract_slice %extracted_slice_439[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_482 = tensor.extract_slice %inserted_slice_477[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_483 = tensor.reshape %extracted_slice_482(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %437 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_483 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_480, %436, %extracted_slice_481 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_484 = tensor.collapse_shape %437 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_485 = tensor.insert_slice %collapsed_484 into %inserted_slice_477[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_486 = tensor.extract_slice %395#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_487 = tensor.extract_slice %extracted_slice_438[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %438 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_487, %extracted_slice_486 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %439 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%438 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %440 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %439) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %441 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%440 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %442 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%440, %441 : tensor<1024xf32>, f32) outs(%440 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %443 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%442 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_488 = tensor.expand_shape %442 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_489 = tensor.extract_slice %extracted_slice_439[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_490 = tensor.extract_slice %inserted_slice_485[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_491 = tensor.reshape %extracted_slice_490(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %444 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_491 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_488, %443, %extracted_slice_489 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_492 = tensor.collapse_shape %444 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_493 = tensor.insert_slice %collapsed_492 into %inserted_slice_485[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_494 = tensor.extract_slice %395#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_495 = tensor.extract_slice %extracted_slice_438[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %445 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_495, %extracted_slice_494 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %446 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%445 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %447 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %446) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %448 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%447 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %449 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%447, %448 : tensor<1024xf32>, f32) outs(%447 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %450 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%449 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_496 = tensor.expand_shape %449 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_497 = tensor.extract_slice %extracted_slice_439[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_498 = tensor.extract_slice %inserted_slice_493[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_499 = tensor.reshape %extracted_slice_498(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %451 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_499 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_496, %450, %extracted_slice_497 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_500 = tensor.collapse_shape %451 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_501 = tensor.insert_slice %collapsed_500 into %inserted_slice_493[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_502 = tensor.extract_slice %395#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_503 = tensor.extract_slice %extracted_slice_438[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %452 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_503, %extracted_slice_502 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %453 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%452 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %454 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %453) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %455 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%454 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %456 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%454, %455 : tensor<1024xf32>, f32) outs(%454 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %457 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%456 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_504 = tensor.expand_shape %456 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_505 = tensor.extract_slice %extracted_slice_439[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_506 = tensor.extract_slice %inserted_slice_501[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_507 = tensor.reshape %extracted_slice_506(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %458 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_507 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_504, %457, %extracted_slice_505 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_508 = tensor.collapse_shape %458 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_509 = tensor.insert_slice %collapsed_508 into %inserted_slice_501[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_510 = tensor.extract_slice %395#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_511 = tensor.extract_slice %extracted_slice_438[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %459 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_511, %extracted_slice_510 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %460 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%459 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %461 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %460) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %462 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%461 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %463 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%461, %462 : tensor<1024xf32>, f32) outs(%461 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %464 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%463 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_512 = tensor.expand_shape %463 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_513 = tensor.extract_slice %extracted_slice_439[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_514 = tensor.extract_slice %inserted_slice_509[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_515 = tensor.reshape %extracted_slice_514(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %465 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_515 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_512, %464, %extracted_slice_513 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_516 = tensor.collapse_shape %465 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_517 = tensor.insert_slice %collapsed_516 into %inserted_slice_509[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_518 = tensor.extract_slice %395#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_519 = tensor.extract_slice %extracted_slice_438[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %466 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_519, %extracted_slice_518 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %467 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%466 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %468 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %467) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %469 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%468 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %470 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%468, %469 : tensor<1024xf32>, f32) outs(%468 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %471 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%470 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_520 = tensor.expand_shape %470 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_521 = tensor.extract_slice %extracted_slice_439[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_522 = tensor.extract_slice %inserted_slice_517[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_523 = tensor.reshape %extracted_slice_522(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %472 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_523 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_520, %471, %extracted_slice_521 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_524 = tensor.collapse_shape %472 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_525 = tensor.insert_slice %collapsed_524 into %inserted_slice_517[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_526 = tensor.extract_slice %395#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_527 = tensor.extract_slice %extracted_slice_438[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %473 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_527, %extracted_slice_526 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %474 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%473 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %475 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %474) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %476 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%475 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %477 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%475, %476 : tensor<1024xf32>, f32) outs(%475 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %478 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%477 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_528 = tensor.expand_shape %477 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_529 = tensor.extract_slice %extracted_slice_439[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_530 = tensor.extract_slice %inserted_slice_525[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_531 = tensor.reshape %extracted_slice_530(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %479 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_531 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_528, %478, %extracted_slice_529 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_532 = tensor.collapse_shape %479 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_533 = tensor.insert_slice %collapsed_532 into %inserted_slice_525[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_534 = tensor.extract_slice %395#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_535 = tensor.extract_slice %extracted_slice_438[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %480 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_535, %extracted_slice_534 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %481 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%480 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %482 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %481) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %483 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%482 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %484 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%482, %483 : tensor<1024xf32>, f32) outs(%482 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %485 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%484 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_536 = tensor.expand_shape %484 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_537 = tensor.extract_slice %extracted_slice_439[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_538 = tensor.extract_slice %inserted_slice_533[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_539 = tensor.reshape %extracted_slice_538(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %486 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_539 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_536, %485, %extracted_slice_537 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_540 = tensor.collapse_shape %486 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_541 = tensor.insert_slice %collapsed_540 into %inserted_slice_533[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_542 = tensor.extract_slice %395#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_543 = tensor.extract_slice %extracted_slice_438[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %487 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_543, %extracted_slice_542 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %488 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%487 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %489 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %488) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %490 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%489 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %491 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%489, %490 : tensor<1024xf32>, f32) outs(%489 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %492 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%491 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_544 = tensor.expand_shape %491 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_545 = tensor.extract_slice %extracted_slice_439[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_546 = tensor.extract_slice %inserted_slice_541[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_547 = tensor.reshape %extracted_slice_546(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %493 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_547 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_544, %492, %extracted_slice_545 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_548 = tensor.collapse_shape %493 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_549 = tensor.insert_slice %collapsed_548 into %inserted_slice_541[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_550 = tensor.extract_slice %395#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_551 = tensor.extract_slice %extracted_slice_438[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %494 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_551, %extracted_slice_550 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %495 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%494 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %496 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %495) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %497 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%496 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %498 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%496, %497 : tensor<1024xf32>, f32) outs(%496 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %499 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%498 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_552 = tensor.expand_shape %498 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_553 = tensor.extract_slice %extracted_slice_439[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_554 = tensor.extract_slice %inserted_slice_549[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_555 = tensor.reshape %extracted_slice_554(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %500 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_555 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_552, %499, %extracted_slice_553 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_556 = tensor.collapse_shape %500 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_557 = tensor.insert_slice %collapsed_556 into %inserted_slice_549[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_558 = tensor.extract_slice %395#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_559 = tensor.extract_slice %extracted_slice_438[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %501 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_559, %extracted_slice_558 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %502 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%501 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %503 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %502) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %504 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%503 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %505 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%503, %504 : tensor<1024xf32>, f32) outs(%503 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %506 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%505 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_560 = tensor.expand_shape %505 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_561 = tensor.extract_slice %extracted_slice_439[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_562 = tensor.extract_slice %inserted_slice_557[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_563 = tensor.reshape %extracted_slice_562(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %507 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_563 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_560, %506, %extracted_slice_561 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_564 = tensor.collapse_shape %507 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_565 = tensor.insert_slice %collapsed_564 into %inserted_slice_557[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %508 = bufferization.materialize_in_destination %inserted_slice_565 in %391 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_566 = tensor.extract_slice %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %509 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_566, %508 : tensor<768x768xf32>, tensor<768xf32>) outs(%388#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.addf %out, %779 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_567 = tensor.extract_slice %arg13[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %510 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%509 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %511 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %510, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %512 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%509, %511, %extracted_slice_567 : tensor<768xf32>, f32, tensor<768xf32>) outs(%508 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %513 = bufferization.materialize_in_destination %512 in %508 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_568 = tensor.extract_slice %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_569 = tensor.extract_slice %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %514 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_568, %513 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %515 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_569, %513 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %extracted_slice_570 = tensor.extract_slice %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %516:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_570, %515 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%514, %509 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32, %out_856: f32):
        %779 = arith.negf %out : f32
        %780 = math.exp %779 : f32
        %781 = arith.addf %780, %cst_6 : f32
        %782 = arith.divf %cst_6, %781 : f32
        %783 = arith.mulf %out, %782 : f32
        %784 = arith.mulf %783, %in_855 : f32
        %785 = arith.mulf %in, %784 : f32
        %786 = arith.addf %out_856, %785 : f32
        linalg.yield %784, %786 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %778#0, %778#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %extracted_slice_571 = tensor.extract_slice %arg5[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %517 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%516#1 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %518 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %517, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %519 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%516#1, %518, %extracted_slice_571 : tensor<768xf32>, f32, tensor<768xf32>) outs(%3 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_572 = tensor.extract_slice %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_573 = tensor.extract_slice %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_574 = tensor.extract_slice %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %520 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_572, %519 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_575 = tensor.extract_slice %inserted_slice_437[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %521 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_575 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_573, %519 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_576 = tensor.extract_slice %inserted_slice_436[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %522 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_576 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_574, %519 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %inserted_slice_577 = tensor.insert_slice %522 into %inserted_slice_436[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %523:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %520, %arg18 = %521) -> (tensor<768xf32>, tensor<768xf32>) {
        %779 = arith.remui %arg16, %c48 : index
        %780 = arith.index_cast %779 : index to i64
        %781 = arith.uitofp %780 : i64 to f32
        %782 = arith.divf %781, %cst_7 : f32
        %783 = math.powf %cst_8, %782 : f32
        %784 = arith.divf %cst_6, %783 : f32
        %785 = arith.mulf %8#0, %784 : f32
        %786 = math.cos %785 : f32
        %787 = math.sin %785 : f32
        %788 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_855 = tensor.extract %arg17[%788] : tensor<768xf32>
        %789 = arith.mulf %extracted, %786 : f32
        %790 = arith.mulf %extracted_855, %787 : f32
        %791 = arith.subf %789, %790 : f32
        %inserted = tensor.insert %791 into %arg17[%arg16] : tensor<768xf32>
        %792 = arith.mulf %extracted, %787 : f32
        %793 = arith.mulf %extracted_855, %786 : f32
        %794 = arith.addf %792, %793 : f32
        %inserted_856 = tensor.insert %794 into %inserted[%788] : tensor<768xf32>
        %795 = bufferization.materialize_in_destination %inserted_856 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %796 = arith.cmpi ult, %arg16, %c768 : index
        %797 = scf.if %796 -> (tensor<768xf32>) {
          %extracted_857 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_858 = tensor.extract %arg18[%788] : tensor<768xf32>
          %798 = arith.mulf %extracted_857, %786 : f32
          %799 = arith.mulf %extracted_858, %787 : f32
          %800 = arith.subf %798, %799 : f32
          %inserted_859 = tensor.insert %800 into %arg18[%arg16] : tensor<768xf32>
          %801 = arith.mulf %extracted_857, %787 : f32
          %802 = arith.mulf %extracted_858, %786 : f32
          %803 = arith.addf %801, %802 : f32
          %inserted_860 = tensor.insert %803 into %inserted_859[%788] : tensor<768xf32>
          %804 = bufferization.materialize_in_destination %inserted_860 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %804 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %795, %797 : tensor<768xf32>, tensor<768xf32>
      }
      cinm.yield %778#0, %778#1 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_578 = tensor.insert_slice %523#1 into %inserted_slice_437[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_579 = tensor.extract_slice %inserted_slice_578[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_580 = tensor.extract_slice %inserted_slice_577[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_581 = tensor.extract_slice %523#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_582 = tensor.extract_slice %extracted_slice_579[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %524 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_582, %extracted_slice_581 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %525 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%524 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %526 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %525) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %527 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%526 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %528 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%526, %527 : tensor<1024xf32>, f32) outs(%526 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %529 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%528 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_583 = tensor.expand_shape %528 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_584 = tensor.extract_slice %extracted_slice_580[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %530 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_583, %529, %extracted_slice_584 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_585 = tensor.collapse_shape %530 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_586 = tensor.insert_slice %collapsed_585 into %519[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_587 = tensor.extract_slice %523#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_588 = tensor.extract_slice %extracted_slice_579[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %531 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_588, %extracted_slice_587 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %532 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%531 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %533 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %532) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %534 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%533 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %535 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%533, %534 : tensor<1024xf32>, f32) outs(%533 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %536 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%535 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_589 = tensor.expand_shape %535 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_590 = tensor.extract_slice %extracted_slice_580[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_591 = tensor.extract_slice %inserted_slice_586[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_592 = tensor.reshape %extracted_slice_591(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %537 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_592 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_589, %536, %extracted_slice_590 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_593 = tensor.collapse_shape %537 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_594 = tensor.insert_slice %collapsed_593 into %inserted_slice_586[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_595 = tensor.extract_slice %523#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_596 = tensor.extract_slice %extracted_slice_579[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %538 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_596, %extracted_slice_595 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %539 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%538 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %540 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %539) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %541 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%540 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %542 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%540, %541 : tensor<1024xf32>, f32) outs(%540 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %543 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%542 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_597 = tensor.expand_shape %542 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_598 = tensor.extract_slice %extracted_slice_580[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_599 = tensor.extract_slice %inserted_slice_594[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_600 = tensor.reshape %extracted_slice_599(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %544 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_600 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_597, %543, %extracted_slice_598 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_601 = tensor.collapse_shape %544 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_602 = tensor.insert_slice %collapsed_601 into %inserted_slice_594[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_603 = tensor.extract_slice %523#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_604 = tensor.extract_slice %extracted_slice_579[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %545 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_604, %extracted_slice_603 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %546 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%545 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %547 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %546) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %548 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%547 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %549 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%547, %548 : tensor<1024xf32>, f32) outs(%547 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %550 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%549 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_605 = tensor.expand_shape %549 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_606 = tensor.extract_slice %extracted_slice_580[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_607 = tensor.extract_slice %inserted_slice_602[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_608 = tensor.reshape %extracted_slice_607(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %551 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_608 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_605, %550, %extracted_slice_606 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_609 = tensor.collapse_shape %551 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_610 = tensor.insert_slice %collapsed_609 into %inserted_slice_602[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_611 = tensor.extract_slice %523#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_612 = tensor.extract_slice %extracted_slice_579[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %552 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_612, %extracted_slice_611 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %553 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%552 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %554 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %553) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %555 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%554 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %556 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%554, %555 : tensor<1024xf32>, f32) outs(%554 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %557 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%556 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_613 = tensor.expand_shape %556 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_614 = tensor.extract_slice %extracted_slice_580[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_615 = tensor.extract_slice %inserted_slice_610[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_616 = tensor.reshape %extracted_slice_615(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %558 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_616 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_613, %557, %extracted_slice_614 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_617 = tensor.collapse_shape %558 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_618 = tensor.insert_slice %collapsed_617 into %inserted_slice_610[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_619 = tensor.extract_slice %523#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_620 = tensor.extract_slice %extracted_slice_579[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %559 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_620, %extracted_slice_619 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %560 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%559 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %561 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %560) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %562 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%561 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %563 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%561, %562 : tensor<1024xf32>, f32) outs(%561 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %564 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%563 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_621 = tensor.expand_shape %563 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_622 = tensor.extract_slice %extracted_slice_580[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_623 = tensor.extract_slice %inserted_slice_618[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_624 = tensor.reshape %extracted_slice_623(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %565 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_624 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_621, %564, %extracted_slice_622 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_625 = tensor.collapse_shape %565 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_626 = tensor.insert_slice %collapsed_625 into %inserted_slice_618[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_627 = tensor.extract_slice %523#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_628 = tensor.extract_slice %extracted_slice_579[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %566 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_628, %extracted_slice_627 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %567 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%566 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %568 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %567) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %569 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%568 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %570 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%568, %569 : tensor<1024xf32>, f32) outs(%568 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %571 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%570 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_629 = tensor.expand_shape %570 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_630 = tensor.extract_slice %extracted_slice_580[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_631 = tensor.extract_slice %inserted_slice_626[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_632 = tensor.reshape %extracted_slice_631(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %572 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_632 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_629, %571, %extracted_slice_630 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_633 = tensor.collapse_shape %572 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_634 = tensor.insert_slice %collapsed_633 into %inserted_slice_626[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_635 = tensor.extract_slice %523#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_636 = tensor.extract_slice %extracted_slice_579[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %573 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_636, %extracted_slice_635 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %574 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%573 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %575 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %574) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %576 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%575 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %577 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%575, %576 : tensor<1024xf32>, f32) outs(%575 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %578 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%577 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_637 = tensor.expand_shape %577 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_638 = tensor.extract_slice %extracted_slice_580[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_639 = tensor.extract_slice %inserted_slice_634[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_640 = tensor.reshape %extracted_slice_639(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %579 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_640 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_637, %578, %extracted_slice_638 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_641 = tensor.collapse_shape %579 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_642 = tensor.insert_slice %collapsed_641 into %inserted_slice_634[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_643 = tensor.extract_slice %523#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_644 = tensor.extract_slice %extracted_slice_579[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %580 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_644, %extracted_slice_643 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %581 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%580 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %582 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %581) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %583 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%582 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %584 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%582, %583 : tensor<1024xf32>, f32) outs(%582 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %585 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%584 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_645 = tensor.expand_shape %584 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_646 = tensor.extract_slice %extracted_slice_580[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_647 = tensor.extract_slice %inserted_slice_642[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_648 = tensor.reshape %extracted_slice_647(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %586 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_648 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_645, %585, %extracted_slice_646 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_649 = tensor.collapse_shape %586 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_650 = tensor.insert_slice %collapsed_649 into %inserted_slice_642[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_651 = tensor.extract_slice %523#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_652 = tensor.extract_slice %extracted_slice_579[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %587 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_652, %extracted_slice_651 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %588 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%587 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %589 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %588) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %590 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%589 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %591 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%589, %590 : tensor<1024xf32>, f32) outs(%589 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %592 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%591 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_653 = tensor.expand_shape %591 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_654 = tensor.extract_slice %extracted_slice_580[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_655 = tensor.extract_slice %inserted_slice_650[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_656 = tensor.reshape %extracted_slice_655(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %593 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_656 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_653, %592, %extracted_slice_654 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_657 = tensor.collapse_shape %593 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_658 = tensor.insert_slice %collapsed_657 into %inserted_slice_650[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_659 = tensor.extract_slice %523#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_660 = tensor.extract_slice %extracted_slice_579[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %594 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_660, %extracted_slice_659 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %595 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%594 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %596 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %595) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %597 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%596 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %598 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%596, %597 : tensor<1024xf32>, f32) outs(%596 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %599 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%598 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_661 = tensor.expand_shape %598 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_662 = tensor.extract_slice %extracted_slice_580[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_663 = tensor.extract_slice %inserted_slice_658[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_664 = tensor.reshape %extracted_slice_663(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %600 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_664 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_661, %599, %extracted_slice_662 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_665 = tensor.collapse_shape %600 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_666 = tensor.insert_slice %collapsed_665 into %inserted_slice_658[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_667 = tensor.extract_slice %523#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_668 = tensor.extract_slice %extracted_slice_579[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %601 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_668, %extracted_slice_667 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %602 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%601 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %603 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %602) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %604 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%603 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %605 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%603, %604 : tensor<1024xf32>, f32) outs(%603 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %606 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%605 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_669 = tensor.expand_shape %605 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_670 = tensor.extract_slice %extracted_slice_580[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_671 = tensor.extract_slice %inserted_slice_666[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_672 = tensor.reshape %extracted_slice_671(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %607 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_672 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_669, %606, %extracted_slice_670 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_673 = tensor.collapse_shape %607 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_674 = tensor.insert_slice %collapsed_673 into %inserted_slice_666[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_675 = tensor.extract_slice %523#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_676 = tensor.extract_slice %extracted_slice_579[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %608 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_676, %extracted_slice_675 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %609 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%608 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %610 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %609) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %611 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%610 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %612 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%610, %611 : tensor<1024xf32>, f32) outs(%610 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %613 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%612 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_677 = tensor.expand_shape %612 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_678 = tensor.extract_slice %extracted_slice_580[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_679 = tensor.extract_slice %inserted_slice_674[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_680 = tensor.reshape %extracted_slice_679(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %614 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_680 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_677, %613, %extracted_slice_678 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_681 = tensor.collapse_shape %614 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_682 = tensor.insert_slice %collapsed_681 into %inserted_slice_674[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_683 = tensor.extract_slice %523#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_684 = tensor.extract_slice %extracted_slice_579[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %615 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_684, %extracted_slice_683 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %616 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%615 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %617 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %616) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %618 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%617 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %619 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%617, %618 : tensor<1024xf32>, f32) outs(%617 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %620 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%619 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_685 = tensor.expand_shape %619 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_686 = tensor.extract_slice %extracted_slice_580[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_687 = tensor.extract_slice %inserted_slice_682[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_688 = tensor.reshape %extracted_slice_687(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %621 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_688 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_685, %620, %extracted_slice_686 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_689 = tensor.collapse_shape %621 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_690 = tensor.insert_slice %collapsed_689 into %inserted_slice_682[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_691 = tensor.extract_slice %523#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_692 = tensor.extract_slice %extracted_slice_579[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %622 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_692, %extracted_slice_691 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %623 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%622 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %624 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %623) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %625 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%624 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %626 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%624, %625 : tensor<1024xf32>, f32) outs(%624 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %627 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%626 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_693 = tensor.expand_shape %626 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_694 = tensor.extract_slice %extracted_slice_580[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_695 = tensor.extract_slice %inserted_slice_690[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_696 = tensor.reshape %extracted_slice_695(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %628 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_696 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_693, %627, %extracted_slice_694 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_697 = tensor.collapse_shape %628 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_698 = tensor.insert_slice %collapsed_697 into %inserted_slice_690[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_699 = tensor.extract_slice %523#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_700 = tensor.extract_slice %extracted_slice_579[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %629 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_700, %extracted_slice_699 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %630 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%629 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %631 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %630) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %632 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%631 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %633 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%631, %632 : tensor<1024xf32>, f32) outs(%631 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %634 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%633 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_701 = tensor.expand_shape %633 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_702 = tensor.extract_slice %extracted_slice_580[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_703 = tensor.extract_slice %inserted_slice_698[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_704 = tensor.reshape %extracted_slice_703(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %635 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_704 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_701, %634, %extracted_slice_702 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_705 = tensor.collapse_shape %635 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_706 = tensor.insert_slice %collapsed_705 into %inserted_slice_698[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %636 = bufferization.materialize_in_destination %inserted_slice_706 in %519 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_707 = tensor.extract_slice %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %637 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_707, %636 : tensor<768x768xf32>, tensor<768xf32>) outs(%516#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.addf %out, %779 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_708 = tensor.extract_slice %arg13[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %638 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%637 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %639 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %638, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %640 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%637, %639, %extracted_slice_708 : tensor<768xf32>, f32, tensor<768xf32>) outs(%636 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %641 = bufferization.materialize_in_destination %640 in %636 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_709 = tensor.extract_slice %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_710 = tensor.extract_slice %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %642 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_709, %641 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %643 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_710, %641 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %extracted_slice_711 = tensor.extract_slice %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %644:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_711, %643 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%642, %637 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32, %out_856: f32):
        %779 = arith.negf %out : f32
        %780 = math.exp %779 : f32
        %781 = arith.addf %780, %cst_6 : f32
        %782 = arith.divf %cst_6, %781 : f32
        %783 = arith.mulf %out, %782 : f32
        %784 = arith.mulf %783, %in_855 : f32
        %785 = arith.mulf %in, %784 : f32
        %786 = arith.addf %out_856, %785 : f32
        linalg.yield %784, %786 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %778#0, %778#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %extracted_slice_712 = tensor.extract_slice %arg5[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %645 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%644#1 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %646 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %645, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %647 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%644#1, %646, %extracted_slice_712 : tensor<768xf32>, f32, tensor<768xf32>) outs(%3 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_713 = tensor.extract_slice %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_714 = tensor.extract_slice %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_715 = tensor.extract_slice %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %648 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_713, %647 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_716 = tensor.extract_slice %inserted_slice_578[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %649 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_716 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_714, %647 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %extracted_slice_717 = tensor.extract_slice %inserted_slice_577[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %650 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_717 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<768xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_715, %647 : tensor<768x768xf32>, tensor<768xf32>) outs(%778 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<768xf32>
      cinm.yield %779 : tensor<768xf32>
    }
    %inserted_slice_718 = tensor.insert_slice %650 into %inserted_slice_577[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %651:2 = cinm.compute on platform #cinm.host_platform -> tensor<768xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %648, %arg18 = %649) -> (tensor<768xf32>, tensor<768xf32>) {
        %779 = arith.remui %arg16, %c48 : index
        %780 = arith.index_cast %779 : index to i64
        %781 = arith.uitofp %780 : i64 to f32
        %782 = arith.divf %781, %cst_7 : f32
        %783 = math.powf %cst_8, %782 : f32
        %784 = arith.divf %cst_6, %783 : f32
        %785 = arith.mulf %8#0, %784 : f32
        %786 = math.cos %785 : f32
        %787 = math.sin %785 : f32
        %788 = arith.addi %arg16, %c1 : index
        %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
        %extracted_855 = tensor.extract %arg17[%788] : tensor<768xf32>
        %789 = arith.mulf %extracted, %786 : f32
        %790 = arith.mulf %extracted_855, %787 : f32
        %791 = arith.subf %789, %790 : f32
        %inserted = tensor.insert %791 into %arg17[%arg16] : tensor<768xf32>
        %792 = arith.mulf %extracted, %787 : f32
        %793 = arith.mulf %extracted_855, %786 : f32
        %794 = arith.addf %792, %793 : f32
        %inserted_856 = tensor.insert %794 into %inserted[%788] : tensor<768xf32>
        %795 = bufferization.materialize_in_destination %inserted_856 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %796 = arith.cmpi ult, %arg16, %c768 : index
        %797 = scf.if %796 -> (tensor<768xf32>) {
          %extracted_857 = tensor.extract %arg18[%arg16] : tensor<768xf32>
          %extracted_858 = tensor.extract %arg18[%788] : tensor<768xf32>
          %798 = arith.mulf %extracted_857, %786 : f32
          %799 = arith.mulf %extracted_858, %787 : f32
          %800 = arith.subf %798, %799 : f32
          %inserted_859 = tensor.insert %800 into %arg18[%arg16] : tensor<768xf32>
          %801 = arith.mulf %extracted_857, %787 : f32
          %802 = arith.mulf %extracted_858, %786 : f32
          %803 = arith.addf %801, %802 : f32
          %inserted_860 = tensor.insert %803 into %inserted_859[%788] : tensor<768xf32>
          %804 = bufferization.materialize_in_destination %inserted_860 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %804 : tensor<768xf32>
        } else {
          scf.yield %arg18 : tensor<768xf32>
        }
        scf.yield %795, %797 : tensor<768xf32>, tensor<768xf32>
      }
      cinm.yield %778#0, %778#1 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_719 = tensor.insert_slice %651#1 into %inserted_slice_578[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_720 = tensor.extract_slice %inserted_slice_719[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_721 = tensor.extract_slice %inserted_slice_718[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_722 = tensor.extract_slice %651#0[0] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_723 = tensor.extract_slice %extracted_slice_720[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %652 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_723, %extracted_slice_722 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %653 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%652 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %654 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %653) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %655 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%654 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %656 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%654, %655 : tensor<1024xf32>, f32) outs(%654 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %657 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%656 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_724 = tensor.expand_shape %656 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_725 = tensor.extract_slice %extracted_slice_721[0, 0] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %658 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_724, %657, %extracted_slice_725 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_726 = tensor.collapse_shape %658 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_727 = tensor.insert_slice %collapsed_726 into %647[0] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_728 = tensor.extract_slice %651#0[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_729 = tensor.extract_slice %extracted_slice_720[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %659 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_729, %extracted_slice_728 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %660 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%659 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %661 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %660) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %662 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%661 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %663 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%661, %662 : tensor<1024xf32>, f32) outs(%661 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %664 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%663 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_730 = tensor.expand_shape %663 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_731 = tensor.extract_slice %extracted_slice_721[0, 48] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_732 = tensor.extract_slice %inserted_slice_727[48] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_733 = tensor.reshape %extracted_slice_732(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %665 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_733 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_730, %664, %extracted_slice_731 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_734 = tensor.collapse_shape %665 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_735 = tensor.insert_slice %collapsed_734 into %inserted_slice_727[48] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_736 = tensor.extract_slice %651#0[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_737 = tensor.extract_slice %extracted_slice_720[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %666 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_737, %extracted_slice_736 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %667 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%666 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %668 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %667) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %669 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%668 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %670 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%668, %669 : tensor<1024xf32>, f32) outs(%668 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %671 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%670 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_738 = tensor.expand_shape %670 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_739 = tensor.extract_slice %extracted_slice_721[0, 96] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_740 = tensor.extract_slice %inserted_slice_735[96] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_741 = tensor.reshape %extracted_slice_740(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %672 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_741 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_738, %671, %extracted_slice_739 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_742 = tensor.collapse_shape %672 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_743 = tensor.insert_slice %collapsed_742 into %inserted_slice_735[96] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_744 = tensor.extract_slice %651#0[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_745 = tensor.extract_slice %extracted_slice_720[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %673 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_745, %extracted_slice_744 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %674 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%673 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %675 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %674) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %676 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%675 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %677 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%675, %676 : tensor<1024xf32>, f32) outs(%675 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %678 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%677 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_746 = tensor.expand_shape %677 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_747 = tensor.extract_slice %extracted_slice_721[0, 144] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_748 = tensor.extract_slice %inserted_slice_743[144] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_749 = tensor.reshape %extracted_slice_748(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %679 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_749 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_746, %678, %extracted_slice_747 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_750 = tensor.collapse_shape %679 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_751 = tensor.insert_slice %collapsed_750 into %inserted_slice_743[144] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_752 = tensor.extract_slice %651#0[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_753 = tensor.extract_slice %extracted_slice_720[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %680 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_753, %extracted_slice_752 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %681 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%680 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %682 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %681) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %683 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%682 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %684 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%682, %683 : tensor<1024xf32>, f32) outs(%682 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %685 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%684 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_754 = tensor.expand_shape %684 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_755 = tensor.extract_slice %extracted_slice_721[0, 192] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_756 = tensor.extract_slice %inserted_slice_751[192] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_757 = tensor.reshape %extracted_slice_756(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %686 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_757 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_754, %685, %extracted_slice_755 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_758 = tensor.collapse_shape %686 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_759 = tensor.insert_slice %collapsed_758 into %inserted_slice_751[192] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_760 = tensor.extract_slice %651#0[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_761 = tensor.extract_slice %extracted_slice_720[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %687 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_761, %extracted_slice_760 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %688 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%687 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %689 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %688) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %690 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%689 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %691 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%689, %690 : tensor<1024xf32>, f32) outs(%689 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %692 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%691 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_762 = tensor.expand_shape %691 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_763 = tensor.extract_slice %extracted_slice_721[0, 240] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_764 = tensor.extract_slice %inserted_slice_759[240] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_765 = tensor.reshape %extracted_slice_764(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %693 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_765 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_762, %692, %extracted_slice_763 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_766 = tensor.collapse_shape %693 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_767 = tensor.insert_slice %collapsed_766 into %inserted_slice_759[240] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_768 = tensor.extract_slice %651#0[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_769 = tensor.extract_slice %extracted_slice_720[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %694 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_769, %extracted_slice_768 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %695 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%694 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %696 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %695) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %697 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%696 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %698 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%696, %697 : tensor<1024xf32>, f32) outs(%696 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %699 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%698 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_770 = tensor.expand_shape %698 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_771 = tensor.extract_slice %extracted_slice_721[0, 288] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_772 = tensor.extract_slice %inserted_slice_767[288] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_773 = tensor.reshape %extracted_slice_772(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %700 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_773 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_770, %699, %extracted_slice_771 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_774 = tensor.collapse_shape %700 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_775 = tensor.insert_slice %collapsed_774 into %inserted_slice_767[288] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_776 = tensor.extract_slice %651#0[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_777 = tensor.extract_slice %extracted_slice_720[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %701 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_777, %extracted_slice_776 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %702 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%701 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %703 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %702) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %704 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%703 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %705 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%703, %704 : tensor<1024xf32>, f32) outs(%703 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %706 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%705 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_778 = tensor.expand_shape %705 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_779 = tensor.extract_slice %extracted_slice_721[0, 336] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_780 = tensor.extract_slice %inserted_slice_775[336] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_781 = tensor.reshape %extracted_slice_780(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %707 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_781 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_778, %706, %extracted_slice_779 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_782 = tensor.collapse_shape %707 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_783 = tensor.insert_slice %collapsed_782 into %inserted_slice_775[336] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_784 = tensor.extract_slice %651#0[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_785 = tensor.extract_slice %extracted_slice_720[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %708 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_785, %extracted_slice_784 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %709 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%708 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %710 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %709) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %711 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%710 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %712 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%710, %711 : tensor<1024xf32>, f32) outs(%710 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %713 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%712 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_786 = tensor.expand_shape %712 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_787 = tensor.extract_slice %extracted_slice_721[0, 384] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_788 = tensor.extract_slice %inserted_slice_783[384] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_789 = tensor.reshape %extracted_slice_788(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %714 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_789 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_786, %713, %extracted_slice_787 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_790 = tensor.collapse_shape %714 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_791 = tensor.insert_slice %collapsed_790 into %inserted_slice_783[384] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_792 = tensor.extract_slice %651#0[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_793 = tensor.extract_slice %extracted_slice_720[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %715 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_793, %extracted_slice_792 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %716 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%715 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %717 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %716) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %718 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%717 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %719 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%717, %718 : tensor<1024xf32>, f32) outs(%717 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %720 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%719 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_794 = tensor.expand_shape %719 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_795 = tensor.extract_slice %extracted_slice_721[0, 432] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_796 = tensor.extract_slice %inserted_slice_791[432] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_797 = tensor.reshape %extracted_slice_796(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %721 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_797 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_794, %720, %extracted_slice_795 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_798 = tensor.collapse_shape %721 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_799 = tensor.insert_slice %collapsed_798 into %inserted_slice_791[432] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_800 = tensor.extract_slice %651#0[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_801 = tensor.extract_slice %extracted_slice_720[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %722 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_801, %extracted_slice_800 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %723 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%722 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %724 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %723) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %725 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%724 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %726 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%724, %725 : tensor<1024xf32>, f32) outs(%724 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %727 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%726 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_802 = tensor.expand_shape %726 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_803 = tensor.extract_slice %extracted_slice_721[0, 480] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_804 = tensor.extract_slice %inserted_slice_799[480] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_805 = tensor.reshape %extracted_slice_804(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %728 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_805 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_802, %727, %extracted_slice_803 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_806 = tensor.collapse_shape %728 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_807 = tensor.insert_slice %collapsed_806 into %inserted_slice_799[480] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_808 = tensor.extract_slice %651#0[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_809 = tensor.extract_slice %extracted_slice_720[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %729 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_809, %extracted_slice_808 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %730 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%729 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %731 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %730) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %732 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%731 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %733 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%731, %732 : tensor<1024xf32>, f32) outs(%731 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %734 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%733 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_810 = tensor.expand_shape %733 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_811 = tensor.extract_slice %extracted_slice_721[0, 528] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_812 = tensor.extract_slice %inserted_slice_807[528] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_813 = tensor.reshape %extracted_slice_812(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %735 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_813 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_810, %734, %extracted_slice_811 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_814 = tensor.collapse_shape %735 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_815 = tensor.insert_slice %collapsed_814 into %inserted_slice_807[528] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_816 = tensor.extract_slice %651#0[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_817 = tensor.extract_slice %extracted_slice_720[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %736 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_817, %extracted_slice_816 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %737 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%736 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %738 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %737) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %739 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%738 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %740 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%738, %739 : tensor<1024xf32>, f32) outs(%738 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %741 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%740 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_818 = tensor.expand_shape %740 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_819 = tensor.extract_slice %extracted_slice_721[0, 576] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_820 = tensor.extract_slice %inserted_slice_815[576] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_821 = tensor.reshape %extracted_slice_820(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %742 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_821 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_818, %741, %extracted_slice_819 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_822 = tensor.collapse_shape %742 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_823 = tensor.insert_slice %collapsed_822 into %inserted_slice_815[576] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_824 = tensor.extract_slice %651#0[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_825 = tensor.extract_slice %extracted_slice_720[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %743 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_825, %extracted_slice_824 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %744 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%743 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %745 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %744) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %746 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%745 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %747 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%745, %746 : tensor<1024xf32>, f32) outs(%745 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %748 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%747 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_826 = tensor.expand_shape %747 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_827 = tensor.extract_slice %extracted_slice_721[0, 624] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_828 = tensor.extract_slice %inserted_slice_823[624] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_829 = tensor.reshape %extracted_slice_828(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %749 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_829 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_826, %748, %extracted_slice_827 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_830 = tensor.collapse_shape %749 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_831 = tensor.insert_slice %collapsed_830 into %inserted_slice_823[624] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_832 = tensor.extract_slice %651#0[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_833 = tensor.extract_slice %extracted_slice_720[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %750 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_833, %extracted_slice_832 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %751 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%750 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %752 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %751) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %753 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%752 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %754 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%752, %753 : tensor<1024xf32>, f32) outs(%752 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %755 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%754 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_834 = tensor.expand_shape %754 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_835 = tensor.extract_slice %extracted_slice_721[0, 672] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_836 = tensor.extract_slice %inserted_slice_831[672] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_837 = tensor.reshape %extracted_slice_836(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %756 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_837 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_834, %755, %extracted_slice_835 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_838 = tensor.collapse_shape %756 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_839 = tensor.insert_slice %collapsed_838 into %inserted_slice_831[672] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %extracted_slice_840 = tensor.extract_slice %651#0[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %extracted_slice_841 = tensor.extract_slice %extracted_slice_720[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %757 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%10 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1024xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_841, %extracted_slice_840 : tensor<1024x48xf32>, tensor<48xf32>) outs(%778 : tensor<1024xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<1024xf32>
      cinm.yield %779 : tensor<1024xf32>
    }
    %758 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%757 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %779 = arith.divf %in, %cst_0 : f32
        linalg.yield %779 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %759 = cinm.compute on platform #cinm.host_platform -> tensor<1024xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = scf.for %arg16 = %9 to %c1024 step %c1 iter_args(%arg17 = %758) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_3 into %arg17[%arg16] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      cinm.yield %778 : tensor<1024xf32>
    }
    %760 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%759 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.maxnumf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %761 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%759, %760 : tensor<1024xf32>, f32) outs(%759 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.subf %in, %in_855 : f32
        %780 = math.exp %779 : f32
        linalg.yield %780 : f32
      } -> tensor<1024xf32>
      cinm.yield %778 : tensor<1024xf32>
    }
    %762 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%761 : tensor<1024xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.addf %in, %out : f32
        linalg.yield %780 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %expanded_842 = tensor.expand_shape %761 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    %extracted_slice_843 = tensor.extract_slice %extracted_slice_721[0, 720] [1024, 48] [1, 1] : tensor<1024x768xf32> to tensor<1024x48xf32>
    %extracted_slice_844 = tensor.extract_slice %inserted_slice_839[720] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %reshape_845 = tensor.reshape %extracted_slice_844(%cst_1) : (tensor<48xf32>, tensor<2xindex>) -> tensor<1x48xf32>
    %763 = cinm.compute -> tensor<1x48xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%reshape_845 : tensor<1x48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<1x48xf32>
      %779 = linalg.generic {indexing_maps = [#map6, #map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded_842, %762, %extracted_slice_843 : tensor<1x1024xf32>, f32, tensor<1024x48xf32>) outs(%778 : tensor<1x48xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %780 = arith.divf %in, %in_855 : f32
        %781 = arith.mulf %780, %in_856 : f32
        %782 = arith.addf %out, %781 : f32
        linalg.yield %782 : f32
      } -> tensor<1x48xf32>
      cinm.yield %779 : tensor<1x48xf32>
    }
    %collapsed_846 = tensor.collapse_shape %763 [[0, 1]] : tensor<1x48xf32> into tensor<48xf32>
    %inserted_slice_847 = tensor.insert_slice %collapsed_846 into %inserted_slice_839[720] [48] [1] : tensor<48xf32> into tensor<768xf32>
    %764 = bufferization.materialize_in_destination %inserted_slice_847 in %647 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_848 = tensor.extract_slice %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %765 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_848, %764 : tensor<768x768xf32>, tensor<768xf32>) outs(%644#1 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.addf %out, %779 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %extracted_slice_849 = tensor.extract_slice %arg13[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %766 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%765 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %767 = cinm.compute on platform #cinm.host_platform -> f32 attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %766, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      cinm.yield %780 : f32
    }
    %768 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%765, %767, %extracted_slice_849 : tensor<768xf32>, f32, tensor<768xf32>) outs(%764 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %out: f32):
        %779 = arith.mulf %in, %in_855 : f32
        %780 = arith.mulf %779, %in_856 : f32
        linalg.yield %780 : f32
      } -> tensor<768xf32>
      cinm.yield %778 : tensor<768xf32>
    }
    %769 = bufferization.materialize_in_destination %768 in %764 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_850 = tensor.extract_slice %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_851 = tensor.extract_slice %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %770 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_850, %769 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %771 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%129 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<2048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_851, %769 : tensor<2048x768xf32>, tensor<768xf32>) outs(%778 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
      ^bb0(%in: f32, %in_855: f32, %out: f32):
        %780 = arith.mulf %in, %in_855 : f32
        %781 = arith.addf %out, %780 : f32
        linalg.yield %781 : f32
      } -> tensor<2048xf32>
      cinm.yield %779 : tensor<2048xf32>
    }
    %extracted_slice_852 = tensor.extract_slice %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %772:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_852, %771 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%770, %765 : tensor<2048xf32>, tensor<768xf32>) {
      ^bb0(%in: f32, %in_855: f32, %out: f32, %out_856: f32):
        %779 = arith.negf %out : f32
        %780 = math.exp %779 : f32
        %781 = arith.addf %780, %cst_6 : f32
        %782 = arith.divf %cst_6, %781 : f32
        %783 = arith.mulf %out, %782 : f32
        %784 = arith.mulf %783, %in_855 : f32
        %785 = arith.mulf %in, %784 : f32
        %786 = arith.addf %out_856, %785 : f32
        linalg.yield %784, %786 : f32, f32
      } -> (tensor<2048xf32>, tensor<768xf32>)
      cinm.yield %778#0, %778#1 : tensor<2048xf32>, tensor<768xf32>
    }
    %773 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%0 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<f32>
      %779 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%772#1 : tensor<768xf32>) outs(%778 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %780 = arith.mulf %in, %in : f32
        %781 = arith.addf %780, %out : f32
        linalg.yield %781 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %779[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %774 = tensor.empty() : tensor<34048x768xf32>
    %775:2 = cinm.compute on platform #cinm.host_platform -> f32, tensor<34048x768xf32> attributes {cinm.available_platforms = [#cinm.host_platform]} {
      %778 = arith.divf %773, %cst_4 : f32
      %779 = arith.addf %778, %cst_5 : f32
      %780 = math.rsqrt %779 : f32
      %781 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%774 : tensor<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<34048x768xf32>
      cinm.yield %780, %781 : f32, tensor<34048x768xf32>
    }
    %inserted_slice_853 = tensor.insert_slice %arg15 into %775#1[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
    %776 = tensor.empty() : tensor<34048xf32>
    %777 = cinm.compute -> tensor<34048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %778 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%776 : tensor<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      } -> tensor<34048xf32>
      %779 = linalg.generic {indexing_maps = [#map3, #map4, #map10, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%inserted_slice_853, %772#1, %775#0, %arg14 : tensor<34048x768xf32>, tensor<768xf32>, f32, tensor<768xf32>) outs(%778 : tensor<34048xf32>) {
      ^bb0(%in: f32, %in_855: f32, %in_856: f32, %in_857: f32, %out: f32):
        %780 = arith.mulf %in_855, %in_856 : f32
        %781 = arith.mulf %780, %in_857 : f32
        %782 = arith.mulf %in, %781 : f32
        %783 = arith.addf %out, %782 : f32
        linalg.yield %783 : f32
      } -> tensor<34048xf32>
      cinm.yield %779 : tensor<34048xf32>
    }
    %extracted_slice_854 = tensor.extract_slice %777[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
    return %extracted_slice_854 : tensor<32000xf32>
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
