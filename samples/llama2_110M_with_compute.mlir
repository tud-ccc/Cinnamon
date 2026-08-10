#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> ()>
#upmem = #upmem.platform<type = v1A, dimensions = 40x64x24>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg14: tensor<768xf32> {bufferization.buffer_layout = #map1}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}) -> tensor<32000xf32> attributes {cinm.available_platforms = [#upmem]} {
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 6.92820311 : f32
    %c1024 = arith.constant 1024 : index
    %cst_3 = arith.constant 7.680000e+02 : f32
    %cst_4 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 4.800000e+01 : f32
    %cst_7 = arith.constant 1.000000e+04 : f32
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %0:3 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %extracted_slice, %arg18 = %arg2, %arg19 = %arg3) -> (tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>) {
      %10 = tensor.empty() : tensor<f32>
      %11 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
        %37 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %38 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg17 : tensor<768xf32>) outs(%37 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %39 = arith.mulf %in, %in : f32
          %40 = arith.addf %39, %out : f32
          linalg.yield %40 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %38[] : tensor<f32>
        cinm.yield %extracted : f32
      }
      %12 = arith.divf %11, %cst_3 : f32
      %13 = arith.addf %12, %cst_4 : f32
      %14 = math.rsqrt %13 : f32
      %15 = tensor.empty() : tensor<768xf32>
      %16 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg5[%arg16, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
        %37 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg17, %14, %extracted_slice_10 : tensor<768xf32>, f32, tensor<768xf32>) outs(%15 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_11: f32, %in_12: f32, %out: f32):
          %38 = arith.mulf %in, %in_11 : f32
          %39 = arith.mulf %38, %in_12 : f32
          linalg.yield %39 : f32
        } -> tensor<768xf32>
        cinm.yield %37 : tensor<768xf32>
      }
      %17 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg6[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
        %37 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%15 : tensor<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<768xf32>
        %38 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %16 : tensor<768x768xf32>, tensor<768xf32>) outs(%37 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %39 = arith.mulf %in, %in_11 : f32
          %40 = arith.addf %out, %39 : f32
          linalg.yield %40 : f32
        } -> tensor<768xf32>
        cinm.yield %38 : tensor<768xf32>
      }
      %18 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg7[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
        %extracted_slice_11 = tensor.extract_slice %arg18[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
        %37 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_11 : tensor<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<768xf32>
        %38 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %16 : tensor<768x768xf32>, tensor<768xf32>) outs(%37 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_12: f32, %out: f32):
          %39 = arith.mulf %in, %in_12 : f32
          %40 = arith.addf %out, %39 : f32
          linalg.yield %40 : f32
        } -> tensor<768xf32>
        cinm.yield %38 : tensor<768xf32>
      }
      %19 = cinm.compute -> tensor<6x1024x768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg8[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
        %extracted_slice_11 = tensor.extract_slice %arg19[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
        %37 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_11 : tensor<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<768xf32>
        %38 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %16 : tensor<768x768xf32>, tensor<768xf32>) outs(%37 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_13: f32, %out: f32):
          %39 = arith.mulf %in, %in_13 : f32
          %40 = arith.addf %out, %39 : f32
          linalg.yield %40 : f32
        } -> tensor<768xf32>
        %inserted_slice_12 = tensor.insert_slice %38 into %arg19[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
        cinm.yield %inserted_slice_12 : tensor<6x1024x768xf32>
      }
      %20 = arith.index_cast %arg1 : index to i64
      %21 = arith.uitofp %20 : i64 to f32
      %22:2 = scf.for %arg20 = %c0 to %c768 step %c2 iter_args(%arg21 = %17, %arg22 = %18) -> (tensor<768xf32>, tensor<768xf32>) {
        %37 = arith.remui %arg20, %c48 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = arith.uitofp %38 : i64 to f32
        %40 = arith.divf %39, %cst_6 : f32
        %41 = math.powf %cst_7, %40 : f32
        %42 = arith.divf %cst_5, %41 : f32
        %43 = arith.mulf %21, %42 : f32
        %44 = math.cos %43 : f32
        %45 = math.sin %43 : f32
        %46 = arith.addi %arg20, %c1 : index
        %extracted = tensor.extract %arg21[%arg20] : tensor<768xf32>
        %extracted_10 = tensor.extract %arg21[%46] : tensor<768xf32>
        %47 = arith.mulf %extracted, %44 : f32
        %48 = arith.mulf %extracted_10, %45 : f32
        %49 = arith.subf %47, %48 : f32
        %inserted = tensor.insert %49 into %arg21[%arg20] : tensor<768xf32>
        %50 = arith.mulf %extracted, %45 : f32
        %51 = arith.mulf %extracted_10, %44 : f32
        %52 = arith.addf %50, %51 : f32
        %inserted_11 = tensor.insert %52 into %inserted[%46] : tensor<768xf32>
        %53 = bufferization.materialize_in_destination %inserted_11 in %arg21 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        %54 = arith.cmpi ult, %arg20, %c768 : index
        %55 = scf.if %54 -> (tensor<768xf32>) {
          %extracted_12 = tensor.extract %arg22[%arg20] : tensor<768xf32>
          %extracted_13 = tensor.extract %arg22[%46] : tensor<768xf32>
          %56 = arith.mulf %extracted_12, %44 : f32
          %57 = arith.mulf %extracted_13, %45 : f32
          %58 = arith.subf %56, %57 : f32
          %inserted_14 = tensor.insert %58 into %arg22[%arg20] : tensor<768xf32>
          %59 = arith.mulf %extracted_12, %45 : f32
          %60 = arith.mulf %extracted_13, %44 : f32
          %61 = arith.addf %59, %60 : f32
          %inserted_15 = tensor.insert %61 into %inserted_14[%46] : tensor<768xf32>
          %62 = bufferization.materialize_in_destination %inserted_15 in %arg22 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
          scf.yield %62 : tensor<768xf32>
        } else {
          scf.yield %arg22 : tensor<768xf32>
        }
        scf.yield %53, %55 : tensor<768xf32>, tensor<768xf32>
      }
      %inserted_slice_9 = tensor.insert_slice %22#1 into %arg18[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %23 = arith.addi %arg1, %c1 : index
      %24 = scf.for %arg20 = %c0 to %c768 step %c48 iter_args(%arg21 = %16) -> (tensor<768xf32>) {
        %37 = tensor.empty() : tensor<1024xf32>
        %38 = scf.for %arg22 = %c0 to %23 step %c1 iter_args(%arg23 = %37) -> (tensor<1024xf32>) {
          %46 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
            %extracted_slice_12 = tensor.extract_slice %22#0[%arg20] [48] [1] : tensor<768xf32> to tensor<48xf32>
            %extracted_slice_13 = tensor.extract_slice %inserted_slice_9[%arg16, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
            %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[%arg22, %arg20] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
            %48 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_0 : f32
            } -> tensor<f32>
            %49 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_12, %extracted_slice_14 : tensor<48xf32>, tensor<48xf32>) outs(%48 : tensor<f32>) {
            ^bb0(%in: f32, %in_15: f32, %out: f32):
              %50 = arith.mulf %in, %in_15 : f32
              %51 = arith.addf %50, %out : f32
              linalg.yield %51 : f32
            } -> tensor<f32>
            %extracted = tensor.extract %49[] : tensor<f32>
            cinm.yield %extracted : f32
          }
          %47 = arith.divf %46, %cst_2 : f32
          %inserted = tensor.insert %47 into %arg23[%arg22] : tensor<1024xf32>
          scf.yield %inserted : tensor<1024xf32>
        }
        %39 = scf.for %arg22 = %23 to %c1024 step %c1 iter_args(%arg23 = %38) -> (tensor<1024xf32>) {
          %inserted = tensor.insert %cst_1 into %arg23[%arg22] : tensor<1024xf32>
          scf.yield %inserted : tensor<1024xf32>
        }
        %40 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
          %46 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %47 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%39 : tensor<1024xf32>) outs(%46 : tensor<f32>) {
          ^bb0(%in: f32, %out: f32):
            %48 = arith.maxnumf %in, %out : f32
            linalg.yield %48 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %47[] : tensor<f32>
          cinm.yield %extracted : f32
        }
        %41 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
          %46 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%39, %40 : tensor<1024xf32>, f32) outs(%39 : tensor<1024xf32>) {
          ^bb0(%in: f32, %in_12: f32, %out: f32):
            %47 = arith.subf %in, %in_12 : f32
            %48 = math.exp %47 : f32
            linalg.yield %48 : f32
          } -> tensor<1024xf32>
          cinm.yield %46 : tensor<1024xf32>
        }
        %42 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
          %46 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_0 : f32
          } -> tensor<f32>
          %47 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%41 : tensor<1024xf32>) outs(%46 : tensor<f32>) {
          ^bb0(%in: f32, %out: f32):
            %48 = arith.addf %in, %out : f32
            linalg.yield %48 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %47[] : tensor<f32>
          cinm.yield %extracted : f32
        }
        %43 = cinm.compute -> tensor<1024xf32> attributes {cinm.available_platforms = [#upmem]} {
          %46 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%41, %42 : tensor<1024xf32>, f32) outs(%39 : tensor<1024xf32>) {
          ^bb0(%in: f32, %in_12: f32, %out: f32):
            %47 = arith.divf %in, %in_12 : f32
            linalg.yield %47 : f32
          } -> tensor<1024xf32>
          cinm.yield %46 : tensor<1024xf32>
        }
        %extracted_slice_10 = tensor.extract_slice %arg21[%arg20] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %44 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_10 : tensor<48xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<48xf32>
        %inserted_slice_11 = tensor.insert_slice %44 into %arg21[%arg20] [48] [1] : tensor<48xf32> into tensor<768xf32>
        %45 = scf.for %arg22 = %c0 to %23 step %c1 iter_args(%arg23 = %inserted_slice_11) -> (tensor<768xf32>) {
          %extracted = tensor.extract %43[%arg22] : tensor<1024xf32>
          %46 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
            %extracted_slice_12 = tensor.extract_slice %arg23[%arg20] [48] [1] : tensor<768xf32> to tensor<48xf32>
            %extracted_slice_13 = tensor.extract_slice %19[%arg16, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
            %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[%arg22, %arg20] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
            %47 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_12, %extracted_slice_14, %extracted : tensor<48xf32>, tensor<48xf32>, f32) outs(%extracted_slice_12 : tensor<48xf32>) {
            ^bb0(%in: f32, %in_16: f32, %in_17: f32, %out: f32):
              %48 = arith.mulf %in_16, %in_17 : f32
              %49 = arith.addf %in, %48 : f32
              linalg.yield %49 : f32
            } -> tensor<48xf32>
            %inserted_slice_15 = tensor.insert_slice %47 into %arg23[%arg20] [48] [1] : tensor<48xf32> into tensor<768xf32>
            cinm.yield %inserted_slice_15 : tensor<768xf32>
          }
          scf.yield %46 : tensor<768xf32>
        }
        scf.yield %45 : tensor<768xf32>
      }
      %25 = bufferization.materialize_in_destination %24 in %16 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %26 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg9[%arg16, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
        %37 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %25 : tensor<768x768xf32>, tensor<768xf32>) outs(%arg17 : tensor<768xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %38 = arith.mulf %in, %in_11 : f32
          %39 = arith.addf %out, %38 : f32
          linalg.yield %39 : f32
        } -> tensor<768xf32>
        cinm.yield %37 : tensor<768xf32>
      }
      %27 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
        %37 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %38 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%26 : tensor<768xf32>) outs(%37 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %39 = arith.mulf %in, %in : f32
          %40 = arith.addf %39, %out : f32
          linalg.yield %40 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %38[] : tensor<f32>
        cinm.yield %extracted : f32
      }
      %28 = arith.divf %27, %cst_3 : f32
      %29 = arith.addf %28, %cst_4 : f32
      %30 = math.rsqrt %29 : f32
      %31 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg13[%arg16, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
        %37 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%26, %30, %extracted_slice_10 : tensor<768xf32>, f32, tensor<768xf32>) outs(%25 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_11: f32, %in_12: f32, %out: f32):
          %38 = arith.mulf %in, %in_11 : f32
          %39 = arith.mulf %38, %in_12 : f32
          linalg.yield %39 : f32
        } -> tensor<768xf32>
        cinm.yield %37 : tensor<768xf32>
      }
      %32 = bufferization.materialize_in_destination %31 in %25 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %33 = tensor.empty() : tensor<2048xf32>
      %34 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg10[%arg16, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
        %37 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%33 : tensor<2048xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<2048xf32>
        %38 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %32 : tensor<2048x768xf32>, tensor<768xf32>) outs(%37 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %39 = arith.mulf %in, %in_11 : f32
          %40 = arith.addf %out, %39 : f32
          linalg.yield %40 : f32
        } -> tensor<2048xf32>
        cinm.yield %38 : tensor<2048xf32>
      }
      %35 = cinm.compute -> tensor<2048xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg12[%arg16, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
        %37 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%33 : tensor<2048xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<2048xf32>
        %38 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %32 : tensor<2048x768xf32>, tensor<768xf32>) outs(%37 : tensor<2048xf32>) attrs =  {cinm.debug_tag = "cinm.op.gemv", linalg.memoized_indexing_maps = [#map3, #map4, #map5]} {
        ^bb0(%in: f32, %in_11: f32, %out: f32):
          %39 = arith.mulf %in, %in_11 : f32
          %40 = arith.addf %out, %39 : f32
          linalg.yield %40 : f32
        } -> tensor<2048xf32>
        cinm.yield %38 : tensor<2048xf32>
      }
      %36:2 = cinm.compute -> tensor<2048xf32>, tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %extracted_slice_10 = tensor.extract_slice %arg11[%arg16, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
        %37:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %35 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%34, %26 : tensor<2048xf32>, tensor<768xf32>) {
        ^bb0(%in: f32, %in_11: f32, %out: f32, %out_12: f32):
          %38 = arith.negf %out : f32
          %39 = math.exp %38 : f32
          %40 = arith.addf %39, %cst_5 : f32
          %41 = arith.divf %cst_5, %40 : f32
          %42 = arith.mulf %out, %41 : f32
          %43 = arith.mulf %42, %in_11 : f32
          %44 = arith.mulf %in, %43 : f32
          %45 = arith.addf %out_12, %44 : f32
          linalg.yield %43, %45 : f32, f32
        } -> (tensor<2048xf32>, tensor<768xf32>)
        cinm.yield %37#0, %37#1 : tensor<2048xf32>, tensor<768xf32>
      }
      scf.yield %36#1, %inserted_slice_9, %19 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %1 = tensor.empty() : tensor<f32>
    %2 = cinm.compute -> f32 attributes {cinm.available_platforms = [#upmem]} {
      %10 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<f32>
      %11 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%0#0 : tensor<768xf32>) outs(%10 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.mulf %in, %in : f32
        %13 = arith.addf %12, %out : f32
        linalg.yield %13 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %11[] : tensor<f32>
      cinm.yield %extracted : f32
    }
    %3 = arith.divf %2, %cst_3 : f32
    %4 = arith.addf %3, %cst_4 : f32
    %5 = math.rsqrt %4 : f32
    %6 = tensor.empty() : tensor<34048x768xf32>
    %7 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<34048x768xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    } -> tensor<34048x768xf32>
    %inserted_slice = tensor.insert_slice %arg15 into %7[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
    %8 = tensor.empty() : tensor<34048xf32>
    %9 = cinm.compute -> tensor<34048xf32> attributes {cinm.available_platforms = [#upmem]} {
      %10 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : tensor<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<34048xf32>
      %11 = linalg.generic {indexing_maps = [#map3, #map4, #map6, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%inserted_slice, %0#0, %5, %arg14 : tensor<34048x768xf32>, tensor<768xf32>, f32, tensor<768xf32>) outs(%10 : tensor<34048xf32>) {
      ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %out: f32):
        %12 = arith.mulf %in_9, %in_10 : f32
        %13 = arith.mulf %12, %in_11 : f32
        %14 = arith.mulf %in, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<34048xf32>
      cinm.yield %11 : tensor<34048xf32>
    }
    %extracted_slice_8 = tensor.extract_slice %9[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
    return %extracted_slice_8 : tensor<32000xf32>
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
    %cst = arith.constant 0xFFC00000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %c1024 = arith.constant 1024 : index
    %cst_1 = arith.constant 6.92820311 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %2 = scf.for %arg4 = %c0 to %c768 step %c48 iter_args(%arg5 = %1) -> (tensor<768xf32>) {
      %3 = tensor.empty() : tensor<1024xf32>
      %4 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %3) -> (tensor<1024xf32>) {
        %extracted_slice_4 = tensor.extract_slice %arg0[%arg4] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_5 = tensor.extract_slice %arg1[%arg6, %arg4] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %15 = tensor.empty() : tensor<f32>
        %16 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%15 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %17 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_4, %extracted_slice_5 : tensor<48xf32>, tensor<48xf32>) outs(%16 : tensor<f32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %19 = arith.mulf %in, %in_7 : f32
          %20 = arith.addf %19, %out : f32
          linalg.yield %20 : f32
        } -> tensor<f32>
        %extracted_6 = tensor.extract %17[] : tensor<f32>
        %18 = arith.divf %extracted_6, %cst_1 : f32
        %inserted = tensor.insert %18 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %5 = scf.for %arg6 = %0 to %c1024 step %c1 iter_args(%arg7 = %4) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_2 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %6 = tensor.empty() : tensor<f32>
      %7 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%6 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %8 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%5 : tensor<1024xf32>) outs(%7 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %15 = arith.maxnumf %in, %out : f32
        linalg.yield %15 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %8[] : tensor<f32>
      %9 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%5, %extracted : tensor<1024xf32>, f32) outs(%5 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %15 = arith.subf %in, %in_4 : f32
        %16 = math.exp %15 : f32
        linalg.yield %16 : f32
      } -> tensor<1024xf32>
      %10 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%6 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<f32>
      %11 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%9 : tensor<1024xf32>) outs(%10 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %15 = arith.addf %in, %out : f32
        linalg.yield %15 : f32
      } -> tensor<f32>
      %extracted_3 = tensor.extract %11[] : tensor<f32>
      %12 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%9, %extracted_3 : tensor<1024xf32>, f32) outs(%5 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %15 = arith.divf %in, %in_4 : f32
        linalg.yield %15 : f32
      } -> tensor<1024xf32>
      %extracted_slice = tensor.extract_slice %arg5[%arg4] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %13 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<48xf32>
      %inserted_slice = tensor.insert_slice %13 into %arg5[%arg4] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %14 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %inserted_slice) -> (tensor<768xf32>) {
        %extracted_slice_4 = tensor.extract_slice %arg7[%arg4] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_5 = tensor.extract_slice %arg2[%arg6, %arg4] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted_6 = tensor.extract %12[%arg6] : tensor<1024xf32>
        %15 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_4, %extracted_slice_5, %extracted_6 : tensor<48xf32>, tensor<48xf32>, f32) outs(%extracted_slice_4 : tensor<48xf32>) {
        ^bb0(%in: f32, %in_8: f32, %in_9: f32, %out: f32):
          %16 = arith.mulf %in_8, %in_9 : f32
          %17 = arith.addf %in, %16 : f32
          linalg.yield %17 : f32
        } -> tensor<48xf32>
        %inserted_slice_7 = tensor.insert_slice %15 into %arg7[%arg4] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_7 : tensor<768xf32>
      }
      scf.yield %14 : tensor<768xf32>
    }
    return %2 : tensor<768xf32>
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
      transform.yield
    }
  }
}
