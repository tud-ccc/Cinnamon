#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: index, %arg1: index, %arg2: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg3: tensor<6x1024x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, bufferization.writable = true}, %arg4: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}, %arg5: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg6: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg7: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg8: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg9: tensor<6x768x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg10: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg11: tensor<6x768x2048xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg12: tensor<6x2048x768xf32> {bufferization.buffer_layout = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}, %arg13: tensor<6x768xf32> {bufferization.buffer_layout = #map3}, %arg14: tensor<768xf32> {bufferization.buffer_layout = #map1}, %arg15: tensor<32000x768xf32> {bufferization.buffer_layout = #map3}) -> tensor<32000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 6.92820311 : f32
    %c1024 = arith.constant 1024 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 7.680000e+02 : f32
    %cst_3 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %cst_4 = arith.constant 1.000000e+00 : f32
    %cst_5 = arith.constant 4.800000e+01 : f32
    %cst_6 = arith.constant 1.000000e+04 : f32
    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>
    %extracted_slice_7 = tensor.extract_slice %arg5[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %0 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %68 = arith.mulf %in, %in : f32
        %69 = arith.addf %68, %out : f32
        linalg.yield %69 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = tensor.empty() : tensor<768xf32>
      %67 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %extracted_slice_7 : tensor<768xf32>, tensor<768xf32>) outs(%66 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %68 = arith.mulf %in, %65 : f32
        %69 = arith.mulf %68, %in_94 : f32
        linalg.yield %69 : f32
      } -> tensor<768xf32>
      cinm.yield %67 : tensor<768xf32>
    }
    %extracted_slice_8 = tensor.extract_slice %arg6[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_9 = tensor.extract_slice %arg7[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_10 = tensor.extract_slice %arg8[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %1:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %extracted_slice_94 = tensor.extract_slice %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
      %60 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_8, %0 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_97: f32, %out: f32):
        %64 = arith.mulf %in, %in_97 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_9, %0 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_97: f32, %out: f32):
        %64 = arith.mulf %in, %in_97 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_10, %0 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_97: f32, %out: f32):
        %64 = arith.mulf %in, %in_97 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %inserted_slice_95 = tensor.insert_slice %62 into %arg2[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %inserted_slice_96 = tensor.insert_slice %63 into %arg3[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %61, %inserted_slice_95, %inserted_slice_96 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %2 = arith.index_cast %arg1 : index to i64
    %3 = arith.uitofp %2 : i64 to f32
    %extracted_slice_11 = tensor.extract_slice %1#1[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %4:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %1#0, %arg18 = %extracted_slice_11) -> (tensor<768xf32>, tensor<768xf32>) {
      %60 = arith.remui %arg16, %c48 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.uitofp %61 : i64 to f32
      %63 = arith.divf %62, %cst_5 : f32
      %64 = math.powf %cst_6, %63 : f32
      %65 = arith.divf %cst_4, %64 : f32
      %66 = arith.mulf %3, %65 : f32
      %67 = math.cos %66 : f32
      %68 = math.sin %66 : f32
      %69 = arith.addi %arg16, %c1 : index
      %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_94 = tensor.extract %arg17[%69] : tensor<768xf32>
      %70 = arith.mulf %extracted, %67 : f32
      %71 = arith.mulf %extracted_94, %68 : f32
      %72 = arith.subf %70, %71 : f32
      %inserted = tensor.insert %72 into %arg17[%arg16] : tensor<768xf32>
      %73 = arith.mulf %extracted, %68 : f32
      %74 = arith.mulf %extracted_94, %67 : f32
      %75 = arith.addf %73, %74 : f32
      %inserted_95 = tensor.insert %75 into %inserted[%69] : tensor<768xf32>
      %76 = bufferization.materialize_in_destination %inserted_95 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %77 = arith.cmpi ult, %arg16, %c768 : index
      %78 = scf.if %77 -> (tensor<768xf32>) {
        %extracted_96 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_97 = tensor.extract %arg18[%69] : tensor<768xf32>
        %79 = arith.mulf %extracted_96, %67 : f32
        %80 = arith.mulf %extracted_97, %68 : f32
        %81 = arith.subf %79, %80 : f32
        %inserted_98 = tensor.insert %81 into %arg18[%arg16] : tensor<768xf32>
        %82 = arith.mulf %extracted_96, %68 : f32
        %83 = arith.mulf %extracted_97, %67 : f32
        %84 = arith.addf %82, %83 : f32
        %inserted_99 = tensor.insert %84 into %inserted_98[%69] : tensor<768xf32>
        %85 = bufferization.materialize_in_destination %inserted_99 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %85 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %76, %78 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice = tensor.insert_slice %4#1 into %1#1[0, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_12 = tensor.extract_slice %inserted_slice[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_13 = tensor.extract_slice %1#2[0, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %5 = arith.addi %arg1, %c1 : index
    %extracted_slice_14 = tensor.extract_slice %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %6 = tensor.empty() : tensor<768xf32>
    %7 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %4#0) -> (tensor<768xf32>) {
      %60 = arith.muli %arg16, %c48 : index
      %61 = tensor.empty() : tensor<1024xf32>
      %62 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %61) -> (tensor<1024xf32>) {
        %extracted_slice_96 = tensor.extract_slice %4#0[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_12[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %67 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = tensor.empty() : tensor<f32>
          %69 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%68 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %70 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%69 : tensor<f32>) {
          ^bb0(%in: f32, %in_98: f32, %out: f32):
            %72 = arith.mulf %in, %in_98 : f32
            %73 = arith.addf %72, %out : f32
            linalg.yield %73 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %70[] : tensor<f32>
          %71 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %71 : f32
        }
        %inserted = tensor.insert %67 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %63 = scf.for %arg18 = %5 to %c1024 step %c1 iter_args(%arg19 = %62) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_0 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %64 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %67 = tensor.empty() : tensor<f32>
        %68 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %69 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%68 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.maxnumf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %69[] : tensor<f32>
        %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.subf %in, %extracted : f32
          %75 = math.exp %74 : f32
          linalg.yield %75 : f32
        } -> tensor<1024xf32>
        %71 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%70 : tensor<1024xf32>) outs(%71 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.addf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted_96 = tensor.extract %72[] : tensor<f32>
        %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%70 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.divf %in, %extracted_96 : f32
          linalg.yield %74 : f32
        } -> tensor<1024xf32>
        cinm.yield %73 : tensor<1024xf32>
      }
      %extracted_slice_94 = tensor.extract_slice %arg17[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %65 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice_95 = tensor.insert_slice %65 into %arg17[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %66 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %inserted_slice_95) -> (tensor<768xf32>) {
        %extracted_slice_96 = tensor.extract_slice %arg19[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_13[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %64[%arg18] : tensor<1024xf32>
        %67 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_96 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_99: f32, %out: f32):
            %69 = arith.mulf %in_99, %extracted : f32
            %70 = arith.addf %in, %69 : f32
            linalg.yield %70 : f32
          } -> tensor<48xf32>
          cinm.yield %68 : tensor<48xf32>
        }
        %inserted_slice_98 = tensor.insert_slice %67 into %arg19[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_98 : tensor<768xf32>
      }
      scf.yield %66 : tensor<768xf32>
    }
    %8 = bufferization.materialize_in_destination %7 in %4#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_15 = tensor.extract_slice %arg9[0, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %9 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_15, %8 : tensor<768x768xf32>, tensor<768xf32>) outs(%8 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.mulf %in, %in_94 : f32
        %63 = arith.addf %out, %62 : f32
        linalg.yield %63 : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice, %60 : tensor<768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.addf %in, %in_94 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %61 : tensor<768xf32>
    }
    %extracted_slice_16 = tensor.extract_slice %arg13[0, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %10 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%9 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%9, %extracted_slice_16 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_17 = tensor.extract_slice %arg10[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_18 = tensor.extract_slice %arg12[0, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %11:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = tensor.empty() : tensor<2048xf32>
      %61 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%60 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<2048xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_17, %10 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_18, %10 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      cinm.yield %62, %63 : tensor<2048xf32>, tensor<2048xf32>
    }
    %mapped = linalg.map ins(%11#0, %11#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%11#0 : tensor<2048xf32>)
      (%in: f32, %in_94: f32) {
        %60 = arith.negf %in : f32
        %61 = math.exp %60 : f32
        %62 = arith.addf %61, %cst_4 : f32
        %63 = arith.divf %cst_4, %62 : f32
        %64 = arith.mulf %in, %63 : f32
        %65 = arith.mulf %64, %in_94 : f32
        linalg.yield %65 : f32
      }
    %extracted_slice_19 = tensor.extract_slice %arg11[0, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %12 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_19, %mapped : tensor<768x2048xf32>, tensor<2048xf32>) outs(%10 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %61 = arith.mulf %in, %in_94 : f32
        %62 = arith.addf %out, %61 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %60 : tensor<768xf32>
    }
    %extracted_slice_20 = tensor.extract_slice %arg5[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %13 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%12 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%12, %extracted_slice_20 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_21 = tensor.extract_slice %arg6[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_22 = tensor.extract_slice %arg7[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_23 = tensor.extract_slice %arg8[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %14:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_14 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_21, %13 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_22, %13 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_23, %13 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %inserted_slice_94 = tensor.insert_slice %62 into %inserted_slice[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %inserted_slice_95 = tensor.insert_slice %63 into %1#2[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %61, %inserted_slice_94, %inserted_slice_95 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %extracted_slice_24 = tensor.extract_slice %14#1[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %15:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %14#0, %arg18 = %extracted_slice_24) -> (tensor<768xf32>, tensor<768xf32>) {
      %60 = arith.remui %arg16, %c48 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.uitofp %61 : i64 to f32
      %63 = arith.divf %62, %cst_5 : f32
      %64 = math.powf %cst_6, %63 : f32
      %65 = arith.divf %cst_4, %64 : f32
      %66 = arith.mulf %3, %65 : f32
      %67 = math.cos %66 : f32
      %68 = math.sin %66 : f32
      %69 = arith.addi %arg16, %c1 : index
      %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_94 = tensor.extract %arg17[%69] : tensor<768xf32>
      %70 = arith.mulf %extracted, %67 : f32
      %71 = arith.mulf %extracted_94, %68 : f32
      %72 = arith.subf %70, %71 : f32
      %inserted = tensor.insert %72 into %arg17[%arg16] : tensor<768xf32>
      %73 = arith.mulf %extracted, %68 : f32
      %74 = arith.mulf %extracted_94, %67 : f32
      %75 = arith.addf %73, %74 : f32
      %inserted_95 = tensor.insert %75 into %inserted[%69] : tensor<768xf32>
      %76 = bufferization.materialize_in_destination %inserted_95 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %77 = arith.cmpi ult, %arg16, %c768 : index
      %78 = scf.if %77 -> (tensor<768xf32>) {
        %extracted_96 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_97 = tensor.extract %arg18[%69] : tensor<768xf32>
        %79 = arith.mulf %extracted_96, %67 : f32
        %80 = arith.mulf %extracted_97, %68 : f32
        %81 = arith.subf %79, %80 : f32
        %inserted_98 = tensor.insert %81 into %arg18[%arg16] : tensor<768xf32>
        %82 = arith.mulf %extracted_96, %68 : f32
        %83 = arith.mulf %extracted_97, %67 : f32
        %84 = arith.addf %82, %83 : f32
        %inserted_99 = tensor.insert %84 into %inserted_98[%69] : tensor<768xf32>
        %85 = bufferization.materialize_in_destination %inserted_99 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %85 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %76, %78 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_25 = tensor.insert_slice %15#1 into %14#1[1, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_26 = tensor.extract_slice %inserted_slice_25[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_27 = tensor.extract_slice %inserted_slice_25[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_28 = tensor.extract_slice %14#2[1, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %16 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %15#0) -> (tensor<768xf32>) {
      %60 = arith.muli %arg16, %c48 : index
      %61 = tensor.empty() : tensor<1024xf32>
      %62 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %61) -> (tensor<1024xf32>) {
        %extracted_slice_96 = tensor.extract_slice %15#0[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_27[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %67 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = tensor.empty() : tensor<f32>
          %69 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%68 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %70 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%69 : tensor<f32>) {
          ^bb0(%in: f32, %in_98: f32, %out: f32):
            %72 = arith.mulf %in, %in_98 : f32
            %73 = arith.addf %72, %out : f32
            linalg.yield %73 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %70[] : tensor<f32>
          %71 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %71 : f32
        }
        %inserted = tensor.insert %67 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %63 = scf.for %arg18 = %5 to %c1024 step %c1 iter_args(%arg19 = %62) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_0 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %64 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %67 = tensor.empty() : tensor<f32>
        %68 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %69 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%68 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.maxnumf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %69[] : tensor<f32>
        %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.subf %in, %extracted : f32
          %75 = math.exp %74 : f32
          linalg.yield %75 : f32
        } -> tensor<1024xf32>
        %71 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%70 : tensor<1024xf32>) outs(%71 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.addf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted_96 = tensor.extract %72[] : tensor<f32>
        %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%70 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.divf %in, %extracted_96 : f32
          linalg.yield %74 : f32
        } -> tensor<1024xf32>
        cinm.yield %73 : tensor<1024xf32>
      }
      %extracted_slice_94 = tensor.extract_slice %arg17[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %65 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice_95 = tensor.insert_slice %65 into %arg17[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %66 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %inserted_slice_95) -> (tensor<768xf32>) {
        %extracted_slice_96 = tensor.extract_slice %arg19[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_28[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %64[%arg18] : tensor<1024xf32>
        %67 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_96 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_99: f32, %out: f32):
            %69 = arith.mulf %in_99, %extracted : f32
            %70 = arith.addf %in, %69 : f32
            linalg.yield %70 : f32
          } -> tensor<48xf32>
          cinm.yield %68 : tensor<48xf32>
        }
        %inserted_slice_98 = tensor.insert_slice %67 into %arg19[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_98 : tensor<768xf32>
      }
      scf.yield %66 : tensor<768xf32>
    }
    %17 = bufferization.materialize_in_destination %16 in %15#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_29 = tensor.extract_slice %arg9[1, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %18 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_29, %17 : tensor<768x768xf32>, tensor<768xf32>) outs(%17 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.mulf %in, %in_94 : f32
        %63 = arith.addf %out, %62 : f32
        linalg.yield %63 : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%12, %60 : tensor<768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.addf %in, %in_94 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %61 : tensor<768xf32>
    }
    %extracted_slice_30 = tensor.extract_slice %arg13[1, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %19 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%18 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%18, %extracted_slice_30 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_31 = tensor.extract_slice %arg10[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_32 = tensor.extract_slice %arg12[1, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %20:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = tensor.empty() : tensor<2048xf32>
      %61 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%60 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<2048xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_31, %19 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_32, %19 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      cinm.yield %62, %63 : tensor<2048xf32>, tensor<2048xf32>
    }
    %mapped_33 = linalg.map ins(%20#0, %20#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%20#0 : tensor<2048xf32>)
      (%in: f32, %in_94: f32) {
        %60 = arith.negf %in : f32
        %61 = math.exp %60 : f32
        %62 = arith.addf %61, %cst_4 : f32
        %63 = arith.divf %cst_4, %62 : f32
        %64 = arith.mulf %in, %63 : f32
        %65 = arith.mulf %64, %in_94 : f32
        linalg.yield %65 : f32
      }
    %extracted_slice_34 = tensor.extract_slice %arg11[1, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %21 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_34, %mapped_33 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%19 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %61 = arith.mulf %in, %in_94 : f32
        %62 = arith.addf %out, %61 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %60 : tensor<768xf32>
    }
    %extracted_slice_35 = tensor.extract_slice %arg5[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %22 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%21 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%21, %extracted_slice_35 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_36 = tensor.extract_slice %arg6[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_37 = tensor.extract_slice %arg7[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_38 = tensor.extract_slice %arg8[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %23:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_26 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_36, %22 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_37, %22 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_38, %22 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %inserted_slice_94 = tensor.insert_slice %62 into %inserted_slice_25[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %inserted_slice_95 = tensor.insert_slice %63 into %14#2[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %61, %inserted_slice_94, %inserted_slice_95 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %extracted_slice_39 = tensor.extract_slice %23#1[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %24:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %23#0, %arg18 = %extracted_slice_39) -> (tensor<768xf32>, tensor<768xf32>) {
      %60 = arith.remui %arg16, %c48 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.uitofp %61 : i64 to f32
      %63 = arith.divf %62, %cst_5 : f32
      %64 = math.powf %cst_6, %63 : f32
      %65 = arith.divf %cst_4, %64 : f32
      %66 = arith.mulf %3, %65 : f32
      %67 = math.cos %66 : f32
      %68 = math.sin %66 : f32
      %69 = arith.addi %arg16, %c1 : index
      %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_94 = tensor.extract %arg17[%69] : tensor<768xf32>
      %70 = arith.mulf %extracted, %67 : f32
      %71 = arith.mulf %extracted_94, %68 : f32
      %72 = arith.subf %70, %71 : f32
      %inserted = tensor.insert %72 into %arg17[%arg16] : tensor<768xf32>
      %73 = arith.mulf %extracted, %68 : f32
      %74 = arith.mulf %extracted_94, %67 : f32
      %75 = arith.addf %73, %74 : f32
      %inserted_95 = tensor.insert %75 into %inserted[%69] : tensor<768xf32>
      %76 = bufferization.materialize_in_destination %inserted_95 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %77 = arith.cmpi ult, %arg16, %c768 : index
      %78 = scf.if %77 -> (tensor<768xf32>) {
        %extracted_96 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_97 = tensor.extract %arg18[%69] : tensor<768xf32>
        %79 = arith.mulf %extracted_96, %67 : f32
        %80 = arith.mulf %extracted_97, %68 : f32
        %81 = arith.subf %79, %80 : f32
        %inserted_98 = tensor.insert %81 into %arg18[%arg16] : tensor<768xf32>
        %82 = arith.mulf %extracted_96, %68 : f32
        %83 = arith.mulf %extracted_97, %67 : f32
        %84 = arith.addf %82, %83 : f32
        %inserted_99 = tensor.insert %84 into %inserted_98[%69] : tensor<768xf32>
        %85 = bufferization.materialize_in_destination %inserted_99 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %85 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %76, %78 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_40 = tensor.insert_slice %24#1 into %23#1[2, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_41 = tensor.extract_slice %inserted_slice_40[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_42 = tensor.extract_slice %inserted_slice_40[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_43 = tensor.extract_slice %23#2[2, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %25 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %24#0) -> (tensor<768xf32>) {
      %60 = arith.muli %arg16, %c48 : index
      %61 = tensor.empty() : tensor<1024xf32>
      %62 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %61) -> (tensor<1024xf32>) {
        %extracted_slice_96 = tensor.extract_slice %24#0[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_42[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %67 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = tensor.empty() : tensor<f32>
          %69 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%68 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %70 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%69 : tensor<f32>) {
          ^bb0(%in: f32, %in_98: f32, %out: f32):
            %72 = arith.mulf %in, %in_98 : f32
            %73 = arith.addf %72, %out : f32
            linalg.yield %73 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %70[] : tensor<f32>
          %71 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %71 : f32
        }
        %inserted = tensor.insert %67 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %63 = scf.for %arg18 = %5 to %c1024 step %c1 iter_args(%arg19 = %62) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_0 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %64 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %67 = tensor.empty() : tensor<f32>
        %68 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %69 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%68 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.maxnumf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %69[] : tensor<f32>
        %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.subf %in, %extracted : f32
          %75 = math.exp %74 : f32
          linalg.yield %75 : f32
        } -> tensor<1024xf32>
        %71 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%70 : tensor<1024xf32>) outs(%71 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.addf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted_96 = tensor.extract %72[] : tensor<f32>
        %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%70 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.divf %in, %extracted_96 : f32
          linalg.yield %74 : f32
        } -> tensor<1024xf32>
        cinm.yield %73 : tensor<1024xf32>
      }
      %extracted_slice_94 = tensor.extract_slice %arg17[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %65 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice_95 = tensor.insert_slice %65 into %arg17[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %66 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %inserted_slice_95) -> (tensor<768xf32>) {
        %extracted_slice_96 = tensor.extract_slice %arg19[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_43[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %64[%arg18] : tensor<1024xf32>
        %67 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_96 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_99: f32, %out: f32):
            %69 = arith.mulf %in_99, %extracted : f32
            %70 = arith.addf %in, %69 : f32
            linalg.yield %70 : f32
          } -> tensor<48xf32>
          cinm.yield %68 : tensor<48xf32>
        }
        %inserted_slice_98 = tensor.insert_slice %67 into %arg19[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_98 : tensor<768xf32>
      }
      scf.yield %66 : tensor<768xf32>
    }
    %26 = bufferization.materialize_in_destination %25 in %24#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_44 = tensor.extract_slice %arg9[2, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %27 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_44, %26 : tensor<768x768xf32>, tensor<768xf32>) outs(%26 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.mulf %in, %in_94 : f32
        %63 = arith.addf %out, %62 : f32
        linalg.yield %63 : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%21, %60 : tensor<768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.addf %in, %in_94 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %61 : tensor<768xf32>
    }
    %extracted_slice_45 = tensor.extract_slice %arg13[2, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %28 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%27 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%27, %extracted_slice_45 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_46 = tensor.extract_slice %arg10[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_47 = tensor.extract_slice %arg12[2, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %29:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = tensor.empty() : tensor<2048xf32>
      %61 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%60 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<2048xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_46, %28 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_47, %28 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      cinm.yield %62, %63 : tensor<2048xf32>, tensor<2048xf32>
    }
    %mapped_48 = linalg.map ins(%29#0, %29#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%29#0 : tensor<2048xf32>)
      (%in: f32, %in_94: f32) {
        %60 = arith.negf %in : f32
        %61 = math.exp %60 : f32
        %62 = arith.addf %61, %cst_4 : f32
        %63 = arith.divf %cst_4, %62 : f32
        %64 = arith.mulf %in, %63 : f32
        %65 = arith.mulf %64, %in_94 : f32
        linalg.yield %65 : f32
      }
    %extracted_slice_49 = tensor.extract_slice %arg11[2, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %30 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_49, %mapped_48 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%28 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %61 = arith.mulf %in, %in_94 : f32
        %62 = arith.addf %out, %61 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %60 : tensor<768xf32>
    }
    %extracted_slice_50 = tensor.extract_slice %arg5[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %31 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%30 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%30, %extracted_slice_50 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_51 = tensor.extract_slice %arg6[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_52 = tensor.extract_slice %arg7[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_53 = tensor.extract_slice %arg8[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %32:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_41 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_51, %31 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_52, %31 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_53, %31 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %inserted_slice_94 = tensor.insert_slice %62 into %inserted_slice_40[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %inserted_slice_95 = tensor.insert_slice %63 into %23#2[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %61, %inserted_slice_94, %inserted_slice_95 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %extracted_slice_54 = tensor.extract_slice %32#1[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %33:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %32#0, %arg18 = %extracted_slice_54) -> (tensor<768xf32>, tensor<768xf32>) {
      %60 = arith.remui %arg16, %c48 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.uitofp %61 : i64 to f32
      %63 = arith.divf %62, %cst_5 : f32
      %64 = math.powf %cst_6, %63 : f32
      %65 = arith.divf %cst_4, %64 : f32
      %66 = arith.mulf %3, %65 : f32
      %67 = math.cos %66 : f32
      %68 = math.sin %66 : f32
      %69 = arith.addi %arg16, %c1 : index
      %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_94 = tensor.extract %arg17[%69] : tensor<768xf32>
      %70 = arith.mulf %extracted, %67 : f32
      %71 = arith.mulf %extracted_94, %68 : f32
      %72 = arith.subf %70, %71 : f32
      %inserted = tensor.insert %72 into %arg17[%arg16] : tensor<768xf32>
      %73 = arith.mulf %extracted, %68 : f32
      %74 = arith.mulf %extracted_94, %67 : f32
      %75 = arith.addf %73, %74 : f32
      %inserted_95 = tensor.insert %75 into %inserted[%69] : tensor<768xf32>
      %76 = bufferization.materialize_in_destination %inserted_95 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %77 = arith.cmpi ult, %arg16, %c768 : index
      %78 = scf.if %77 -> (tensor<768xf32>) {
        %extracted_96 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_97 = tensor.extract %arg18[%69] : tensor<768xf32>
        %79 = arith.mulf %extracted_96, %67 : f32
        %80 = arith.mulf %extracted_97, %68 : f32
        %81 = arith.subf %79, %80 : f32
        %inserted_98 = tensor.insert %81 into %arg18[%arg16] : tensor<768xf32>
        %82 = arith.mulf %extracted_96, %68 : f32
        %83 = arith.mulf %extracted_97, %67 : f32
        %84 = arith.addf %82, %83 : f32
        %inserted_99 = tensor.insert %84 into %inserted_98[%69] : tensor<768xf32>
        %85 = bufferization.materialize_in_destination %inserted_99 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %85 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %76, %78 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_55 = tensor.insert_slice %33#1 into %32#1[3, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_56 = tensor.extract_slice %inserted_slice_55[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_57 = tensor.extract_slice %inserted_slice_55[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_58 = tensor.extract_slice %32#2[3, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %34 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %33#0) -> (tensor<768xf32>) {
      %60 = arith.muli %arg16, %c48 : index
      %61 = tensor.empty() : tensor<1024xf32>
      %62 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %61) -> (tensor<1024xf32>) {
        %extracted_slice_96 = tensor.extract_slice %33#0[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_57[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %67 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = tensor.empty() : tensor<f32>
          %69 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%68 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %70 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%69 : tensor<f32>) {
          ^bb0(%in: f32, %in_98: f32, %out: f32):
            %72 = arith.mulf %in, %in_98 : f32
            %73 = arith.addf %72, %out : f32
            linalg.yield %73 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %70[] : tensor<f32>
          %71 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %71 : f32
        }
        %inserted = tensor.insert %67 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %63 = scf.for %arg18 = %5 to %c1024 step %c1 iter_args(%arg19 = %62) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_0 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %64 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %67 = tensor.empty() : tensor<f32>
        %68 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %69 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%68 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.maxnumf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %69[] : tensor<f32>
        %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.subf %in, %extracted : f32
          %75 = math.exp %74 : f32
          linalg.yield %75 : f32
        } -> tensor<1024xf32>
        %71 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%70 : tensor<1024xf32>) outs(%71 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.addf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted_96 = tensor.extract %72[] : tensor<f32>
        %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%70 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.divf %in, %extracted_96 : f32
          linalg.yield %74 : f32
        } -> tensor<1024xf32>
        cinm.yield %73 : tensor<1024xf32>
      }
      %extracted_slice_94 = tensor.extract_slice %arg17[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %65 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice_95 = tensor.insert_slice %65 into %arg17[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %66 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %inserted_slice_95) -> (tensor<768xf32>) {
        %extracted_slice_96 = tensor.extract_slice %arg19[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_58[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %64[%arg18] : tensor<1024xf32>
        %67 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_96 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_99: f32, %out: f32):
            %69 = arith.mulf %in_99, %extracted : f32
            %70 = arith.addf %in, %69 : f32
            linalg.yield %70 : f32
          } -> tensor<48xf32>
          cinm.yield %68 : tensor<48xf32>
        }
        %inserted_slice_98 = tensor.insert_slice %67 into %arg19[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_98 : tensor<768xf32>
      }
      scf.yield %66 : tensor<768xf32>
    }
    %35 = bufferization.materialize_in_destination %34 in %33#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_59 = tensor.extract_slice %arg9[3, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %36 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_59, %35 : tensor<768x768xf32>, tensor<768xf32>) outs(%35 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.mulf %in, %in_94 : f32
        %63 = arith.addf %out, %62 : f32
        linalg.yield %63 : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%30, %60 : tensor<768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.addf %in, %in_94 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %61 : tensor<768xf32>
    }
    %extracted_slice_60 = tensor.extract_slice %arg13[3, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %37 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%36 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%36, %extracted_slice_60 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_61 = tensor.extract_slice %arg10[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_62 = tensor.extract_slice %arg12[3, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %38:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = tensor.empty() : tensor<2048xf32>
      %61 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%60 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<2048xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_61, %37 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_62, %37 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      cinm.yield %62, %63 : tensor<2048xf32>, tensor<2048xf32>
    }
    %mapped_63 = linalg.map ins(%38#0, %38#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%38#0 : tensor<2048xf32>)
      (%in: f32, %in_94: f32) {
        %60 = arith.negf %in : f32
        %61 = math.exp %60 : f32
        %62 = arith.addf %61, %cst_4 : f32
        %63 = arith.divf %cst_4, %62 : f32
        %64 = arith.mulf %in, %63 : f32
        %65 = arith.mulf %64, %in_94 : f32
        linalg.yield %65 : f32
      }
    %extracted_slice_64 = tensor.extract_slice %arg11[3, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %39 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_64, %mapped_63 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%37 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %61 = arith.mulf %in, %in_94 : f32
        %62 = arith.addf %out, %61 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %60 : tensor<768xf32>
    }
    %extracted_slice_65 = tensor.extract_slice %arg5[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %40 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%39 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%39, %extracted_slice_65 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_66 = tensor.extract_slice %arg6[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_67 = tensor.extract_slice %arg7[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_68 = tensor.extract_slice %arg8[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %41:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_56 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_66, %40 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_67, %40 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_68, %40 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %inserted_slice_94 = tensor.insert_slice %62 into %inserted_slice_55[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %inserted_slice_95 = tensor.insert_slice %63 into %32#2[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %61, %inserted_slice_94, %inserted_slice_95 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %extracted_slice_69 = tensor.extract_slice %41#1[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %42:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %41#0, %arg18 = %extracted_slice_69) -> (tensor<768xf32>, tensor<768xf32>) {
      %60 = arith.remui %arg16, %c48 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.uitofp %61 : i64 to f32
      %63 = arith.divf %62, %cst_5 : f32
      %64 = math.powf %cst_6, %63 : f32
      %65 = arith.divf %cst_4, %64 : f32
      %66 = arith.mulf %3, %65 : f32
      %67 = math.cos %66 : f32
      %68 = math.sin %66 : f32
      %69 = arith.addi %arg16, %c1 : index
      %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_94 = tensor.extract %arg17[%69] : tensor<768xf32>
      %70 = arith.mulf %extracted, %67 : f32
      %71 = arith.mulf %extracted_94, %68 : f32
      %72 = arith.subf %70, %71 : f32
      %inserted = tensor.insert %72 into %arg17[%arg16] : tensor<768xf32>
      %73 = arith.mulf %extracted, %68 : f32
      %74 = arith.mulf %extracted_94, %67 : f32
      %75 = arith.addf %73, %74 : f32
      %inserted_95 = tensor.insert %75 into %inserted[%69] : tensor<768xf32>
      %76 = bufferization.materialize_in_destination %inserted_95 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %77 = arith.cmpi ult, %arg16, %c768 : index
      %78 = scf.if %77 -> (tensor<768xf32>) {
        %extracted_96 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_97 = tensor.extract %arg18[%69] : tensor<768xf32>
        %79 = arith.mulf %extracted_96, %67 : f32
        %80 = arith.mulf %extracted_97, %68 : f32
        %81 = arith.subf %79, %80 : f32
        %inserted_98 = tensor.insert %81 into %arg18[%arg16] : tensor<768xf32>
        %82 = arith.mulf %extracted_96, %68 : f32
        %83 = arith.mulf %extracted_97, %67 : f32
        %84 = arith.addf %82, %83 : f32
        %inserted_99 = tensor.insert %84 into %inserted_98[%69] : tensor<768xf32>
        %85 = bufferization.materialize_in_destination %inserted_99 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %85 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %76, %78 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_70 = tensor.insert_slice %42#1 into %41#1[4, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_71 = tensor.extract_slice %inserted_slice_70[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %extracted_slice_72 = tensor.extract_slice %inserted_slice_70[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_73 = tensor.extract_slice %41#2[4, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %43 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %42#0) -> (tensor<768xf32>) {
      %60 = arith.muli %arg16, %c48 : index
      %61 = tensor.empty() : tensor<1024xf32>
      %62 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %61) -> (tensor<1024xf32>) {
        %extracted_slice_96 = tensor.extract_slice %42#0[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_72[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %67 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = tensor.empty() : tensor<f32>
          %69 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%68 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %70 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%69 : tensor<f32>) {
          ^bb0(%in: f32, %in_98: f32, %out: f32):
            %72 = arith.mulf %in, %in_98 : f32
            %73 = arith.addf %72, %out : f32
            linalg.yield %73 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %70[] : tensor<f32>
          %71 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %71 : f32
        }
        %inserted = tensor.insert %67 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %63 = scf.for %arg18 = %5 to %c1024 step %c1 iter_args(%arg19 = %62) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_0 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %64 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %67 = tensor.empty() : tensor<f32>
        %68 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %69 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%68 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.maxnumf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %69[] : tensor<f32>
        %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.subf %in, %extracted : f32
          %75 = math.exp %74 : f32
          linalg.yield %75 : f32
        } -> tensor<1024xf32>
        %71 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%70 : tensor<1024xf32>) outs(%71 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.addf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted_96 = tensor.extract %72[] : tensor<f32>
        %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%70 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.divf %in, %extracted_96 : f32
          linalg.yield %74 : f32
        } -> tensor<1024xf32>
        cinm.yield %73 : tensor<1024xf32>
      }
      %extracted_slice_94 = tensor.extract_slice %arg17[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %65 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice_95 = tensor.insert_slice %65 into %arg17[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %66 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %inserted_slice_95) -> (tensor<768xf32>) {
        %extracted_slice_96 = tensor.extract_slice %arg19[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_73[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %64[%arg18] : tensor<1024xf32>
        %67 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_96 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_99: f32, %out: f32):
            %69 = arith.mulf %in_99, %extracted : f32
            %70 = arith.addf %in, %69 : f32
            linalg.yield %70 : f32
          } -> tensor<48xf32>
          cinm.yield %68 : tensor<48xf32>
        }
        %inserted_slice_98 = tensor.insert_slice %67 into %arg19[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_98 : tensor<768xf32>
      }
      scf.yield %66 : tensor<768xf32>
    }
    %44 = bufferization.materialize_in_destination %43 in %42#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_74 = tensor.extract_slice %arg9[4, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %45 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_74, %44 : tensor<768x768xf32>, tensor<768xf32>) outs(%44 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.mulf %in, %in_94 : f32
        %63 = arith.addf %out, %62 : f32
        linalg.yield %63 : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%39, %60 : tensor<768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.addf %in, %in_94 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %61 : tensor<768xf32>
    }
    %extracted_slice_75 = tensor.extract_slice %arg13[4, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %46 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%45 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%45, %extracted_slice_75 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_76 = tensor.extract_slice %arg10[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_77 = tensor.extract_slice %arg12[4, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %47:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = tensor.empty() : tensor<2048xf32>
      %61 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%60 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<2048xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_76, %46 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_77, %46 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      cinm.yield %62, %63 : tensor<2048xf32>, tensor<2048xf32>
    }
    %mapped_78 = linalg.map ins(%47#0, %47#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%47#0 : tensor<2048xf32>)
      (%in: f32, %in_94: f32) {
        %60 = arith.negf %in : f32
        %61 = math.exp %60 : f32
        %62 = arith.addf %61, %cst_4 : f32
        %63 = arith.divf %cst_4, %62 : f32
        %64 = arith.mulf %in, %63 : f32
        %65 = arith.mulf %64, %in_94 : f32
        linalg.yield %65 : f32
      }
    %extracted_slice_79 = tensor.extract_slice %arg11[4, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %48 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_79, %mapped_78 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%46 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %61 = arith.mulf %in, %in_94 : f32
        %62 = arith.addf %out, %61 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %60 : tensor<768xf32>
    }
    %extracted_slice_80 = tensor.extract_slice %arg5[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %49 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%48 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%48, %extracted_slice_80 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_81 = tensor.extract_slice %arg6[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_82 = tensor.extract_slice %arg7[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %extracted_slice_83 = tensor.extract_slice %arg8[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %50:3 = cinm.compute_ -> tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_71 : tensor<768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_81, %49 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_82, %49 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_83, %49 : tensor<768x768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %64 = arith.mulf %in, %in_96 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<768xf32>
      %inserted_slice_94 = tensor.insert_slice %62 into %inserted_slice_70[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      %inserted_slice_95 = tensor.insert_slice %63 into %41#2[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
      cinm.yield %61, %inserted_slice_94, %inserted_slice_95 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
    }
    %extracted_slice_84 = tensor.extract_slice %50#1[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>
    %51:2 = scf.for %arg16 = %c0 to %c768 step %c2 iter_args(%arg17 = %50#0, %arg18 = %extracted_slice_84) -> (tensor<768xf32>, tensor<768xf32>) {
      %60 = arith.remui %arg16, %c48 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.uitofp %61 : i64 to f32
      %63 = arith.divf %62, %cst_5 : f32
      %64 = math.powf %cst_6, %63 : f32
      %65 = arith.divf %cst_4, %64 : f32
      %66 = arith.mulf %3, %65 : f32
      %67 = math.cos %66 : f32
      %68 = math.sin %66 : f32
      %69 = arith.addi %arg16, %c1 : index
      %extracted = tensor.extract %arg17[%arg16] : tensor<768xf32>
      %extracted_94 = tensor.extract %arg17[%69] : tensor<768xf32>
      %70 = arith.mulf %extracted, %67 : f32
      %71 = arith.mulf %extracted_94, %68 : f32
      %72 = arith.subf %70, %71 : f32
      %inserted = tensor.insert %72 into %arg17[%arg16] : tensor<768xf32>
      %73 = arith.mulf %extracted, %68 : f32
      %74 = arith.mulf %extracted_94, %67 : f32
      %75 = arith.addf %73, %74 : f32
      %inserted_95 = tensor.insert %75 into %inserted[%69] : tensor<768xf32>
      %76 = bufferization.materialize_in_destination %inserted_95 in %arg17 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      %77 = arith.cmpi ult, %arg16, %c768 : index
      %78 = scf.if %77 -> (tensor<768xf32>) {
        %extracted_96 = tensor.extract %arg18[%arg16] : tensor<768xf32>
        %extracted_97 = tensor.extract %arg18[%69] : tensor<768xf32>
        %79 = arith.mulf %extracted_96, %67 : f32
        %80 = arith.mulf %extracted_97, %68 : f32
        %81 = arith.subf %79, %80 : f32
        %inserted_98 = tensor.insert %81 into %arg18[%arg16] : tensor<768xf32>
        %82 = arith.mulf %extracted_96, %68 : f32
        %83 = arith.mulf %extracted_97, %67 : f32
        %84 = arith.addf %82, %83 : f32
        %inserted_99 = tensor.insert %84 into %inserted_98[%69] : tensor<768xf32>
        %85 = bufferization.materialize_in_destination %inserted_99 in %arg18 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        scf.yield %85 : tensor<768xf32>
      } else {
        scf.yield %arg18 : tensor<768xf32>
      }
      scf.yield %76, %78 : tensor<768xf32>, tensor<768xf32>
    }
    %inserted_slice_85 = tensor.insert_slice %51#1 into %50#1[5, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32> into tensor<6x1024x768xf32>
    %extracted_slice_86 = tensor.extract_slice %inserted_slice_85[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %extracted_slice_87 = tensor.extract_slice %50#2[5, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %52 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %51#0) -> (tensor<768xf32>) {
      %60 = arith.muli %arg16, %c48 : index
      %61 = tensor.empty() : tensor<1024xf32>
      %62 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %61) -> (tensor<1024xf32>) {
        %extracted_slice_96 = tensor.extract_slice %51#0[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_86[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %67 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = tensor.empty() : tensor<f32>
          %69 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%68 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %70 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%69 : tensor<f32>) {
          ^bb0(%in: f32, %in_98: f32, %out: f32):
            %72 = arith.mulf %in, %in_98 : f32
            %73 = arith.addf %72, %out : f32
            linalg.yield %73 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %70[] : tensor<f32>
          %71 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %71 : f32
        }
        %inserted = tensor.insert %67 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %63 = scf.for %arg18 = %5 to %c1024 step %c1 iter_args(%arg19 = %62) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_0 into %arg19[%arg18] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %64 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %67 = tensor.empty() : tensor<f32>
        %68 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %69 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%63 : tensor<1024xf32>) outs(%68 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.maxnumf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %69[] : tensor<f32>
        %70 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%63 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.subf %in, %extracted : f32
          %75 = math.exp %74 : f32
          linalg.yield %75 : f32
        } -> tensor<1024xf32>
        %71 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%67 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %72 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%70 : tensor<1024xf32>) outs(%71 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.addf %in, %out : f32
          linalg.yield %74 : f32
        } -> tensor<f32>
        %extracted_96 = tensor.extract %72[] : tensor<f32>
        %73 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%70 : tensor<1024xf32>) outs(%63 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %74 = arith.divf %in, %extracted_96 : f32
          linalg.yield %74 : f32
        } -> tensor<1024xf32>
        cinm.yield %73 : tensor<1024xf32>
      }
      %extracted_slice_94 = tensor.extract_slice %arg17[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %65 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice_94 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice_95 = tensor.insert_slice %65 into %arg17[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %66 = scf.for %arg18 = %c0 to %5 step %c1 iter_args(%arg19 = %inserted_slice_95) -> (tensor<768xf32>) {
        %extracted_slice_96 = tensor.extract_slice %arg19[%60] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_97 = tensor.extract_slice %extracted_slice_87[%arg18, %60] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %64[%arg18] : tensor<1024xf32>
        %67 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %68 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_96, %extracted_slice_97 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_96 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_99: f32, %out: f32):
            %69 = arith.mulf %in_99, %extracted : f32
            %70 = arith.addf %in, %69 : f32
            linalg.yield %70 : f32
          } -> tensor<48xf32>
          cinm.yield %68 : tensor<48xf32>
        }
        %inserted_slice_98 = tensor.insert_slice %67 into %arg19[%60] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_98 : tensor<768xf32>
      }
      scf.yield %66 : tensor<768xf32>
    }
    %53 = bufferization.materialize_in_destination %52 in %51#0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %extracted_slice_88 = tensor.extract_slice %arg9[5, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    %54 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_88, %53 : tensor<768x768xf32>, tensor<768xf32>) outs(%53 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.mulf %in, %in_94 : f32
        %63 = arith.addf %out, %62 : f32
        linalg.yield %63 : f32
      } -> tensor<768xf32>
      %61 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%48, %60 : tensor<768xf32>, tensor<768xf32>) outs(%60 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %62 = arith.addf %in, %in_94 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %61 : tensor<768xf32>
    }
    %extracted_slice_89 = tensor.extract_slice %arg13[5, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
    %55 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%54 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%54, %extracted_slice_89 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %extracted_slice_90 = tensor.extract_slice %arg10[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %extracted_slice_91 = tensor.extract_slice %arg12[5, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %56:2 = cinm.compute_ -> tensor<2048xf32>, tensor<2048xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = tensor.empty() : tensor<2048xf32>
      %61 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%60 : tensor<2048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<2048xf32>
      %62 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_90, %55 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      %63 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_91, %55 : tensor<2048x768xf32>, tensor<768xf32>) outs(%61 : tensor<2048xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %64 = arith.mulf %in, %in_94 : f32
        %65 = arith.addf %out, %64 : f32
        linalg.yield %65 : f32
      } -> tensor<2048xf32>
      cinm.yield %62, %63 : tensor<2048xf32>, tensor<2048xf32>
    }
    %mapped_92 = linalg.map ins(%56#0, %56#1 : tensor<2048xf32>, tensor<2048xf32>) outs(%56#0 : tensor<2048xf32>)
      (%in: f32, %in_94: f32) {
        %60 = arith.negf %in : f32
        %61 = math.exp %60 : f32
        %62 = arith.addf %61, %cst_4 : f32
        %63 = arith.divf %cst_4, %62 : f32
        %64 = arith.mulf %in, %63 : f32
        %65 = arith.mulf %64, %in_94 : f32
        linalg.yield %65 : f32
      }
    %extracted_slice_93 = tensor.extract_slice %arg11[5, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    %57 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
      %60 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_93, %mapped_92 : tensor<768x2048xf32>, tensor<2048xf32>) outs(%55 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %61 = arith.mulf %in, %in_94 : f32
        %62 = arith.addf %out, %61 : f32
        linalg.yield %62 : f32
      } -> tensor<768xf32>
      cinm.yield %60 : tensor<768xf32>
    }
    %58 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %60 = tensor.empty() : tensor<f32>
      %61 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%60 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %62 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%57 : tensor<768xf32>) outs(%61 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %67 = arith.mulf %in, %in : f32
        %68 = arith.addf %67, %out : f32
        linalg.yield %68 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %62[] : tensor<f32>
      %63 = arith.divf %extracted, %cst_2 : f32
      %64 = arith.addf %63, %cst_3 : f32
      %65 = math.rsqrt %64 : f32
      %66 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%57, %arg14 : tensor<768xf32>, tensor<768xf32>) outs(%6 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_94: f32, %out: f32):
        %67 = arith.mulf %in, %65 : f32
        %68 = arith.mulf %67, %in_94 : f32
        linalg.yield %68 : f32
      } -> tensor<768xf32>
      cinm.yield %66 : tensor<768xf32>
    }
    %59 = cinm.compute_ -> tensor<32000xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 8, 16>} {
      %60 = tensor.empty() : tensor<34048x768xf32>
      %61 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%60 : tensor<34048x768xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<34048x768xf32>
      %inserted_slice_94 = tensor.insert_slice %arg15 into %61[0, 0] [32000, 768] [1, 1] : tensor<32000x768xf32> into tensor<34048x768xf32>
      %62 = tensor.empty() : tensor<34048xf32>
      %63 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%62 : tensor<34048xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<34048xf32>
      %64 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%inserted_slice_94, %58 : tensor<34048x768xf32>, tensor<768xf32>) outs(%63 : tensor<34048xf32>) {
      ^bb0(%in: f32, %in_96: f32, %out: f32):
        %65 = arith.mulf %in, %in_96 : f32
        %66 = arith.addf %out, %65 : f32
        linalg.yield %66 : f32
      } -> tensor<34048xf32>
      %extracted_slice_95 = tensor.extract_slice %64[0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
      cinm.yield %extracted_slice_95 : tensor<32000xf32>
    }
    return %59 : tensor<32000xf32>
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
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 6.92820311 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %2 = scf.for %arg4 = %c0 to %c6 step %c1 iter_args(%arg5 = %1) -> (tensor<768xf32>) {
      %3 = arith.muli %arg4, %c48 : index
      %4 = tensor.empty() : tensor<1024xf32>
      %5 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %4) -> (tensor<1024xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg0[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %10 = cinm.compute_ -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %11 = tensor.empty() : tensor<f32>
          %12 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%11 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %13 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<48xf32>, tensor<48xf32>) outs(%12 : tensor<f32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %15 = arith.mulf %in, %in_4 : f32
            %16 = arith.addf %15, %out : f32
            linalg.yield %16 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %13[] : tensor<f32>
          %14 = arith.divf %extracted, %cst_0 : f32
          cinm.yield %14 : f32
        }
        %inserted = tensor.insert %10 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %6 = scf.for %arg6 = %0 to %c1024 step %c1 iter_args(%arg7 = %5) -> (tensor<1024xf32>) {
        %inserted = tensor.insert %cst_1 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %7 = cinm.compute_ -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %10 = tensor.empty() : tensor<f32>
        %11 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_1 : f32
        } -> tensor<f32>
        %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%6 : tensor<1024xf32>) outs(%11 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.maxnumf %in, %out : f32
          linalg.yield %17 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %12[] : tensor<f32>
        %13 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%6 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.subf %in, %extracted : f32
          %18 = math.exp %17 : f32
          linalg.yield %18 : f32
        } -> tensor<1024xf32>
        %14 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%10 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %15 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%13 : tensor<1024xf32>) outs(%14 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.addf %in, %out : f32
          linalg.yield %17 : f32
        } -> tensor<f32>
        %extracted_2 = tensor.extract %15[] : tensor<f32>
        %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%13 : tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %17 = arith.divf %in, %extracted_2 : f32
          linalg.yield %17 : f32
        } -> tensor<1024xf32>
        cinm.yield %16 : tensor<1024xf32>
      }
      %extracted_slice = tensor.extract_slice %arg5[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
      %8 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg5[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %9 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %inserted_slice) -> (tensor<768xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg7[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_3 = tensor.extract_slice %arg2[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %7[%arg6] : tensor<1024xf32>
        %10 = cinm.compute_ -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<48xf32>, tensor<48xf32>) outs(%extracted_slice_2 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %12 = arith.mulf %in_5, %extracted : f32
            %13 = arith.addf %in, %12 : f32
            linalg.yield %13 : f32
          } -> tensor<48xf32>
          cinm.yield %11 : tensor<48xf32>
        }
        %inserted_slice_4 = tensor.insert_slice %10 into %arg7[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_4 : tensor<768xf32>
      }
      scf.yield %9 : tensor<768xf32>
    }
    return %2 : tensor<768xf32>
  }
  func.func @rmsnorm(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = cinm.compute_ -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<768xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.mulf %in, %in : f32
        %10 = arith.addf %9, %out : f32
        linalg.yield %10 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = arith.divf %extracted, %cst_1 : f32
      %5 = arith.addf %4, %cst_0 : f32
      %6 = math.rsqrt %5 : f32
      %7 = tensor.empty() : tensor<768xf32>
      %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<768xf32>, tensor<768xf32>) outs(%7 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %9 = arith.mulf %in, %6 : f32
        %10 = arith.mulf %9, %in_2 : f32
        linalg.yield %10 : f32
      } -> tensor<768xf32>
      cinm.yield %8 : tensor<768xf32>
    }
    return %0 : tensor<768xf32>
  }
  func.func @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = cinm.compute_ -> tensor<1024xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.maxnumf %in, %out : f32
        linalg.yield %8 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.subf %in, %extracted : f32
        %9 = math.exp %8 : f32
        linalg.yield %9 : f32
      } -> tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%4 : tensor<1024xf32>) outs(%5 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.addf %in, %out : f32
        linalg.yield %8 : f32
      } -> tensor<f32>
      %extracted_1 = tensor.extract %6[] : tensor<f32>
      %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.divf %in, %extracted_1 : f32
        linalg.yield %8 : f32
      } -> tensor<1024xf32>
      cinm.yield %7 : tensor<1024xf32>
    }
    return %0 : tensor<1024xf32>
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

