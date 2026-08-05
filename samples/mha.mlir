#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0) -> ()>
  func.func @mha(%arg0: tensor<768xf32>{bufferization.writable=true}, %arg1: tensor<1024x768xf32>, %arg2: tensor<1024x768xf32>, %arg3: index) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %cst_1 = arith.constant 6.92820311 : f32
    %0 = arith.addi %arg3, %c1 : index
    %1 = tensor.empty() : tensor<768xf32>
    %2 = scf.for %arg4 = %c0 to %c6 step %c1 iter_args(%arg5 = %1) -> (tensor<768xf32>) {
      %3 = arith.muli %arg4, %c48 : index
      %4 = tensor.empty() : tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<1024xf32>
      %6 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %5) -> (tensor<1024xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_2 = tensor.extract_slice %arg1[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %11 = cinm.compute -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %12 = tensor.empty() : tensor<f32>
          %13 = linalg.generic {indexing_maps = [#map4], iterator_types = []} outs(%12 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %14 = linalg.generic {indexing_maps = [#map, #map, #map5], iterator_types = ["reduction"]} ins(%extracted_slice, %extracted_slice_2 : tensor<48xf32>, tensor<48xf32>) outs(%13 : tensor<f32>) {
          ^bb0(%in: f32, %in_3: f32, %out: f32):
            %16 = arith.mulf %in, %in_3 : f32
            %17 = arith.addf %16, %out : f32
            linalg.yield %17 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %14[] : tensor<f32>
          %15 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %15 : f32
        }
        %inserted = tensor.insert %11 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %70 = func.call @softmax(%6) : (tensor<1024xf32>) -> tensor<1024xf32>
      %7 = bufferization.materialize_in_destination %70 in %6 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
      %8 = tensor.empty() : tensor<48xf32>
      %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%8 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg5[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %10 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %inserted_slice) -> (tensor<768xf32>) {
        %extracted_slice = tensor.extract_slice %arg7[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %7[%arg6] : tensor<1024xf32>
        %11 = cinm.compute -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%extracted_slice, %extracted_slice_2 : tensor<48xf32>, tensor<48xf32>) outs(%8 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %13 = arith.mulf %in_4, %extracted : f32
            %14 = arith.addf %in, %13 : f32
            linalg.yield %14 : f32
          } -> tensor<48xf32>
          cinm.yield %12 : tensor<48xf32>
        }
        %inserted_slice_3 = tensor.insert_slice %11 into %arg7[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_3 : tensor<768xf32>
      }
      scf.yield %10 : tensor<768xf32>
    }
    return %2 : tensor<768xf32>
  }

  func.func private @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = cinm.compute -> tensor<1024xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map4], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.maxnumf %in, %out : f32
        linalg.yield %9 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = tensor.empty() : tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.subf %in, %extracted : f32
        %10 = math.exp %9 : f32
        linalg.yield %10 : f32
      } -> tensor<1024xf32>
      %6 = linalg.generic {indexing_maps = [#map4], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %7 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction"]} ins(%5 : tensor<1024xf32>) outs(%6 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.addf %in, %out : f32
        linalg.yield %9 : f32
      } -> tensor<f32>
      %extracted_1 = tensor.extract %7[] : tensor<f32>
      %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.divf %in, %extracted_1 : f32
        linalg.yield %9 : f32
      } -> tensor<1024xf32>
      cinm.yield %8 : tensor<1024xf32>
    }
    return %0 : tensor<1024xf32>
  }
