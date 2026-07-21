#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0) -> ()>
module {
  func.func @mha(%arg0: tensor<768xf32> {bufferization.writable = true}, %arg1: tensor<1024x768xf32>, %arg2: tensor<1024x768xf32>, %arg3: index) -> tensor<768xf32> {
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
        %12 = cinm.compute -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %13 = tensor.empty() : tensor<f32>
          %14 = linalg.generic {indexing_maps = [#map1], iterator_types = []} outs(%13 : tensor<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<f32>
          %15 = linalg.generic {indexing_maps = [#map, #map, #map2], iterator_types = ["reduction"]} ins(%extracted_slice, %extracted_slice_2 : tensor<48xf32>, tensor<48xf32>) outs(%14 : tensor<f32>) {
          ^bb0(%in: f32, %in_3: f32, %out: f32):
            %17 = arith.mulf %in, %in_3 : f32
            %18 = arith.addf %17, %out : f32
            linalg.yield %18 : f32
          } -> tensor<f32>
          %extracted = tensor.extract %15[] : tensor<f32>
          %16 = arith.divf %extracted, %cst_1 : f32
          cinm.yield %16 : f32
        }
        %inserted = tensor.insert %12 into %arg7[%arg6] : tensor<1024xf32>
        scf.yield %inserted : tensor<1024xf32>
      }
      %7 = cinm.compute -> tensor<1024xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %12 = tensor.empty() : tensor<f32>
        %13 = linalg.generic {indexing_maps = [#map1], iterator_types = []} outs(%12 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        } -> tensor<f32>
        %14 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction"]} ins(%6 : tensor<1024xf32>) outs(%13 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %19 = arith.maxnumf %in, %out : f32
          linalg.yield %19 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %14[] : tensor<f32>
        %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%6 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %19 = arith.subf %in, %extracted : f32
          %20 = math.exp %19 : f32
          linalg.yield %20 : f32
        } -> tensor<1024xf32>
        %16 = linalg.generic {indexing_maps = [#map1], iterator_types = []} outs(%12 : tensor<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<f32>
        %17 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction"]} ins(%15 : tensor<1024xf32>) outs(%16 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %19 = arith.addf %in, %out : f32
          linalg.yield %19 : f32
        } -> tensor<f32>
        %extracted_2 = tensor.extract %17[] : tensor<f32>
        %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%15 : tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %19 = arith.divf %in, %extracted_2 : f32
          linalg.yield %19 : f32
        } -> tensor<1024xf32>
        cinm.yield %18 : tensor<1024xf32>
      }
      %8 = bufferization.materialize_in_destination %7 in %6 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
      %9 = tensor.empty() : tensor<48xf32>
      %10 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%9 : tensor<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<48xf32>
      %inserted_slice = tensor.insert_slice %10 into %arg5[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
      %11 = scf.for %arg6 = %c0 to %0 step %c1 iter_args(%arg7 = %inserted_slice) -> (tensor<768xf32>) {
        %extracted_slice = tensor.extract_slice %arg7[%3] [48] [1] : tensor<768xf32> to tensor<48xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%arg6, %3] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
        %extracted = tensor.extract %8[%arg6] : tensor<1024xf32>
        %12 = cinm.compute -> tensor<48xf32>
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%extracted_slice, %extracted_slice_2 : tensor<48xf32>, tensor<48xf32>) outs(%9 : tensor<48xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %14 = arith.mulf %in_4, %extracted : f32
            %15 = arith.addf %in, %14 : f32
            linalg.yield %15 : f32
          } -> tensor<48xf32>
          cinm.yield %13 : tensor<48xf32>
        }
        %inserted_slice_3 = tensor.insert_slice %12 into %arg7[%3] [48] [1] : tensor<48xf32> into tensor<768xf32>
        scf.yield %inserted_slice_3 : tensor<768xf32>
      }
      scf.yield %11 : tensor<768xf32>
    }
    return %2 : tensor<768xf32>
  }
}

