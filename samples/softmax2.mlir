#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module {
  func.func @softie(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %0 = cinm.compute -> tensor<1024xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.maxnumf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = tensor.empty() : tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg0, %extracted : tensor<1024xf32>, f32) outs(%4 : tensor<1024xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.subf %in, %in_0 : f32
        %7 = math.exp %6 : f32
        linalg.yield %7 : f32
      } -> tensor<1024xf32>
      %u = bufferization.materialize_in_destination %5 in %arg0: (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
      cinm.yield %u : tensor<1024xf32>
    }
    return %0 : tensor<1024xf32>
  }
//   func.func @softmax(%arg0: tensor<1024xf32> {bufferization.writable = true}) -> tensor<1024xf32> {
//     %cst = arith.constant 0.000000e+00 : f32
//     %cst_0 = arith.constant 0xFF800000 : f32
//     %0 = cinm.compute -> tensor<1024xf32>
//          attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
//       %1 = tensor.empty() : tensor<f32>
//       %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) {
//       ^bb0(%out: f32):
//         linalg.yield %cst_0 : f32
//       } -> tensor<f32>
//       %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1024xf32>) outs(%2 : tensor<f32>) {
//       ^bb0(%in: f32, %out: f32):
//         %11 = arith.maxnumf %in, %out : f32
//         linalg.yield %11 : f32
//       } -> tensor<f32>
//       %extracted = tensor.extract %3[] : tensor<f32>
//       %4 = tensor.empty() : tensor<1024xf32>
//       %5 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg0, %extracted : tensor<1024xf32>, f32) outs(%4 : tensor<1024xf32>) {
//       ^bb0(%in: f32, %in_2: f32, %out: f32):
//         %11 = arith.subf %in, %in_2 : f32
//         %12 = math.exp %11 : f32
//         linalg.yield %12 : f32
//       } -> tensor<1024xf32>
//       %6 = tensor.empty() : tensor<f32>
//       %7 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%6 : tensor<f32>) {
//       ^bb0(%out: f32):
//         linalg.yield %cst : f32
//       } -> tensor<f32>
//       %8 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%5 : tensor<1024xf32>) outs(%7 : tensor<f32>) {
//       ^bb0(%in: f32, %out: f32):
//         %11 = arith.addf %in, %out : f32
//         linalg.yield %11 : f32
//       } -> tensor<f32>
//       %extracted_1 = tensor.extract %8[] : tensor<f32>
//       %9 = tensor.empty() : tensor<1024xf32>
//       %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%5, %extracted_1 : tensor<1024xf32>, f32) outs(%9 : tensor<1024xf32>) {
//       ^bb0(%in: f32, %in_2: f32, %out: f32):
//         %11 = arith.divf %in, %in_2 : f32
//         linalg.yield %11 : f32
//       } -> tensor<1024xf32>
//       cinm.yield %10 : tensor<1024xf32>
//     }
//     return %0 : tensor<1024xf32>
//   }
}

