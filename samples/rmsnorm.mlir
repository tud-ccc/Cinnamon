#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0) -> ()>
module {
  func.func @rmsnorm(%arg0: tensor<768xf32> {bufferization.writable = true}, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = cinm.compute -> tensor<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %1 = tensor.empty() : tensor<f32>
      %2 = linalg.generic {indexing_maps = [#map4], iterator_types = []} outs(%1 : tensor<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["reduction"]} ins(%arg0 : tensor<768xf32>) outs(%2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.mulf %in, %in : f32
        %11 = arith.addf %10, %out : f32
        linalg.yield %11 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %3[] : tensor<f32>
      %4 = arith.divf %extracted, %cst_1 : f32
      %5 = arith.addf %4, %cst_0 : f32
      %6 = math.rsqrt %5 : f32
      %7 = tensor.empty() : tensor<768xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map5, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %6, %arg1 : tensor<768xf32>, f32, tensor<768xf32>) outs(%7 : tensor<768xf32>) {
      ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
        %10 = arith.mulf %in, %in_2 : f32
        %11 = arith.mulf %10, %in_3 : f32
        linalg.yield %11 : f32
      } -> tensor<768xf32>
      // %9 = bufferization.materialize_in_destination %8 in %arg0 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
      cinm.yield %8 : tensor<768xf32>
    }
    return %0 : tensor<768xf32>
  }
}

