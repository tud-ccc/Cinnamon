#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module {
  func.func @rmsnorm(%arg0: memref<768xf32>, %arg1: memref<768xf32>) -> memref<768xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.99999974E-6 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %0 = cinm.compute -> memref<768xf32>
         attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
      linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%alloc : memref<f32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : memref<768xf32>) outs(%alloc : memref<f32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.mulf %in, %in : f32
        %6 = arith.addf %5, %out : f32
        linalg.yield %6 : f32
      }
      %1 = memref.load %alloc[] : memref<f32>
      %2 = arith.divf %1, %cst_1 : f32
      %3 = arith.addf %2, %cst_0 : f32
      %4 = math.rsqrt %3 : f32
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
      linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %4, %arg1 : memref<768xf32>, f32, memref<768xf32>) outs(%alloc_2 : memref<768xf32>) {
      ^bb0(%in: f32, %in_3: f32, %in_4: f32, %out: f32):
        %5 = arith.mulf %in, %in_3 : f32
        %6 = arith.mulf %5, %in_4 : f32
        linalg.yield %6 : f32
      }
      cinm.yield %alloc_2 : memref<768xf32>
    }
    return %0 : memref<768xf32>
  }
}
