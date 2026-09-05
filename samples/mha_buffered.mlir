#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0) -> ()>
module {
  func.func @mha(%arg0: memref<768xf32, strided<[?], offset: ?>>, %arg1: memref<1024x768xf32, strided<[?, ?], offset: ?>>, %arg2: memref<1024x768xf32, strided<[?, ?], offset: ?>>, %arg3: index, %arg4: memref<768xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %cst_1 = arith.constant 6.92820311 : f32
    %0 = arith.addi %arg3, %c1 : index
    scf.for %arg5 = %c0 to %c6 step %c1 {
      %1 = arith.muli %arg5, %c48 : index
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_2 : memref<1024xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      scf.for %arg6 = %c0 to %0 step %c1 {
        %subview_4 = memref.subview %arg0[%1] [48] [1] : memref<768xf32, strided<[?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %subview_5 = memref.subview %arg1[%arg6, %1] [1, 48] [1, 1] : memref<1024x768xf32, strided<[?, ?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %2 = cinm.compute -> f32
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 1, 8>} {
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<f32>
          linalg.generic {indexing_maps = [#map1], iterator_types = []} outs(%alloc_6 : memref<f32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          }
          linalg.generic {indexing_maps = [#map, #map, #map2], iterator_types = ["reduction"]} ins(%subview_4, %subview_5 : memref<48xf32, strided<[?], offset: ?>>, memref<48xf32, strided<[?], offset: ?>>) outs(%alloc_6 : memref<f32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %5 = arith.mulf %in, %in_7 : f32
            %6 = arith.addf %5, %out : f32
            linalg.yield %6 : f32
          }
          %3 = memref.load %alloc_6[] : memref<f32>
          %4 = arith.divf %3, %cst_1 : f32
          cinm.yield %4 : f32
        }
        memref.store %2, %alloc_2[%arg6] : memref<1024xf32>
      }
      cinm.compute
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
        %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
        linalg.generic {indexing_maps = [#map1], iterator_types = []} outs(%alloc_4 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        }
        linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction"]} ins(%alloc_2 : memref<1024xf32>) outs(%alloc_4 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.maxnumf %in, %out : f32
          linalg.yield %4 : f32
        }
        %2 = memref.load %alloc_4[] : memref<f32>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_2 : memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.subf %in, %2 : f32
          %5 = math.exp %4 : f32
          linalg.yield %5 : f32
        }
        linalg.generic {indexing_maps = [#map1], iterator_types = []} outs(%alloc_4 : memref<f32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        }
        linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction"]} ins(%alloc : memref<1024xf32>) outs(%alloc_4 : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.addf %in, %out : f32
          linalg.yield %4 : f32
        }
        %3 = memref.load %alloc_4[] : memref<f32>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc : memref<1024xf32>) outs(%alloc : memref<1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.divf %in, %3 : f32
          linalg.yield %4 : f32
        }
        cinm.yield
      }
      memref.copy %alloc, %alloc_2 : memref<1024xf32> to memref<1024xf32>
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<48xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_3 : memref<48xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      %subview = memref.subview %arg4[%1] [48] [1] : memref<768xf32> to memref<48xf32, strided<[1], offset: ?>>
      memref.copy %alloc_3, %subview : memref<48xf32> to memref<48xf32, strided<[1], offset: ?>>
      scf.for %arg6 = %c0 to %0 step %c1 {
        %subview_4 = memref.subview %arg2[%arg6, %1] [1, 48] [1, 1] : memref<1024x768xf32, strided<[?, ?], offset: ?>> to memref<48xf32, strided<[?], offset: ?>>
        %2 = memref.load %alloc_2[%arg6] : memref<1024xf32>
        cinm.compute
             attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 2, 32, 16>} {
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%subview, %subview_4 : memref<48xf32, strided<[1], offset: ?>>, memref<48xf32, strided<[?], offset: ?>>) outs(%alloc_3 : memref<48xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %3 = arith.mulf %in_5, %2 : f32
            %4 = arith.addf %in, %3 : f32
            linalg.yield %4 : f32
          }
          cinm.yield
        }
        memref.copy %alloc_3, %subview : memref<48xf32> to memref<48xf32, strided<[1], offset: ?>>
      }
    }
    return
  }
}
