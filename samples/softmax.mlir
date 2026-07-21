#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0) -> ()>


func.func @softie(%arg17 : index, %arg1: index, %arg2 : memref<6x1024x768xf32> {bufferization.writable=true}) {

  %cst_2 = arith.constant 1.000000e+04 : f32
  %cst_24 = arith.constant 1.000000e+04 : f32
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
  %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 4.800000e+01 : f32

    %extracted_slice = tensor.extract_slice %arg4[%arg0, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>

        %extracted_slice_8 = tensor.extract_slice %arg19[%arg16, %arg1, 0] [1, 1, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<768xf32>

      %4:3 = cinm.compute -> tensor<768xf32>, tensor<768xf32>, tensor<768xf32>
           attributes {bufferSizesInBytes = array<i64: 0, 65536, 0>, workgroupShape = array<i64: 1, 6, 8>} {
        %13 = tensor.empty() : tensor<768xf32>
        %14 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%13 : tensor<768xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<768xf32>
      
        %16 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_5, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%14 : tensor<768xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %20 = arith.mulf %in, %in_17 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<768xf32>
        %18 = bufferization.materialize_in_destination %16 in %extracted_slice_7 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        // %17 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_6, %3 : tensor<768x768xf32>, tensor<768xf32>) outs(%14 : tensor<768xf32>) {
        // ^bb0(%in: f32, %in_17: f32, %out: f32):
        //   %20 = arith.mulf %in, %in_17 : f32
        //   %21 = arith.addf %out, %20 : f32
        //   linalg.yield %21 : f32
        // } -> tensor<768xf32>
        // %19 = bufferization.materialize_in_destination %17 in %extracted_slice_8 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
        cinm.yield  %18 : tensor<768xf32> //, tensor<768xf32>, tensor<768xf32>
        // cinm.yield %15, %18, %19 : tensor<768xf32>, tensor<768xf32>, tensor<768xf32>
      }
  %subview_16 = memref.subview %arg2[%arg17, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
  // %2 = cinm.compute -> memref<768xf32, strided<[1], offset: ?>> {
    %subview_33 = memref.subview %arg2[%arg17, %arg1, 0] [1, 1, 768] [1, 1, 1] : memref<6x1024x768xf32> to memref<768xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview_33 : memref<768xf32, strided<[1], offset: ?>>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview_33 : memref<768xf32, strided<[1], offset: ?>>) {
    ^bb0(%out: f32):
      linalg.yield %cst_24 : f32
    }
    memref.copy %subview_33, %alloc_9 : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32>
    // cinm.yield %subview_33 : memref<768xf32, strided<[1], offset: ?>>
  // }

  memref.copy %alloc_9, %subview_16 : memref<768xf32> to memref<768xf32, strided<[1], offset: ?>>

	return 
}

// func.func @softmax(%vec : tensor<1024xf32>{bufferization.writable=true}) -> tensor<1024xf32> {
// 	%r = cinm.compute -> tensor<1024xf32> {
// 		%max = cinm.op.reduce max (%vec) : tensor<1024xf32> -> f32
//     %maxv = tensor.splat %max : tensor<1024xf32>
// 		%t = cinm.op.elementwise sub %vec, %maxv  into %vec: tensor<1024xf32> into tensor<1024xf32>
//     %e = cinm.op.elementwise exp %t  into %vec: tensor<1024xf32> into tensor<1024xf32>
// 		%s = cinm.op.reduce add (%e) : tensor<1024xf32> -> f32
//     %sumv = tensor.splat %s : tensor<1024xf32>
// 		%r = cinm.op.elementwise div %e, %sumv into %vec : tensor<1024xf32> into tensor<1024xf32>
// 		cinm.yield %r : tensor<1024xf32>
// 	}

// 	return %r : tensor<1024xf32>
// }