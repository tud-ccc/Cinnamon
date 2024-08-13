#map = affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 8+ d2*8 + d3)>
module {
  func.func @gemv(%arg0: tensor<1024xi8>) -> tensor<1024xi8> {
    %0 = cnm.workgroup : !cnm.workgroup<16x64x1x8>
    %cst = arith.constant dense<0> : tensor<1024xi8>
    %1 = cnm.alloc() for %0 : !cnm.buffer<i8 on 16x64x1x8, level 0>
    cnm.scatter %arg0 into %1[#map] of %0 : tensor<1024xi8> into !cnm.buffer<i8 on 16x64x1x8, level 0>
    %2 = cnm.alloc() for %0 : !cnm.buffer<i8 on 16x64x1x8, level 0>
    cnm.scatter %arg0 into %2[#map] of %0 : tensor<1024xi8> into !cnm.buffer<i8 on 16x64x1x8, level 0>
    %3 = cnm.alloc() for %0 : !cnm.buffer<i8 on 16x64x1x8, level 0>
    cnm.scatter %cst into %3[#map] of %0 : tensor<1024xi8> into !cnm.buffer<i8 on 16x64x1x8, level 0>
    cnm.launch %0 in(%1, %2 : !cnm.buffer<i8 on 16x64x1x8, level 0>, !cnm.buffer<i8 on 16x64x1x8, level 0>) out(%3 : !cnm.buffer<i8 on 16x64x1x8, level 0>) on !cnm.workgroup<16x64x1x8> {
    ^bb0(%arg1: memref<i8>, %arg2: memref<i8>, %arg3: memref<i8>):
      linalg.add ins(%arg1, %arg2 : memref<i8>, memref<i8>) outs(%arg3 : memref<i8>)
    }
    %4 = tensor.empty() : tensor<1024xi8>
    %5 = cnm.gather %3[#map] of %0 into %4 : !cnm.buffer<i8 on 16x64x1x8, level 0> into tensor<1024xi8>
    %cst_0 = arith.constant dense<1024> : tensor<1xi64>
    %reshape = tensor.reshape %5(%cst_0) : (tensor<1024xi8>, tensor<1xi64>) -> tensor<1024xi8>
    cnm.free_workgroup %0 : !cnm.workgroup<16x64x1x8>
    return %reshape : tensor<1024xi8>
  }
}