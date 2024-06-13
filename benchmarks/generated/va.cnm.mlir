#map = affine_map<(d0, d1) -> (d0 * 128 + d1)>
module {
  func.func @va_8(%arg0: tensor<8x2097152xi32>, %arg1: tensor<8x2097152xi32>) {
    %cst = arith.constant dense<[8, 2097152]> : tensor<2xi64>
    %cst_0 = arith.constant dense<16777216> : tensor<1xi64>
    %reshape = tensor.reshape %arg0(%cst_0) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_1 = tensor.reshape %arg1(%cst_0) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %0 = tensor.empty() : tensor<16777216xi32>
    %1 = affine.for %arg2 = 0 to 16777216 step 16384 iter_args(%arg3 = %0) -> (tensor<16777216xi32>) {
      %extracted_slice = tensor.extract_slice %reshape[%arg2] [16384] [1] : tensor<16777216xi32> to tensor<16384xi32>
      %extracted_slice_2 = tensor.extract_slice %reshape_1[%arg2] [16384] [1] : tensor<16777216xi32> to tensor<16384xi32>
      %2 = cnm.workgroup : !cnm.workgroup<8x128>
      %cst_3 = arith.constant dense<0> : tensor<16384xi32>
      %cst_4 = arith.constant dense<[1024, 16]> : tensor<2xi64>
      %reshape_5 = tensor.reshape %extracted_slice(%cst_4) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %3 = cnm.alloc() for %2 : !cnm.buffer<16xi32 on 8x128, level 0>
      %4 = cnm.scatter %reshape_5 into %3[#map] of %2 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128, level 0>
      %reshape_6 = tensor.reshape %extracted_slice_2(%cst_4) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %5 = cnm.alloc() for %2 : !cnm.buffer<16xi32 on 8x128, level 0>
      %6 = cnm.scatter %reshape_6 into %5[#map] of %2 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128, level 0>
      %reshape_7 = tensor.reshape %cst_3(%cst_4) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %7 = cnm.alloc() for %2 : !cnm.buffer<16xi32 on 8x128, level 0>
      %8 = cnm.scatter %reshape_7 into %7[#map] of %2 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128, level 0>
      %9 = cnm.launch %2 in(%3, %5 : !cnm.buffer<16xi32 on 8x128, level 0>, !cnm.buffer<16xi32 on 8x128, level 0>) out(%7 : !cnm.buffer<16xi32 on 8x128, level 0>) on !cnm.workgroup<8x128> {
      ^bb0(%arg4: memref<16xi32>, %arg5: memref<16xi32>, %arg6: memref<16xi32>):
        linalg.add ins(%arg4, %arg5 : memref<16xi32>, memref<16xi32>) outs(%arg6 : memref<16xi32>)
      }
      %output, %token = cnm.gather %7[#map] of %2 : !cnm.buffer<16xi32 on 8x128, level 0> into tensor<1024x16xi32>
      %cst_8 = arith.constant dense<16384> : tensor<1xi64>
      %reshape_9 = tensor.reshape %output(%cst_8) : (tensor<1024x16xi32>, tensor<1xi64>) -> tensor<16384xi32>
      %inserted_slice = tensor.insert_slice %reshape_9 into %arg3[%arg2] [16384] [1] : tensor<16384xi32> into tensor<16777216xi32>
      affine.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
  func.func @va_16(%arg0: tensor<16x1048576xi32>, %arg1: tensor<16x1048576xi32>) {
    %cst = arith.constant dense<[16, 1048576]> : tensor<2xi64>
    %cst_0 = arith.constant dense<16777216> : tensor<1xi64>
    %reshape = tensor.reshape %arg0(%cst_0) : (tensor<16x1048576xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_1 = tensor.reshape %arg1(%cst_0) : (tensor<16x1048576xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %0 = tensor.empty() : tensor<16777216xi32>
    %1 = affine.for %arg2 = 0 to 16777216 step 32768 iter_args(%arg3 = %0) -> (tensor<16777216xi32>) {
      %extracted_slice = tensor.extract_slice %reshape[%arg2] [32768] [1] : tensor<16777216xi32> to tensor<32768xi32>
      %extracted_slice_2 = tensor.extract_slice %reshape_1[%arg2] [32768] [1] : tensor<16777216xi32> to tensor<32768xi32>
      %2 = cnm.workgroup : !cnm.workgroup<16x128>
      %cst_3 = arith.constant dense<0> : tensor<32768xi32>
      %cst_4 = arith.constant dense<[2048, 16]> : tensor<2xi64>
      %reshape_5 = tensor.reshape %extracted_slice(%cst_4) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %3 = cnm.alloc() for %2 : !cnm.buffer<16xi32 on 16x128, level 0>
      %4 = cnm.scatter %reshape_5 into %3[#map] of %2 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128, level 0>
      %reshape_6 = tensor.reshape %extracted_slice_2(%cst_4) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %5 = cnm.alloc() for %2 : !cnm.buffer<16xi32 on 16x128, level 0>
      %6 = cnm.scatter %reshape_6 into %5[#map] of %2 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128, level 0>
      %reshape_7 = tensor.reshape %cst_3(%cst_4) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %7 = cnm.alloc() for %2 : !cnm.buffer<16xi32 on 16x128, level 0>
      %8 = cnm.scatter %reshape_7 into %7[#map] of %2 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128, level 0>
      %9 = cnm.launch %2 in(%3, %5 : !cnm.buffer<16xi32 on 16x128, level 0>, !cnm.buffer<16xi32 on 16x128, level 0>) out(%7 : !cnm.buffer<16xi32 on 16x128, level 0>) on !cnm.workgroup<16x128> {
      ^bb0(%arg4: memref<16xi32>, %arg5: memref<16xi32>, %arg6: memref<16xi32>):
        linalg.add ins(%arg4, %arg5 : memref<16xi32>, memref<16xi32>) outs(%arg6 : memref<16xi32>)
      }
      %output, %token = cnm.gather %7[#map] of %2 : !cnm.buffer<16xi32 on 16x128, level 0> into tensor<2048x16xi32>
      %cst_8 = arith.constant dense<32768> : tensor<1xi64>
      %reshape_9 = tensor.reshape %output(%cst_8) : (tensor<2048x16xi32>, tensor<1xi64>) -> tensor<32768xi32>
      %inserted_slice = tensor.insert_slice %reshape_9 into %arg3[%arg2] [32768] [1] : tensor<32768xi32> into tensor<16777216xi32>
      affine.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
}

