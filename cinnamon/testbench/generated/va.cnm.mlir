#map = affine_map<(d0, d1, d2) -> (d0 * 128 + d1 + d2)>
module {
  func.func @va_8(%arg0: memref<8x2097152xi32>, %arg1: memref<8x2097152xi32>) {
    %0 = bufferization.to_tensor %arg1 : memref<8x2097152xi32>
    %1 = bufferization.to_tensor %arg0 : memref<8x2097152xi32>
    %cst = arith.constant dense<[8, 2097152]> : tensor<2xi64>
    %cst_0 = arith.constant dense<16777216> : tensor<1xi64>
    %reshape = tensor.reshape %1(%cst_0) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_1 = tensor.reshape %0(%cst_0) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %2 = tensor.empty() : tensor<16777216xi32>
    %c0 = arith.constant 0 : index
    %c16777216 = arith.constant 16777216 : index
    %c16384 = arith.constant 16384 : index
    %3 = scf.for %arg2 = %c0 to %c16777216 step %c16384 iter_args(%arg3 = %2) -> (tensor<16777216xi32>) {
      %extracted_slice = tensor.extract_slice %reshape[%arg2] [16384] [1] : tensor<16777216xi32> to tensor<16384xi32>
      %extracted_slice_2 = tensor.extract_slice %reshape_1[%arg2] [16384] [1] : tensor<16777216xi32> to tensor<16384xi32>
      %4 = cnm.workgroup : !cnm.workgroup<8x128x1>
      %cst_3 = arith.constant dense<[1024, 16]> : tensor<2xi64>
      %reshape_4 = tensor.reshape %extracted_slice(%cst_3) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %5 = cnm.alloc() for %4 : !cnm.buffer<16xi32 on 8x128x1, level 0>
      %6 = cnm.scatter %reshape_4 into %5[#map] of %4 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128x1, level 0>
      %reshape_5 = tensor.reshape %extracted_slice_2(%cst_3) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %7 = cnm.alloc() for %4 : !cnm.buffer<16xi32 on 8x128x1, level 0>
      %8 = cnm.scatter %reshape_5 into %7[#map] of %4 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128x1, level 0>
      %cst_6 = arith.constant dense<0> : tensor<1024x16xi32>
      %9 = cnm.alloc() for %4 : !cnm.buffer<16xi32 on 8x128x1, level 0>
      %10 = cnm.scatter %cst_6 into %9[#map] of %4 : tensor<1024x16xi32> into !cnm.buffer<16xi32 on 8x128x1, level 0>
      %11 = cnm.launch %4 in(%5, %7 : !cnm.buffer<16xi32 on 8x128x1, level 0>, !cnm.buffer<16xi32 on 8x128x1, level 0>) out(%9 : !cnm.buffer<16xi32 on 8x128x1, level 0>) on !cnm.workgroup<8x128x1> {
      ^bb0(%arg4: memref<16xi32>, %arg5: memref<16xi32>, %arg6: memref<16xi32>):
        linalg.add ins(%arg4, %arg5 : memref<16xi32>, memref<16xi32>) outs(%arg6 : memref<16xi32>)
      }
      %output, %token = cnm.gather %9[#map] of %4 : !cnm.buffer<16xi32 on 8x128x1, level 0> into tensor<1024x16xi32>
      %cst_7 = arith.constant dense<16384> : tensor<1xi64>
      %reshape_8 = tensor.reshape %output(%cst_7) : (tensor<1024x16xi32>, tensor<1xi64>) -> tensor<16384xi32>
      %inserted_slice = tensor.insert_slice %reshape_8 into %arg3[%arg2] [16384] [1] : tensor<16384xi32> into tensor<16777216xi32>
      scf.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
  func.func @va_16(%arg0: memref<16x1048576xi32>, %arg1: memref<16x1048576xi32>) {
    %0 = bufferization.to_tensor %arg1 : memref<16x1048576xi32>
    %1 = bufferization.to_tensor %arg0 : memref<16x1048576xi32>
    %cst = arith.constant dense<[16, 1048576]> : tensor<2xi64>
    %cst_0 = arith.constant dense<16777216> : tensor<1xi64>
    %reshape = tensor.reshape %1(%cst_0) : (tensor<16x1048576xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_1 = tensor.reshape %0(%cst_0) : (tensor<16x1048576xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %2 = tensor.empty() : tensor<16777216xi32>
    %c0 = arith.constant 0 : index
    %c16777216 = arith.constant 16777216 : index
    %c32768 = arith.constant 32768 : index
    %3 = scf.for %arg2 = %c0 to %c16777216 step %c32768 iter_args(%arg3 = %2) -> (tensor<16777216xi32>) {
      %extracted_slice = tensor.extract_slice %reshape[%arg2] [32768] [1] : tensor<16777216xi32> to tensor<32768xi32>
      %extracted_slice_2 = tensor.extract_slice %reshape_1[%arg2] [32768] [1] : tensor<16777216xi32> to tensor<32768xi32>
      %4 = cnm.workgroup : !cnm.workgroup<16x128x1>
      %cst_3 = arith.constant dense<[2048, 16]> : tensor<2xi64>
      %reshape_4 = tensor.reshape %extracted_slice(%cst_3) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %5 = cnm.alloc() for %4 : !cnm.buffer<16xi32 on 16x128x1, level 0>
      %6 = cnm.scatter %reshape_4 into %5[#map] of %4 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128x1, level 0>
      %reshape_5 = tensor.reshape %extracted_slice_2(%cst_3) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %7 = cnm.alloc() for %4 : !cnm.buffer<16xi32 on 16x128x1, level 0>
      %8 = cnm.scatter %reshape_5 into %7[#map] of %4 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128x1, level 0>
      %cst_6 = arith.constant dense<0> : tensor<2048x16xi32>
      %9 = cnm.alloc() for %4 : !cnm.buffer<16xi32 on 16x128x1, level 0>
      %10 = cnm.scatter %cst_6 into %9[#map] of %4 : tensor<2048x16xi32> into !cnm.buffer<16xi32 on 16x128x1, level 0>
      %11 = cnm.launch %4 in(%5, %7 : !cnm.buffer<16xi32 on 16x128x1, level 0>, !cnm.buffer<16xi32 on 16x128x1, level 0>) out(%9 : !cnm.buffer<16xi32 on 16x128x1, level 0>) on !cnm.workgroup<16x128x1> {
      ^bb0(%arg4: memref<16xi32>, %arg5: memref<16xi32>, %arg6: memref<16xi32>):
        linalg.add ins(%arg4, %arg5 : memref<16xi32>, memref<16xi32>) outs(%arg6 : memref<16xi32>)
      }
      %output, %token = cnm.gather %9[#map] of %4 : !cnm.buffer<16xi32 on 16x128x1, level 0> into tensor<2048x16xi32>
      %cst_7 = arith.constant dense<32768> : tensor<1xi64>
      %reshape_8 = tensor.reshape %output(%cst_7) : (tensor<2048x16xi32>, tensor<1xi64>) -> tensor<32768xi32>
      %inserted_slice = tensor.insert_slice %reshape_8 into %arg3[%arg2] [32768] [1] : tensor<32768xi32> into tensor<16777216xi32>
      scf.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
}

