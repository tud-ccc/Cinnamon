module {
  func.func @va(%arg0: tensor<8192x8192xi32>, %arg1: tensor<8192x8192xi32>, %arg2: tensor<8192x8192xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c8192 = arith.constant 8192 : index
    %c4096 = arith.constant 4096 : index
    affine.for %arg3 = 0 to %c4 {
      affine.for %arg4 = 0 to %c4 {
        %0 = arith.muli %arg3, %c8192 : index
        %1 = arith.muli %arg4, %c8192 : index
        %extracted_slice = tensor.extract_slice %arg0[%0, %1] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
        %extracted_slice_0 = tensor.extract_slice %arg1[%0, %1] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
        %extracted_slice_1 = tensor.extract_slice %arg2[%0, %1] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
        %2 = hbmpim.set_dev_config : !hbmpim.configuration<16x64x1x8>
        %3 = builtin.unrealized_conversion_cast %2 : !hbmpim.configuration<16x64x1x8> to !cnm.workgroup<16x64x1x8>
        %cst = arith.constant dense<[1024, 8]> : tensor<2xi64>
        %reshape = tensor.reshape %extracted_slice(%cst) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
        %reshape_2 = tensor.reshape %extracted_slice_0(%cst) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
        %reshape_3 = tensor.reshape %extracted_slice_1(%cst) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c0_4 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %4 = builtin.unrealized_conversion_cast %reshape : tensor<1024x8xi32> to memref<1024x8xi32>
        hbmpim.preload_no_replacement %2, %4, %c0_4, %c0 : !hbmpim.configuration<16x64x1x8>, memref<1024x8xi32>, index, index
        %5 = builtin.unrealized_conversion_cast %reshape_2 : tensor<1024x8xi32> to memref<1024x8xi32>
        hbmpim.preload_no_replacement %2, %5, %c0_4, %c0 : !hbmpim.configuration<16x64x1x8>, memref<1024x8xi32>, index, index
        hbmpim.execute_el_wise %2, %c512, ALL_BANK, ADD, %c0_4, %c128, %c256 : !hbmpim.configuration<16x64x1x8>, index, index, index, index
        %6 = builtin.unrealized_conversion_cast %reshape_3 : tensor<1024x8xi32> to memref<1024x8xi32>
        hbmpim.read_data %2, %6, %c128, %c0 : !hbmpim.configuration<16x64x1x8>, memref<1024x8xi32>, index, index
      }
    }
    return
  }
}