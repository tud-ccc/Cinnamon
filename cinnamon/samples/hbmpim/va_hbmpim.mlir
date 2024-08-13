module {
  func.func @va(%arg0: tensor<16x8192xi32>, %arg1: tensor<16x8192xi32>, %arg2: tensor<16x8192xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c4096 = arith.constant 4096 : index
    affine.for %arg3 = 0 to %c16 {
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [1, 8192] [1, 1] : tensor<16x8192xi32> to tensor<8192xi32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [1, 8192] [1, 1] : tensor<16x8192xi32> to tensor<8192xi32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, 0] [1, 8192] [1, 1] : tensor<16x8192xi32> to tensor<8192xi32>
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      hbmpim.set_dev_config : !hbmpim.device_configuration<16x64x1x8>
      %dim = arith.divui %c8192, %c16 : index
      // hbmpim.execute_el_wise %dim, ALL_BANK, ADD, %c0, %c256, %c128 : index, index, index, index
    }
    return
  }
}