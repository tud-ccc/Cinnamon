module {
  func.func @gemv(%arg0: tensor<1x8192x8192xi32>, %arg1: tensor<1x8192xi32>, %arg2: tensor<1x8192xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c8192 = arith.constant 8192 : index
    %c4096 = arith.constant 4096 : index
    %0 = hbmpim.set_dev_config : !hbmpim.configuration<16x64x1x8>
    %1 = builtin.unrealized_conversion_cast %0 : !hbmpim.configuration<16x64x1x8> to !cnm.workgroup<16x64x1x8>
    %cst = arith.constant dense<[1024, 8]> : tensor<2xi64>
    %cst_0 = arith.constant dense<8192> : tensor<2xi64>
    %reshape = tensor.reshape %arg0(%cst_0) : (tensor<1x8192x8192xi32>, tensor<2xi64>) -> tensor<8192x8192xi32>
    %2 = builtin.unrealized_conversion_cast %reshape : tensor<8192x8192xi32> to memref<8192x8192xi32>
    %cst_1 = arith.constant dense<8192> : tensor<1xi64>
    %reshape_2 = tensor.reshape %arg1(%cst_1) : (tensor<1x8192xi32>, tensor<1xi64>) -> tensor<8192xi32>
    %3 = builtin.unrealized_conversion_cast %reshape_2 : tensor<8192xi32> to memref<8192xi32>
    %cst_3 = arith.constant dense<8192> : tensor<1xi64>
    %reshape_4 = tensor.reshape %arg2(%cst_3) : (tensor<1x8192xi32>, tensor<1xi64>) -> tensor<8192xi32>
    %4 = builtin.unrealized_conversion_cast %reshape_4 : tensor<8192xi32> to memref<8192xi32>
    hbmpim.launch %0 in(%2, %3 : memref<8192x8192xi32>, memref<8192xi32>) out(%4 : memref<8192xi32>) on !hbmpim.configuration<16x64x1x8> {
    ^bb0(%arg3: memref<8192x8192xi32>, %arg4: memref<8192xi32>, %arg5: memref<8192xi32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      hbmpim.preload_gemv %arg3, %c0, %c0 : memref<8192x8192xi32>, index, index
      hbmpim.preload_gemv %arg4, %c0, %c0 : memref<8192xi32>, index, index
      %c8192_5 = arith.constant 8192 : index
      %c8192_6 = arith.constant 8192 : index
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      %5 = arith.divui %c8192_6, %c512 : index
      %6 = arith.divui %c8192_5, %c8 : index
      %7 = arith.muli %5, %6 : index
      %8 = arith.muli %c8, %c8 : index
      %9 = arith.muli %8, %c2 : index
      %10 = arith.divui %7, %9 : index
      hbmpim.execute_gemv %c1, %c8192_5, %c8192_6, %5, %6 {is_tree = false} : index, index, index, index, index
      hbmpim.read_result %arg5, ODD_BANK, %c8192_6, %c0, %c0, %10 : memref<8192xi32>, index, index, index, index
    }
    return
  }
}