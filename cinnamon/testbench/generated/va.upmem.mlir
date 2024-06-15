#map = affine_map<(d0, d1, d2) -> (d0 * 128 + d1 + d2)>
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
      %2 = upmem.alloc_dpus : !upmem.hierarchy<8x128x1>
      %cst_3 = arith.constant dense<0> : tensor<16384xi32>
      %cst_4 = arith.constant dense<[1024, 16]> : tensor<2xi64>
      %reshape_5 = tensor.reshape %extracted_slice(%cst_4) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %3 = builtin.unrealized_conversion_cast %reshape_5 : tensor<1024x16xi32> to memref<1024x16xi32>
      %c0 = arith.constant 0 : index
      %4 = upmem.scatter %3[16, #map] onto %2 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %reshape_6 = tensor.reshape %extracted_slice_2(%cst_4) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %5 = builtin.unrealized_conversion_cast %reshape_6 : tensor<1024x16xi32> to memref<1024x16xi32>
      %6 = upmem.scatter %5[16, #map] onto %2 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %reshape_7 = tensor.reshape %cst_3(%cst_4) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %7 = builtin.unrealized_conversion_cast %reshape_7 : tensor<1024x16xi32> to memref<1024x16xi32>
      %8 = upmem.scatter %7[16, #map] onto %2 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      upmem.launch %2 ranks(%arg4 upto %c8) dpus(%arg5 upto %c128) tasklets(%arg6 upto %c1) on !upmem.hierarchy<8x128x1> {
        %10 = upmem.dpu_heap_base_addr : index
        %c16 = arith.constant 16 : index
        %11 = upmem.pwram_alloc : memref<16xi32>
        %12 = arith.addi %10, %c16 : index
        %13 = upmem.pwram_alloc : memref<16xi32>
        %14 = arith.addi %12, %c16 : index
        %15 = upmem.pwram_alloc : memref<16xi32>
        upmem.memcpy  mram_to_wram %11, %c16, %10 : memref<16xi32>, index, index
        upmem.memcpy  mram_to_wram %13, %c16, %12 : memref<16xi32>, index, index
        linalg.add ins(%11, %13 : memref<16xi32>, memref<16xi32>) outs(%15 : memref<16xi32>)
        upmem.memcpy  wram_to_mram %15, %c16, %14 : memref<16xi32>, index, index
        upmem.terminator
      }
      %alloc = memref.alloc() : memref<1024x16xi32>
      upmem.gather %alloc[16, #map] from %2 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %9 = builtin.unrealized_conversion_cast %alloc : memref<1024x16xi32> to tensor<1024x16xi32>
      %cst_8 = arith.constant dense<16384> : tensor<1xi64>
      %reshape_9 = tensor.reshape %9(%cst_8) : (tensor<1024x16xi32>, tensor<1xi64>) -> tensor<16384xi32>
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
      %2 = upmem.alloc_dpus : !upmem.hierarchy<16x128x1>
      %cst_3 = arith.constant dense<0> : tensor<32768xi32>
      %cst_4 = arith.constant dense<[2048, 16]> : tensor<2xi64>
      %reshape_5 = tensor.reshape %extracted_slice(%cst_4) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %3 = builtin.unrealized_conversion_cast %reshape_5 : tensor<2048x16xi32> to memref<2048x16xi32>
      %c0 = arith.constant 0 : index
      %4 = upmem.scatter %3[16, #map] onto %2 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %reshape_6 = tensor.reshape %extracted_slice_2(%cst_4) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %5 = builtin.unrealized_conversion_cast %reshape_6 : tensor<2048x16xi32> to memref<2048x16xi32>
      %6 = upmem.scatter %5[16, #map] onto %2 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %reshape_7 = tensor.reshape %cst_3(%cst_4) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %7 = builtin.unrealized_conversion_cast %reshape_7 : tensor<2048x16xi32> to memref<2048x16xi32>
      %8 = upmem.scatter %7[16, #map] onto %2 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %c16 = arith.constant 16 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      upmem.launch %2 ranks(%arg4 upto %c16) dpus(%arg5 upto %c128) tasklets(%arg6 upto %c1) on !upmem.hierarchy<16x128x1> {
        %10 = upmem.dpu_heap_base_addr : index
        %c16_10 = arith.constant 16 : index
        %11 = upmem.pwram_alloc : memref<16xi32>
        %12 = arith.addi %10, %c16_10 : index
        %13 = upmem.pwram_alloc : memref<16xi32>
        %14 = arith.addi %12, %c16_10 : index
        %15 = upmem.pwram_alloc : memref<16xi32>
        upmem.memcpy  mram_to_wram %11, %c16_10, %10 : memref<16xi32>, index, index
        upmem.memcpy  mram_to_wram %13, %c16_10, %12 : memref<16xi32>, index, index
        linalg.add ins(%11, %13 : memref<16xi32>, memref<16xi32>) outs(%15 : memref<16xi32>)
        upmem.memcpy  wram_to_mram %15, %c16_10, %14 : memref<16xi32>, index, index
        upmem.terminator
      }
      %alloc = memref.alloc() : memref<2048x16xi32>
      upmem.gather %alloc[16, #map] from %2 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %9 = builtin.unrealized_conversion_cast %alloc : memref<2048x16xi32> to tensor<2048x16xi32>
      %cst_8 = arith.constant dense<32768> : tensor<1xi64>
      %reshape_9 = tensor.reshape %9(%cst_8) : (tensor<2048x16xi32>, tensor<1xi64>) -> tensor<32768xi32>
      %inserted_slice = tensor.insert_slice %reshape_9 into %arg3[%arg2] [32768] [1] : tensor<32768xi32> into tensor<16777216xi32>
      affine.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
}

