#map = affine_map<(d0, d1, d2) -> (d0 * 128 + d1 + d2)>
module {
  func.func @va_8(%arg0: memref<8x2097152xi32>, %arg1: memref<8x2097152xi32>) {
    %cst = arith.constant dense<16384> : tensor<1xi64>
    %cst_0 = arith.constant dense<0> : tensor<1024x16xi32>
    %cst_1 = arith.constant dense<[1024, 16]> : tensor<2xi64>
    %c16384 = arith.constant 16384 : index
    %c16777216 = arith.constant 16777216 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant dense<16777216> : tensor<1xi64>
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<8x2097152xi32> to tensor<8x2097152xi32>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<8x2097152xi32> to tensor<8x2097152xi32>
    %reshape = tensor.reshape %1(%cst_2) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_3 = tensor.reshape %0(%cst_2) : (tensor<8x2097152xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %2 = tensor.empty() : tensor<16777216xi32>
    %3 = scf.for %arg2 = %c0 to %c16777216 step %c16384 iter_args(%arg3 = %2) -> (tensor<16777216xi32>) {
      %extracted_slice = tensor.extract_slice %reshape[%arg2] [16384] [1] : tensor<16777216xi32> to tensor<16384xi32>
      %extracted_slice_4 = tensor.extract_slice %reshape_3[%arg2] [16384] [1] : tensor<16777216xi32> to tensor<16384xi32>
      %4 = upmem.alloc_dpus : !upmem.hierarchy<8x128x1>
      %reshape_5 = tensor.reshape %extracted_slice(%cst_1) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %5 = builtin.unrealized_conversion_cast %reshape_5 : tensor<1024x16xi32> to memref<1024x16xi32>
      %6 = upmem.scatter %5[16, #map] onto %4 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %reshape_6 = tensor.reshape %extracted_slice_4(%cst_1) : (tensor<16384xi32>, tensor<2xi64>) -> tensor<1024x16xi32>
      %7 = builtin.unrealized_conversion_cast %reshape_6 : tensor<1024x16xi32> to memref<1024x16xi32>
      %8 = upmem.scatter %7[16, #map] onto %4 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %9 = builtin.unrealized_conversion_cast %cst_0 : tensor<1024x16xi32> to memref<1024x16xi32>
      %10 = upmem.scatter %9[16, #map] onto %4 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      upmem.launch_func  @main::@main %4 : <8x128x1> 
      %alloc = memref.alloc() : memref<1024x16xi32>
      upmem.gather %alloc[16, #map] from %4 at %c0 : memref<1024x16xi32> onto !upmem.hierarchy<8x128x1>
      %11 = builtin.unrealized_conversion_cast %alloc : memref<1024x16xi32> to tensor<1024x16xi32>
      %reshape_7 = tensor.reshape %11(%cst) : (tensor<1024x16xi32>, tensor<1xi64>) -> tensor<16384xi32>
      %inserted_slice = tensor.insert_slice %reshape_7 into %arg3[%arg2] [16384] [1] : tensor<16384xi32> into tensor<16777216xi32>
      scf.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
  upmem.module @main{
    upmem.func @main() kernel {
      %0 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %1 = upmem.dpu_heap_base_addr : index
      %2 = upmem.pwram_alloc : memref<16xi32>
      %3 = arith.addi %1, %c16 : index
      %4 = upmem.pwram_alloc : memref<16xi32>
      %5 = arith.addi %3, %c16 : index
      %6 = upmem.pwram_alloc : memref<16xi32>
      upmem.memcpy  mram_to_wram %2, %c16, %1 : memref<16xi32>, index, index
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %7 = memref.load %2[%arg0] : memref<16xi32>
        %8 = memref.load %4[%arg0] : memref<16xi32>
        %9 = arith.addi %7, %8 : i32
        memref.store %9, %6[%arg0] : memref<16xi32>
      }
      upmem.memcpy  wram_to_mram %6, %c16, %5 : memref<16xi32>, index, index
      upmem.return
    }
  }
  func.func @va_16(%arg0: memref<16x1048576xi32>, %arg1: memref<16x1048576xi32>) {
    %cst = arith.constant dense<32768> : tensor<1xi64>
    %cst_0 = arith.constant dense<0> : tensor<2048x16xi32>
    %cst_1 = arith.constant dense<[2048, 16]> : tensor<2xi64>
    %c32768 = arith.constant 32768 : index
    %c16777216 = arith.constant 16777216 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant dense<16777216> : tensor<1xi64>
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<16x1048576xi32> to tensor<16x1048576xi32>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<16x1048576xi32> to tensor<16x1048576xi32>
    %reshape = tensor.reshape %1(%cst_2) : (tensor<16x1048576xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %reshape_3 = tensor.reshape %0(%cst_2) : (tensor<16x1048576xi32>, tensor<1xi64>) -> tensor<16777216xi32>
    %2 = tensor.empty() : tensor<16777216xi32>
    %3 = scf.for %arg2 = %c0 to %c16777216 step %c32768 iter_args(%arg3 = %2) -> (tensor<16777216xi32>) {
      %extracted_slice = tensor.extract_slice %reshape[%arg2] [32768] [1] : tensor<16777216xi32> to tensor<32768xi32>
      %extracted_slice_4 = tensor.extract_slice %reshape_3[%arg2] [32768] [1] : tensor<16777216xi32> to tensor<32768xi32>
      %4 = upmem.alloc_dpus : !upmem.hierarchy<16x128x1>
      %reshape_5 = tensor.reshape %extracted_slice(%cst_1) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %5 = builtin.unrealized_conversion_cast %reshape_5 : tensor<2048x16xi32> to memref<2048x16xi32>
      %6 = upmem.scatter %5[16, #map] onto %4 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %reshape_6 = tensor.reshape %extracted_slice_4(%cst_1) : (tensor<32768xi32>, tensor<2xi64>) -> tensor<2048x16xi32>
      %7 = builtin.unrealized_conversion_cast %reshape_6 : tensor<2048x16xi32> to memref<2048x16xi32>
      %8 = upmem.scatter %7[16, #map] onto %4 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %9 = builtin.unrealized_conversion_cast %cst_0 : tensor<2048x16xi32> to memref<2048x16xi32>
      %10 = upmem.scatter %9[16, #map] onto %4 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      upmem.launch_func  @main_0::@main %4 : <16x128x1> 
      %alloc = memref.alloc() : memref<2048x16xi32>
      upmem.gather %alloc[16, #map] from %4 at %c0 : memref<2048x16xi32> onto !upmem.hierarchy<16x128x1>
      %11 = builtin.unrealized_conversion_cast %alloc : memref<2048x16xi32> to tensor<2048x16xi32>
      %reshape_7 = tensor.reshape %11(%cst) : (tensor<2048x16xi32>, tensor<1xi64>) -> tensor<32768xi32>
      %inserted_slice = tensor.insert_slice %reshape_7 into %arg3[%arg2] [32768] [1] : tensor<32768xi32> into tensor<16777216xi32>
      scf.yield %inserted_slice : tensor<16777216xi32>
    }
    return
  }
  upmem.module @main_0{
    upmem.func @main() kernel {
      %0 = upmem.tasklet_id : index
      %c16 = arith.constant 16 : index
      %1 = upmem.dpu_heap_base_addr : index
      %2 = upmem.pwram_alloc : memref<16xi32>
      %3 = arith.addi %1, %c16 : index
      %4 = upmem.pwram_alloc : memref<16xi32>
      %5 = arith.addi %3, %c16 : index
      %6 = upmem.pwram_alloc : memref<16xi32>
      upmem.memcpy  mram_to_wram %2, %c16, %1 : memref<16xi32>, index, index
      upmem.memcpy  mram_to_wram %4, %c16, %3 : memref<16xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %7 = memref.load %2[%arg0] : memref<16xi32>
        %8 = memref.load %4[%arg0] : memref<16xi32>
        %9 = arith.addi %7, %8 : i32
        memref.store %9, %6[%arg0] : memref<16xi32>
      }
      upmem.memcpy  wram_to_mram %6, %c16, %5 : memref<16xi32>, index, index
      upmem.return
    }
  }
}

