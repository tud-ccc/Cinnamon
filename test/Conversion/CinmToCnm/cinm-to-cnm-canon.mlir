module {
  func.func @mm_dimm4_nopt(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024x256xi32>) -> tensor<8x256xi32> {
    %0 = cinm.compute_block on accelerator #upmem.array<4x128x1, <type = v1A, dimensions = 32x128x1>> (%arg2 = %arg0 : tensor<8x1024xi32>, %arg3 = %arg1 : tensor<1024x256xi32>) -> tensor<8x256xi32> {
      %cst = arith.constant dense<0> : tensor<4x128xi32>
      %1 = tensor.empty() : tensor<8x256xi32>
      %2 = affine.for %i = 0 to 8 step 4 iter_args(%acc = %1) -> (tensor<8x256xi32>) {
        %3 = affine.for %i_0 = 0 to 256 step 128 iter_args(%acc_1 = %acc) -> (tensor<8x256xi32>) {
          %extracted_slice = tensor.extract_slice %arg2[%i, 0] [4, 1024] [1, 1] : tensor<8x1024xi32> to tensor<4x1024xi32>
          %extracted_slice_2 = tensor.extract_slice %arg3[0, %i_0] [1024, 128] [1, 1] : tensor<1024x256xi32> to tensor<1024x128xi32>
          %4 = cinm.op.gemm %extracted_slice, %extracted_slice_2 into %cst : tensor<4x1024xi32>, tensor<1024x128xi32> into tensor<4x128xi32> -> tensor<4x128xi32>
          %inserted_slice = tensor.insert_slice %4 into %acc_1[%i, %i_0] [4, 128] [1, 1] : tensor<4x128xi32> into tensor<8x256xi32>
          affine.yield %inserted_slice : tensor<8x256xi32>
        }
        affine.yield %3 : tensor<8x256xi32>
      }
      cinm.yield %2 : tensor<8x256xi32>
    }
    return %0 : tensor<8x256xi32>
  }
  func.func @mm_dimm4_opt(%arg0: tensor<16x1024xi32>, %arg1: tensor<1024x128xi32>) -> tensor<16x128xi32> {
    %0 = cinm.compute_block on accelerator #upmem.array<4x128x1, <type = v1A, dimensions = 32x128x1>> (%arg2 = %arg0 : tensor<16x1024xi32>, %arg3 = %arg1 : tensor<1024x128xi32>) -> tensor<16x128xi32> {
      %cst = arith.constant dense<0> : tensor<4x128xi32>
      %1 = tensor.empty() : tensor<16x128xi32>
      %2 = affine.for %i = 0 to 16 step 4 iter_args(%acc = %1) -> (tensor<16x128xi32>) {
        %extracted_slice = tensor.extract_slice %arg2[%i, 0] [4, 1024] [1, 1] : tensor<16x1024xi32> to tensor<4x1024xi32>
        %3 = cinm.op.gemm %extracted_slice, %arg3 into %cst : tensor<4x1024xi32>, tensor<1024x128xi32> into tensor<4x128xi32> -> tensor<4x128xi32>
        %inserted_slice = tensor.insert_slice %3 into %acc[%i, 0] [4, 128] [1, 1] : tensor<4x128xi32> into tensor<16x128xi32>
        affine.yield %inserted_slice : tensor<16x128xi32>
      }
      cinm.yield %2 : tensor<16x128xi32>
    }
    return %0 : tensor<16x128xi32>
  }
  func.func @mm_dimm8_nopt(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024x128xi32>) -> tensor<8x128xi32> {
    %0 = cinm.compute_block on accelerator #upmem.array<8x128x1, <type = v1A, dimensions = 32x128x1>> (%arg2 = %arg0 : tensor<8x1024xi32>, %arg3 = %arg1 : tensor<1024x128xi32>) -> tensor<8x128xi32> {
      %cst = arith.constant dense<0> : tensor<8x128xi32>
      %1 = cinm.op.gemm %arg2, %arg3 into %cst : tensor<8x1024xi32>, tensor<1024x128xi32> into tensor<8x128xi32> -> tensor<8x128xi32>
      cinm.yield %1 : tensor<8x128xi32>
    }
    return %0 : tensor<8x128xi32>
  }
  func.func @mm_dimm8_opt(%arg0: tensor<16x1024xi32>, %arg1: tensor<1024x64xi32>) -> tensor<16x64xi32> {
    %0 = cinm.compute_block on accelerator #upmem.array<8x128x1, <type = v1A, dimensions = 32x128x1>> (%arg2 = %arg0 : tensor<16x1024xi32>, %arg3 = %arg1 : tensor<1024x64xi32>) -> tensor<16x64xi32> {
      %cst = arith.constant dense<0> : tensor<16x64xi32>
      %1 = cinm.op.gemm %arg2, %arg3 into %cst : tensor<16x1024xi32>, tensor<1024x64xi32> into tensor<16x64xi32> -> tensor<16x64xi32>
      cinm.yield %1 : tensor<16x64xi32>
    }
    return %0 : tensor<16x64xi32>
  }
  func.func @mm_dimm16_nopt(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024x512xi32>) -> tensor<8x512xi32> {
    %0 = cinm.compute_block on accelerator #upmem.array<16x64x1, <type = v1A, dimensions = 32x128x1>> (%arg2 = %arg0 : tensor<8x1024xi32>, %arg3 = %arg1 : tensor<1024x512xi32>) -> tensor<8x512xi32> {
      %cst = arith.constant dense<0> : tensor<2x512xi32>
      %1 = tensor.empty() : tensor<8x512xi32>
      %2 = affine.for %i = 0 to 8 step 2 iter_args(%acc = %1) -> (tensor<8x512xi32>) {
        %extracted_slice = tensor.extract_slice %arg2[%i, 0] [2, 1024] [1, 1] : tensor<8x1024xi32> to tensor<2x1024xi32>
        %3 = cinm.op.gemm %extracted_slice, %arg3 into %cst : tensor<2x1024xi32>, tensor<1024x512xi32> into tensor<2x512xi32> -> tensor<2x512xi32>
        %inserted_slice = tensor.insert_slice %3 into %acc[%i, 0] [2, 512] [1, 1] : tensor<2x512xi32> into tensor<8x512xi32>
        affine.yield %inserted_slice : tensor<8x512xi32>
      }
      cinm.yield %2 : tensor<8x512xi32>
    }
    return %0 : tensor<8x512xi32>
  }
  func.func @mm_dimm16_opt(%arg0: tensor<16x1024xi32>, %arg1: tensor<1024x512xi32>) -> tensor<16x512xi32> {
    %0 = cinm.compute_block on accelerator #upmem.array<16x64x1, <type = v1A, dimensions = 32x128x1>> (%arg2 = %arg0 : tensor<16x1024xi32>, %arg3 = %arg1 : tensor<1024x512xi32>) -> tensor<16x512xi32> {
      %cst = arith.constant dense<0> : tensor<16x64xi32>
      %1 = tensor.empty() : tensor<16x512xi32>
      %2 = affine.for %i = 0 to 512 step 64 iter_args(%acc = %1) -> (tensor<16x512xi32>) {
        %extracted_slice = tensor.extract_slice %arg3[0, %i] [1024, 64] [1, 1] : tensor<1024x512xi32> to tensor<1024x64xi32>
        %3 = cinm.op.gemm %arg2, %extracted_slice into %cst : tensor<16x1024xi32>, tensor<1024x64xi32> into tensor<16x64xi32> -> tensor<16x64xi32>
        %inserted_slice = tensor.insert_slice %3 into %acc[0, %i] [16, 64] [1, 1] : tensor<16x64xi32> into tensor<16x512xi32>
        affine.yield %inserted_slice : tensor<16x512xi32>
      }
      cinm.yield %2 : tensor<16x512xi32>
    }
    return %0 : tensor<16x512xi32>
  }
}

