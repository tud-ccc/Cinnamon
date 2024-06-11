module {
  func.func @mm_dimm1_nopt(%arg0: tensor<1x1024xi32>, %arg1: tensor<1024x512xi32>) -> tensor<1x512xi32> {
    %c0 = arith.constant 0 : index
    %0 = cinm.compute attributes {workgroupShape = array<i64: 1, 128>} -> tensor<1x512xi32> {
      %1 = tensor.empty() : tensor<1x512xi32>
      %2 = affine.for %arg2 = 0 to 512 step 128 iter_args(%arg3 = %1) -> (tensor<1x512xi32>) {
        %c0_i32 = arith.constant 0 : i32
        %splat = tensor.splat %c0_i32 : tensor<1x128xi32>
        %3 = affine.for %arg4 = 0 to 1024 step 32 iter_args(%arg5 = %splat) -> (tensor<1x128xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[%c0, %arg4] [1, 32] [1, 1] : tensor<1x1024xi32> to tensor<1x32xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, %arg2] [32, 128] [1, 1] : tensor<1024x512xi32> to tensor<32x128xi32>
          %4 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {cinm.notile} : (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
          affine.yield %4 : tensor<1x128xi32>
        }
        %inserted_slice = tensor.insert_slice %3 into %arg3[%c0, %arg2] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<1x512xi32>
        affine.yield %inserted_slice : tensor<1x512xi32>
      }
      cinm.yield %2 : tensor<1x512xi32>
    }
    return %0 : tensor<1x512xi32>
  }
  func.func @mm_dimm1_opt(%arg0: tensor<16x64xi32>, %arg1: tensor<64x512xi32>) -> tensor<16x512xi32> {
    %0 = cinm.compute attributes {workgroupShape = array<i64: 1, 128>} -> tensor<16x512xi32> {
      %1 = tensor.empty() : tensor<16x512xi32>
      %2 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %1) -> (tensor<16x512xi32>) {
        %3 = affine.for %arg4 = 0 to 512 step 128 iter_args(%arg5 = %arg3) -> (tensor<16x512xi32>) {
          %c0_i32 = arith.constant 0 : i32
          %splat = tensor.splat %c0_i32 : tensor<1x128xi32>
          %4 = affine.for %arg6 = 0 to 64 step 32 iter_args(%arg7 = %splat) -> (tensor<1x128xi32>) {
            %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg6] [1, 32] [1, 1] : tensor<16x64xi32> to tensor<1x32xi32>
            %extracted_slice_0 = tensor.extract_slice %arg1[%arg6, %arg4] [32, 128] [1, 1] : tensor<64x512xi32> to tensor<32x128xi32>
            %5 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg7 {cinm.notile} : (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
            affine.yield %5 : tensor<1x128xi32>
          }
          %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<16x512xi32>
          affine.yield %inserted_slice : tensor<16x512xi32>
        }
        affine.yield %3 : tensor<16x512xi32>
      }
      cinm.yield %2 : tensor<16x512xi32>
    }
    return %0 : tensor<16x512xi32>
  }
  func.func @mm_dimm2_nopt(%arg0: tensor<1x1024xi32>, %arg1: tensor<1024x256xi32>) -> tensor<1x256xi32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = cinm.compute attributes {workgroupShape = array<i64: 2, 128>} -> tensor<1x256xi32> {
      %1 = tensor.empty() : tensor<1x256xi32>
      %c0_i32 = arith.constant 0 : i32
      %splat = tensor.splat %c0_i32 : tensor<1x256xi32>
      %2 = affine.for %arg2 = 0 to 1024 step 32 iter_args(%arg3 = %splat) -> (tensor<1x256xi32>) {
        %extracted_slice = tensor.extract_slice %arg0[%c0, %arg2] [1, 32] [1, 1] : tensor<1x1024xi32> to tensor<1x32xi32>
        %extracted_slice_1 = tensor.extract_slice %arg1[%arg2, %c0_0] [32, 256] [1, 1] : tensor<1024x256xi32> to tensor<32x256xi32>
        %3 = cinm.op.gemm %extracted_slice, %extracted_slice_1 plus %arg3 {cinm.notile} : (tensor<1x32xi32>, tensor<32x256xi32>) -> tensor<1x256xi32>
        affine.yield %3 : tensor<1x256xi32>
      }
      %inserted_slice = tensor.insert_slice %2 into %1[%c0, %c0_0] [1, 256] [1, 1] : tensor<1x256xi32> into tensor<1x256xi32>
      cinm.yield %inserted_slice : tensor<1x256xi32>
    }
    return %0 : tensor<1x256xi32>
  }
  func.func @mm_dimm2_opt(%arg0: tensor<16x64xi32>, %arg1: tensor<64x256xi32>) -> tensor<16x256xi32> {
    %c0 = arith.constant 0 : index
    %0 = cinm.compute attributes {workgroupShape = array<i64: 2, 128>} -> tensor<16x256xi32> {
      %1 = tensor.empty() : tensor<16x256xi32>
      %2 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %1) -> (tensor<16x256xi32>) {
        %c0_i32 = arith.constant 0 : i32
        %splat = tensor.splat %c0_i32 : tensor<1x256xi32>
        %3 = affine.for %arg4 = 0 to 64 step 32 iter_args(%arg5 = %splat) -> (tensor<1x256xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4] [1, 32] [1, 1] : tensor<16x64xi32> to tensor<1x32xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, %c0] [32, 256] [1, 1] : tensor<64x256xi32> to tensor<32x256xi32>
          %4 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {cinm.notile} : (tensor<1x32xi32>, tensor<32x256xi32>) -> tensor<1x256xi32>
          affine.yield %4 : tensor<1x256xi32>
        }
        %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2, %c0] [1, 256] [1, 1] : tensor<1x256xi32> into tensor<16x256xi32>
        affine.yield %inserted_slice : tensor<16x256xi32>
      }
      cinm.yield %2 : tensor<16x256xi32>
    }
    return %0 : tensor<16x256xi32>
  }
  func.func @mm_dimm4_nopt(%arg0: tensor<1x1024xi32>, %arg1: tensor<1024x128xi32>) -> tensor<1x128xi32> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = cinm.compute attributes {workgroupShape = array<i64: 4, 128>} -> tensor<1x128xi32> {
      %1 = tensor.empty() : tensor<1x128xi32>
      %c0_i32 = arith.constant 0 : i32
      %splat = tensor.splat %c0_i32 : tensor<4x128xi32>
      %2 = affine.for %arg2 = 0 to 1024 step 32 iter_args(%arg3 = %splat) -> (tensor<4x128xi32>) {
        %extracted_slice = tensor.extract_slice %arg0[%c0, %arg2] [4, 32] [1, 1] : tensor<1x1024xi32> to tensor<4x32xi32>
        %extracted_slice_1 = tensor.extract_slice %arg1[%arg2, %c0_0] [32, 128] [1, 1] : tensor<1024x128xi32> to tensor<32x128xi32>
        %3 = cinm.op.gemm %extracted_slice, %extracted_slice_1 plus %arg3 {cinm.notile} : (tensor<4x32xi32>, tensor<32x128xi32>) -> tensor<4x128xi32>
        affine.yield %3 : tensor<4x128xi32>
      }
      %inserted_slice = tensor.insert_slice %2 into %1[%c0, %c0_0] [4, 128] [1, 1] : tensor<4x128xi32> into tensor<1x128xi32>
      cinm.yield %inserted_slice : tensor<1x128xi32>
    }
    return %0 : tensor<1x128xi32>
  }
  func.func @mm_dimm4_opt(%arg0: tensor<16x64xi32>, %arg1: tensor<64x128xi32>) -> tensor<16x128xi32> {
    %c0 = arith.constant 0 : index
    %0 = cinm.compute attributes {workgroupShape = array<i64: 4, 128>} -> tensor<16x128xi32> {
      %1 = tensor.empty() : tensor<16x128xi32>
      %2 = affine.for %arg2 = 0 to 16 step 4 iter_args(%arg3 = %1) -> (tensor<16x128xi32>) {
        %c0_i32 = arith.constant 0 : i32
        %splat = tensor.splat %c0_i32 : tensor<4x128xi32>
        %3 = affine.for %arg4 = 0 to 64 step 32 iter_args(%arg5 = %splat) -> (tensor<4x128xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg4] [4, 32] [1, 1] : tensor<16x64xi32> to tensor<4x32xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, %c0] [32, 128] [1, 1] : tensor<64x128xi32> to tensor<32x128xi32>
          %4 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {cinm.notile} : (tensor<4x32xi32>, tensor<32x128xi32>) -> tensor<4x128xi32>
          affine.yield %4 : tensor<4x128xi32>
        }
        %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2, %c0] [4, 128] [1, 1] : tensor<4x128xi32> into tensor<16x128xi32>
        affine.yield %inserted_slice : tensor<16x128xi32>
      }
      cinm.yield %2 : tensor<16x128xi32>
    }
    return %0 : tensor<16x128xi32>
  }
}

