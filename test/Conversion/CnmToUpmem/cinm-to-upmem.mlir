// RUN: cinm-opt %s --convert-cnm-to-upmem

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, 0)>
module {
  func.func @main() {
    %cst = arith.constant dense<0> : tensor<16x1xi32>
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = cnm.workgroup : !cnm.workgroup<1x16x1>
    %2 = affine.for %arg0 = 0 to 64 step 16 iter_args(%arg1 = %0) -> (tensor<64x64xi32>) {
      %3 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %arg1) -> (tensor<64x64xi32>) {
        %extracted_slice = tensor.extract_slice %0[%arg0, 0] [16, 64] [1, 1] : tensor<64x64xi32> to tensor<16x64xi32>
        %extracted_slice_0 = tensor.extract_slice %0[0, %arg2] [64, 1] [1, 1] : tensor<64x64xi32> to tensor<64x1xi32>
        %4 = tensor.empty() : tensor<1x64xi32>
        %transposed = linalg.transpose ins(%extracted_slice_0 : tensor<64x1xi32>) outs(%4 : tensor<1x64xi32>) permutation = [1, 0] 
        %5 = cnm.alloc() for %1 : !cnm.buffer<64xi32 on 1x16x1, level 0>
        %6 = cnm.alloc() for %1 : !cnm.buffer<64xi32 on 1x16x1, level 0>
        %7 = cnm.alloc() for %1 : !cnm.buffer<i32 on 1x16x1, level 0>
        cnm.scatter %extracted_slice into %5[#map] of %1 : tensor<16x64xi32> into !cnm.buffer<64xi32 on 1x16x1, level 0>
        cnm.scatter %transposed into %6[#map1] of %1 : tensor<1x64xi32> into !cnm.buffer<64xi32 on 1x16x1, level 0>
        cnm.scatter %cst into %7[#map2] of %1 : tensor<16x1xi32> into !cnm.buffer<i32 on 1x16x1, level 0>
        cnm.launch %1 in(%5, %6 : !cnm.buffer<64xi32 on 1x16x1, level 0>, !cnm.buffer<64xi32 on 1x16x1, level 0>) out(%7 : !cnm.buffer<i32 on 1x16x1, level 0>) on !cnm.workgroup<1x16x1> {
        ^bb0(%arg4: memref<64xi32>, %arg5: memref<64xi32>, %arg6: memref<i32>):
          linalg.reduce ins(%arg4, %arg5 : memref<64xi32>, memref<64xi32>) outs(%arg6 : memref<i32>) dimensions = [0] 
            (%in: i32, %in_1: i32, %init: i32) {
              %9 = arith.muli %in, %in_1 : i32
              %10 = arith.addi %9, %init : i32
              linalg.yield %10 : i32
            }
        }
        %8 = cnm.gather %7[#map2] of %1 into %cst : !cnm.buffer<i32 on 1x16x1, level 0> into tensor<16x1xi32>
        %inserted_slice = tensor.insert_slice %8 into %arg3[%arg0, %arg2] [16, 1] [1, 1] : tensor<16x1xi32> into tensor<64x64xi32>
        affine.yield %inserted_slice : tensor<64x64xi32>
      }
      affine.yield %3 : tensor<64x64xi32>
    }
    cnm.free_workgroup %1 : !cnm.workgroup<1x16x1>
    return
  }
}

