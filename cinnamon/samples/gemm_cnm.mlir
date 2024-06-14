#map = affine_map<(d0, d1) -> (d0 floordiv 16, (d0 mod 16) floordiv 2, d0 mod 2, d1)>
#map1 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 2, d0 mod 2)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 16 + d1 * 2 + d2)>
module {
  func.func @main() {
    %0 = tensor.empty() : tensor<1024x1024xi32>
    %1 = affine.for %arg0 = 0 to 1024 step 64 iter_args(%arg1 = %0) -> (tensor<1024x1024xi32>) {
      %2 = affine.for %arg2 = 0 to 1024 step 64 iter_args(%arg3 = %arg1) -> (tensor<1024x1024xi32>) {
        %extracted_slice = tensor.extract_slice %0[%arg0, 0] [64, 1024] [1, 1] : tensor<1024x1024xi32> to tensor<64x1024xi32>
        %extracted_slice_0 = tensor.extract_slice %0[0, %arg2] [1024, 64] [1, 1] : tensor<1024x1024xi32> to tensor<1024x64xi32>
        %generated = tensor.generate  {
        ^bb0(%arg4: index, %arg5: index):
          %extracted_slice_1 = tensor.extract_slice %extracted_slice[%arg4, 0] [1, 1024] [1, 1] : tensor<64x1024xi32> to tensor<1024xi32>
          %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[0, %arg5] [1024, 1] [1, 1] : tensor<1024x64xi32> to tensor<1024xi32>
          %cst = arith.constant dense<0> : tensor<i32>
          %cst_3 = arith.constant dense<0> : tensor<64xi32>
          %cst_4 = arith.constant dense<[64, 16]> : tensor<2xi64>
          %reshape = tensor.reshape %extracted_slice_1(%cst_4) : (tensor<1024xi32>, tensor<2xi64>) -> tensor<64x16xi32>
          %reshape_5 = tensor.reshape %extracted_slice_2(%cst_4) : (tensor<1024xi32>, tensor<2xi64>) -> tensor<64x16xi32>
          %3 = cnm.workgroup : !cnm.workgroup<4x8x2>
          %4 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 4x8x2, level 0>
          %5 = cnm.scatter %reshape into %4[#map] of %3 : tensor<64x16xi32> into !cnm.buffer<16xi32 on 4x8x2, level 0>
          %6 = cnm.alloc() for %3 : !cnm.buffer<16xi32 on 4x8x2, level 0>
          %7 = cnm.scatter %reshape_5 into %6[#map] of %3 : tensor<64x16xi32> into !cnm.buffer<16xi32 on 4x8x2, level 0>
          %8 = cnm.alloc() for %3 : !cnm.buffer<i32 on 4x8x2, level 0>
          %9 = cnm.scatter %cst_3 into %8[#map1] of %3 : tensor<64xi32> into !cnm.buffer<i32 on 4x8x2, level 0>
          %10 = cnm.launch %3 in(%4, %6 : !cnm.buffer<16xi32 on 4x8x2, level 0>, !cnm.buffer<16xi32 on 4x8x2, level 0>) out(%8 : !cnm.buffer<i32 on 4x8x2, level 0>) on !cnm.workgroup<4x8x2> {
          ^bb0(%arg6: memref<16xi32>, %arg7: memref<16xi32>, %arg8: memref<i32>):
            linalg.reduce ins(%arg6, %arg7 : memref<16xi32>, memref<16xi32>) outs(%arg8 : memref<i32>) dimensions = [0] 
              (%in: i32, %in_6: i32, %init: i32) {
                %11 = arith.muli %in, %in_6 : i32
                %12 = arith.addi %11, %init : i32
                linalg.yield %12 : i32
              }
          }
          %output, %token = cnm.gather %8[#map2] of %3 : !cnm.buffer<i32 on 4x8x2, level 0> into tensor<64xi32>
          %reduced = linalg.reduce ins(%output : tensor<64xi32>) outs(%cst : tensor<i32>) dimensions = [0] 
            (%in: i32, %init: i32) {
              %11 = arith.addi %in, %init : i32
              linalg.yield %11 : i32
            }
          %extracted = tensor.extract %reduced[] : tensor<i32>
          tensor.yield %extracted : i32
        } : tensor<64x64xi32>
        %inserted_slice = tensor.insert_slice %generated into %arg3[%arg0, %arg2] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<1024x1024xi32>
        affine.yield %inserted_slice : tensor<1024x1024xi32>
      }
      affine.yield %2 : tensor<1024x1024xi32>
    }
    return
  }
}

