#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d1 mod 128)>
#map2 = affine_map<(d0, d1) -> (d1 floordiv 128, d1 mod 128)>
module {
  memref.global "private" constant @__constant_1x128xi32 : memref<1x128xi32> = dense<0> {alignment = 64 : i64}
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
          %10 = cnm.launch %3 ins(%4, %6 : !cnm.buffer<16xi32 on 4x8x2, level 0>, !cnm.buffer<16xi32 on 4x8x2, level 0>) outs(%8 : !cnm.buffer<i32 on 4x8x2, level 0>) on !cnm.workgroup<4x8x2> {
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
          }
          %5 = cnm.alloc() for %4 : !cnm.buffer<32xi32 on 1x128, level 0>
          %6 = cnm.alloc() for %4 : !cnm.buffer<32xi32 on 1x128, level 0>
          %7 = cnm.alloc() for %4 : !cnm.buffer<i32 on 1x128, level 0>
          cnm.scatter %subview_4 into %5[#map] of %4 : memref<1x32xi32, strided<[1024, 1], offset: ?>> into !cnm.buffer<32xi32 on 1x128, level 0>
          cnm.scatter %alloc_3 into %6[#map1] of %4 : memref<128x32xi32> into !cnm.buffer<32xi32 on 1x128, level 0>
          cnm.scatter %arg5 into %7[#map2] of %4 : memref<1x128xi32> into !cnm.buffer<i32 on 1x128, level 0>
          cnm.launch %4 in(%5, %6 : !cnm.buffer<32xi32 on 1x128, level 0>, !cnm.buffer<32xi32 on 1x128, level 0>) out(%7 : !cnm.buffer<i32 on 1x128, level 0>) on !cnm.workgroup<1x128> {
          ^bb0(%arg6: memref<32xi32>, %arg7: memref<32xi32>, %arg8: memref<i32>):
            %c0_6 = arith.constant 0 : index
            %c32_7 = arith.constant 32 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg9 = %c0_6 to %c32_7 step %c1_8 {
              %8 = memref.load %arg6[%arg9] : memref<32xi32>
              %9 = memref.load %arg7[%arg9] : memref<32xi32>
              %10 = memref.load %arg8[] : memref<i32>
              %11 = arith.muli %8, %9 : i32
              %12 = arith.addi %11, %10 : i32
              memref.store %12, %arg8[] : memref<i32>
            }
          }
          cnm.gather %7[#map2] of %4 into %arg5 : !cnm.buffer<i32 on 1x128, level 0> into memref<1x128xi32>
          scf.yield %arg5 : memref<1x128xi32>
        }
        %subview = memref.subview %arg3[%arg0, %arg2] [1, 128] [1, 1] : memref<1024x1024xi32> to memref<1x128xi32, strided<[1024, 1], offset: ?>>
        memref.copy %3, %subview : memref<1x128xi32> to memref<1x128xi32, strided<[1024, 1], offset: ?>>
        scf.yield %arg3 : memref<1024x1024xi32>
      }
      scf.yield %1 : memref<1024x1024xi32>
    }
    return
  }
}
