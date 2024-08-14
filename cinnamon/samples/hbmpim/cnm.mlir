#map = affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 8+ d2*8 + d3)>


func.func @va(%A: tensor<8192x8192xi32>, %B: tensor<8192x8192xi32>, %C: tensor<8192x8192xi32>) {

    %c0_i32 = arith.constant 0 : i32
    %cst4 = arith.constant 4 : index 
    %cst8192 = arith.constant 8192 : index
    %cst4096 = arith.constant 4096: index
    
    affine.for %arg0 = 0 to %cst4{
        affine.for %arg1 = 0 to %cst4{
            %index1 = arith.muli %arg0, %cst8192 : index
            %index2 = arith.muli %arg1, %cst8192 : index
            %sub_arr1 = tensor.extract_slice %A[%index1, %index2] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
            %sub_arr2 = tensor.extract_slice %B[%index1, %index2] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
            %sub_arr3 = tensor.extract_slice %C[%index1, %index2] [1, 8192] [1, 1] : tensor<8192x8192xi32> to tensor<8192xi32>
            %wg = cnm.workgroup { cnm.physical_dims = ["banks", "channels", "ranks", "grf"] } : !cnm.workgroup<16x64x1x8>
            %shape = arith.constant dense<[1024, 8]> : tensor<2xi64>
            %sub_arr1_reshaped = tensor.reshape %sub_arr1(%shape) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
            %sub_arr2_reshaped = tensor.reshape %sub_arr2(%shape) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
            %sub_arr3_reshaped = tensor.reshape %sub_arr3(%shape) : (tensor<8192xi32>, tensor<2xi64>) -> tensor<1024x8xi32>
            %A_buf = cnm.alloc() for %wg : !cnm.buffer<8xi32 on 16x64x1x8, level 0>
            %B_buf = cnm.alloc() for %wg : !cnm.buffer<8xi32 on 16x64x1x8, level 0>
            %C_buf = cnm.alloc() for %wg : !cnm.buffer<8xi32 on 16x64x1x8, level 0>
            cnm.scatter %sub_arr1_reshaped into %A_buf[#map] of %wg : tensor<1024x8xi32> into !cnm.buffer<8xi32 on 16x64x1x8, level 0>
            cnm.scatter %sub_arr2_reshaped into %B_buf[#map] of %wg : tensor<1024x8xi32> into !cnm.buffer<8xi32 on 16x64x1x8, level 0>
            cnm.scatter %sub_arr3_reshaped into %C_buf[#map] of %wg : tensor<1024x8xi32> into !cnm.buffer<8xi32 on 16x64x1x8, level 0>
            cnm.launch %wg in(%A_buf, %B_buf: !cnm.buffer<8xi32 on 16x64x1x8, level 0>, !cnm.buffer<8xi32 on 16x64x1x8, level 0>) out(%C_buf : !cnm.buffer<8xi32 on 16x64x1x8, level 0>) on !cnm.workgroup<16x64x1x8> {
                ^bb0(%a: memref<8xi32>, %b: memref<8xi32>,  %c: memref<8xi32>):
                linalg.add ins(%a, %b: memref<8xi32>, memref<8xi32>) outs(%c: memref<8xi32>)
            }
            %res = cnm.gather %C_buf[#map] of %wg into %sub_arr3_reshaped: !cnm.buffer<8xi32 on 16x64x1x8, level 0> into tensor<1024x8xi32>
            // cnm.free_workgroup %wg : !cnm.workgroup<16x64x1x8>
        }
    }
    return 
}
