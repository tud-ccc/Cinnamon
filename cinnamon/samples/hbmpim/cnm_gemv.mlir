#map = affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 8+ d2*8 + d3)>


func.func @gemv(%A: tensor<1x8192x8192xi32>, %B: tensor<1x8192xi32>, %C: tensor<1x8192xi32>) {

    %c0_i32 = arith.constant 0 : i32
    %cst4 = arith.constant 4 : index 
    %cst8192 = arith.constant 8192 : index
    %cst4096 = arith.constant 4096: index
    
    %wg = cnm.workgroup { cnm.physical_dims = ["banks", "channels", "ranks", "grf"] } : !cnm.workgroup<16x64x1x8>
    %shape = arith.constant dense<[1024, 8]> : tensor<2xi64>
    %A_buf = cnm.alloc() for %wg : !cnm.buffer<8192x8192xi32 on 16x64x1x8, level 0>
    %B_buf = cnm.alloc() for %wg : !cnm.buffer<8192xi32 on 16x64x1x8, level 0>
    %C_buf = cnm.alloc() for %wg : !cnm.buffer<8192xi32 on 16x64x1x8, level 0>
    cnm.scatter %A into %A_buf[#map] of %wg : tensor<1x8192x8192xi32> into !cnm.buffer<8192x8192xi32 on 16x64x1x8, level 0>
    cnm.scatter %B into %B_buf[#map] of %wg : tensor<1x8192xi32> into !cnm.buffer<8192xi32 on 16x64x1x8, level 0>
    cnm.scatter %C into %C_buf[#map] of %wg : tensor<1x8192xi32> into !cnm.buffer<8192xi32 on 16x64x1x8, level 0>
    cnm.launch %wg in(%A_buf, %B_buf: !cnm.buffer<8192x8192xi32 on 16x64x1x8, level 0>, !cnm.buffer<8192xi32 on 16x64x1x8, level 0>) out(%C_buf : !cnm.buffer<8192xi32 on 16x64x1x8, level 0>) on !cnm.workgroup<16x64x1x8> {
        ^bb0(%a: memref<8192x8192xi32>, %b: memref<8192xi32>,  %c: memref<8192xi32>):
        linalg.matvec ins(%a, %b: memref<8192x8192xi32>, memref<8192xi32>) outs(%c: memref<8192xi32>)
    }
    %res = cnm.gather %C_buf[#map] of %wg into %C : !cnm.buffer<8192xi32 on 16x64x1x8, level 0> into tensor<1x8192xi32>
    // cnm.free_workgroup %wg : !cnm.workgroup<16x64x1x8>
    return 
}
