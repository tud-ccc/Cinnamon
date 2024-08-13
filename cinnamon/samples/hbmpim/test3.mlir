// #scatter_map = affine_map<(i) -> (
//     (i ceildiv 8192) mod 16,
//     (i ceildiv 512) mod 64,
//     1,
//     i mod 8
// )>

#map = affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 8+ d2*8 + d3)>


func.func @va(%A: tensor<1024x8xi32>) {

    %c0_i32 = arith.constant 0 : i32
    %cst4 = arith.constant 4 : index 
    %cst8192 = arith.constant 8192 : index
    %cst4096 = arith.constant 4096: index
    %wg = cnm.workgroup { cnm.physical_dims = ["banks", "channels", "ranks", "grf"] } : !cnm.workgroup<16x64x1x8> 
    %A_buf = cnm.alloc() for %wg : !cnm.buffer<8xi32 on 16x64x1x8, level 0>
    cnm.scatter %A into %A_buf[#map] of %wg : tensor<1024x8xi32> into !cnm.buffer<8xi32 on 16x64x1x8, level 0>    
    return 
}
