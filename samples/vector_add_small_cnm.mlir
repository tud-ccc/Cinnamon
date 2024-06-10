#scatter_map = affine_map<(i) -> (
    (i floordiv 32768) mod 2,
    (i floordiv 1024) mod 32,
    (i floordiv 64) mod 16,
    i mod 64
)>

#gather_map = affine_map<(wg0, wg1, wg2, b0) -> (
    wg0 * 32768 +
    wg1 * 1024 +
    wg2 * 64 +
    b0
)>

func.func @add(%a: tensor<65536xi32>, %b: tensor<65536xi32>) -> tensor<65536xi32> {
    %wg = cnm.workgroup : !cnm.workgroup<2x32x16>
    %a_buf = cnm.alloc () for %wg : !cnm.buffer<64xi32 on 2x32x16, level 0>
    %b_buf = cnm.alloc () for %wg : !cnm.buffer<64xi32 on 2x32x16, level 0>
    %c_buf = cnm.alloc () for %wg : !cnm.buffer<64xi32 on 2x32x16, level 0>
    %a_token = cnm.scatter %a into %a_buf[#scatter_map] of %wg : tensor<65536xi32> into !cnm.buffer<64xi32 on 2x32x16, level 0>
    %b_token = cnm.scatter %b into %b_buf[#scatter_map] of %wg : tensor<65536xi32> into !cnm.buffer<64xi32 on 2x32x16, level 0>
    %10 = cnm.launch %wg in (%a_buf, %b_buf : !cnm.buffer<64xi32 on 2x32x16, level 0>, !cnm.buffer<64xi32 on 2x32x16, level 0>) out (%c_buf : !cnm.buffer<64xi32 on 2x32x16, level 0>) on !cnm.workgroup<2x32x16> {
    ^bb0(%a_slice: memref<64xi32>, %b_slice: memref<64xi32>, %c_slice: memref<64xi32>):
        affine.for %i = 0 to 64 step 1 {
            %a_elem = affine.load %a_slice[%i] : memref<64xi32>
            %b_elem = affine.load %b_slice[%i] : memref<64xi32>
            %c_elem = arith.addi %a_elem, %b_elem : i32
            affine.store %c_elem, %c_slice[%i] : memref<64xi32>
        }
    }
    %c, %c_token = cnm.gather %c_buf[#gather_map] of %wg : !cnm.buffer<64xi32 on 2x32x16, level 0> into tensor<65536xi32>
    return %c : tensor<65536xi32>
}
