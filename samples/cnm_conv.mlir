#scatter_map = affine_map<(d0, d1) -> (d0 floordiv 16, d1 floordiv 16, d0 mod 16, d1 mod 16)>
#gather_map = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d2, d1 * 16 + d3)>

#im2col_traits = {
    indexing_maps = [
        affine_map<(N,H,W,KH,KW,C)->(N,H+KH,W+KW,C)>,
        affine_map<(N,H,W,KH,KW,C)->(N,H,W,KH,KW,C)>],
    iterator_types = [
        "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}

func.func @conv(%img : tensor<1x128x128x3xi16>, %flt : tensor<3x3x3x8xi16>, %bias:  tensor<1x126x126x8xi16>) {
    // %init = linalg.init_tensor [1, 126, 126, 3, 3, 3] : tensor<1x126x126x3x3x3xi16>
    %cst0_i = arith.constant 0 : index
    %cst128 = arith.constant 128 : index
    %cst15888_i = arith.constant 15888 : index
    %c0_i16 = arith.constant 0 : i16
    %init = bufferization.alloc_tensor() : tensor<1x126x126x3x3x3xi16>

    %cbuf = linalg.generic #im2col_traits
        ins(%img: tensor<1x128x128x3xi16>)
        outs(%init: tensor<1x126x126x3x3x3xi16>) {
    ^bb0(%arg0: i16, %arg1: i16):
        linalg.yield %arg0 : i16
    } -> tensor<1x126x126x3x3x3xi16>

    %im2col_img = tensor.collapse_shape %cbuf [[0,1,2],[3,4,5]]
        : tensor<1x126x126x3x3x3xi16> into tensor<15876x27xi16>

    %im2col_flt = tensor.collapse_shape %flt [[0,1,2],[3]]
        : tensor<3x3x3x8xi16> into tensor<27x8xi16>

    %A_pad= tensor.pad %im2col_img low[0, 0] high[12, 5] {
        ^bb0(%arg0 : index, %arg1 : index):
            tensor.yield %c0_i16 : i16
    } : tensor<15876x27xi16> to tensor<15888x32xi16>

    %B_pad= tensor.pad %im2col_flt low[0, 0] high[5, 8] {
        ^bb0(%arg0 : index, %arg1 : index):
            tensor.yield %c0_i16 : i16
    } : tensor<27x8xi16> to tensor<32x16xi16>

    %in = bufferization.alloc_tensor() : tensor<15888x16xi16>

    %C_pad = scf.for %o0 = %cst0_i to %cst15888_i step %cst128 iter_args(%in_result = %in) -> tensor<15888x16xi16> {
        %A_tile = tensor.extract_slice %A_pad[%o0, %cst0_i][128, 32][1, 1]
            : tensor<15888x32xi16> to tensor<128x32xi16>
        %wg = cnm.workgroup : !cnm.workgroup<8x2>
        %A_buf = cnm.alloc() for %wg { cnm.physical_space = "global" }
        : !cnm.buffer<16x16xi16 on 8x2, level 0>
        %B_buf = cnm.alloc() for %wg { cnm.physical_space = "global" }
        : !cnm.buffer<16x16xi16 on 8x2, level 0>
        %C_buf_init = cnm.alloc() for %wg { cnm.physical_space = "global" }
        : !cnm.buffer<16x16xi16 on 8x2, level 0>
        %C_buf = cnm.set_zero %C_buf_init :  !cnm.buffer<16x16xi16 on 8x2, level 0>
        %sc_a_token = cnm.scatter %A_tile into %A_buf[#scatter_map] of %wg
            : tensor<128x32xi16> into !cnm.buffer<16x16xi16 on 8x2, level 0>
        %sc_b_token = cnm.scatter %B_pad into %B_buf[#scatter_map] of %wg
            : tensor<32x16xi16> into !cnm.buffer<16x16xi16 on 8x2, level 0>
        %e_token = cnm.launch %wg in(%A_buf, %B_buf : !cnm.buffer<16x16xi16 on 8x2, level 0>, !cnm.buffer<16x16xi16 on 8x2, level 0>) out(%C_buf : !cnm.buffer<16x16xi16 on 8x2, level 0>) on !cnm.workgroup<8x2> {
            ^bb0(%A_space: memref<16x16xi16>, %B_space: memref<16x16xi16>, %C_space: memref<16x16xi16>):
                affine.for %arg3 = 0 to 16 {
                    affine.for %arg4 = 0 to 16 {
                        affine.for %arg5 = 0 to 16 {
                            %0 = affine.load %A_space[%arg3, %arg5] : memref<16x16xi16>
                            %1 = affine.load %B_space[%arg5, %arg4] : memref<16x16xi16>
                            %2 = affine.load %C_space[%arg3, %arg4] : memref<16x16xi16>
                            %3 = arith.muli %0, %1 : i16
                            %4 = arith.addi %2, %3 : i16
                            affine.store %4, %C_space[%arg3, %arg4] : memref<16x16xi16>
                        }
                    }
                } // This inner most block can be anything. Here is just as an example of how you can do the matmul using affine. it can be cinm op as well.
        }

        %C_tile, %g_token = cnm.gather %C_buf[#gather_map] of %wg
            : !cnm.buffer<16x16xi16 on 8x2, level 0> into tensor<128x16xi16>
        %out_result = tensor.insert_slice %C_tile into %in_result[%o0, 0][128, 16][1, 1]
        : tensor<128x16xi16> into tensor<15888x16xi16>
        scf.yield %out_result: tensor<15888x16xi16>
    }

    return
}
