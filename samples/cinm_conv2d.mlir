#im2col_traits = {
    indexing_maps = [
        affine_map<(N,H,W,KH,KW,C)->(N,H+KH,W+KW,C)>,
        affine_map<(N,H,W,KH,KW,C)->(N,H,W,KH,KW,C)>],
    iterator_types = [
        "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}

func.func @conv(%img : tensor<1x128x128x3xi16>, %flt : tensor<3x3x3x8xi16>, %bias:  tensor<1x126x126x8xi16>) {
    // %init = linalg.init_tensor [1, 126, 126, 3, 3, 3] : tensor<1x126x126x3x3x3xi16>
    %init = bufferization.alloc_tensor() : tensor<1x126x126x3x3x3xi16>

    %cbuf = linalg.generic #im2col_traits
        ins(%img: tensor<1x128x128x3xi16>)
        outs(%init: tensor<1x126x126x3x3x3xi16>) {
    ^bb0(%arg0: i16, %arg1: i16):
        linalg.yield %arg0 : i16
    } -> tensor<1x126x126x3x3x3xi16>
    %im2col = tensor.collapse_shape %cbuf [[0,1,2],[3,4,5]]
        : tensor<1x126x126x3x3x3xi16> into tensor<15876x27xi16>

    %rbuf = cinm.compute -> tensor<15876x8xi16> {
        %flt = arith.constant dense<"..."> : tensor<27x8xi16>
        %conv = cinm.op.gemm %im2col, %flt : (tensor<15876x27xi16>, tensor<27x8xi16>) -> tensor<15876x8xi16>
        cinm.yield %conv : tensor<15876x8xi16>
    }

    %conv = tensor.expand_shape %rbuf [[0,1,2],[3,4,5]]
        : tensor<15876x8xi16> into tensor<1x126x126x8xi16>
    return
}
