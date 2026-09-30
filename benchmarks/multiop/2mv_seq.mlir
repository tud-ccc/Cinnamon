//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// z = W2 * (W1 * x): a two-gemv chain, llama's FFN shape. The mv suite is
// the mm suite at one activation row, written as the matrix-vector op the
// decode path actually issues: y = W x, with the weight stored transposed
// relative to the gemm form (x W = (W^T x)^T), which is how a projection
// weight is laid out anyway.
//
// The size class names the bytes of each i32 weight, as in 2mm_seq.mlir: a
// weight is M x K and the classes are 1MB (512x512), 64MB (4096x4096),
// 256MB (8192x8192) and 512MB (8192x16384, the shape prim_mtv's 512MB point
// uses). The second weight is K x M so both hold M*K elements and the chain
// closes back on K. W1 and W2 are static.
// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_seq_1MB(%x: tensor<512xi32>, %W1: tensor<512x512xi32>{cinm.static}, %W2: tensor<512x512xi32>{cinm.static}) -> tensor<512xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    %z = cinm.op.gemv %W2, %y : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    func.return %z : tensor<512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_seq_64MB(%x: tensor<4096xi32>, %W1: tensor<4096x4096xi32>{cinm.static}, %W2: tensor<4096x4096xi32>{cinm.static}) -> tensor<4096xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %z = cinm.op.gemv %W2, %y : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    func.return %z : tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_seq_256MB(%x: tensor<8192xi32>, %W1: tensor<8192x8192xi32>{cinm.static}, %W2: tensor<8192x8192xi32>{cinm.static}) -> tensor<8192xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    %z = cinm.op.gemv %W2, %y : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    func.return %z : tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_seq_512MB(%x: tensor<16384xi32>, %W1: tensor<8192x16384xi32>{cinm.static}, %W2: tensor<16384x8192xi32>{cinm.static}) -> tensor<16384xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    %z = cinm.op.gemv %W2, %y : tensor<16384x8192xi32>, tensor<8192xi32> -> tensor<16384xi32>
    func.return %z : tensor<16384xi32>
}
