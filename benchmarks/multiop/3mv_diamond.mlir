//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// j = (W1 * x) * (W2 * x), elementwise: the diamond -- two independent
// gemvs joining in a third op. This is llama's SwiGLU gate, where the gate
// and up projections run on the same activation and meet in an elementwise
// product, which is what the diamond shape looks like once the activation
// is a vector: a matrix-vector op cannot take a vector as its matrix, so
// the join that closes two branches is elementwise rather than another
// contraction (3mm_diamond.mlir closes its branches with a gemm because
// there the branches are still matrices).
//
// Size class = bytes of each M x K i32 weight (see 2mv_seq.mlir). W1 and W2
// are static and have the same shape, so both branches are one program
// class; the join is a cheap elementwise op over a vector, and whether it
// is worth offloading at all is a question the transfer-bound gate answers.
// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_diam_1MB(%x: tensor<512xi32>, %W1: tensor<512x512xi32>{cinm.static}, %W2: tensor<512x512xi32>{cinm.static}) -> tensor<512xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemv %W1, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    %r = cinm.op.gemv %W2, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    %j = cinm.op.elementwise mul %l, %r : tensor<512xi32>
    func.return %j : tensor<512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_diam_64MB(%x: tensor<4096xi32>, %W1: tensor<4096x4096xi32>{cinm.static}, %W2: tensor<4096x4096xi32>{cinm.static}) -> tensor<4096xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemv %W1, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %r = cinm.op.gemv %W2, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %j = cinm.op.elementwise mul %l, %r : tensor<4096xi32>
    func.return %j : tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_diam_256MB(%x: tensor<8192xi32>, %W1: tensor<8192x8192xi32>{cinm.static}, %W2: tensor<8192x8192xi32>{cinm.static}) -> tensor<8192xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemv %W1, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    %r = cinm.op.gemv %W2, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    %j = cinm.op.elementwise mul %l, %r : tensor<8192xi32>
    func.return %j : tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_diam_512MB(%x: tensor<16384xi32>, %W1: tensor<8192x16384xi32>{cinm.static}, %W2: tensor<8192x16384xi32>{cinm.static}) -> tensor<8192xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemv %W1, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    %r = cinm.op.gemv %W2, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    %j = cinm.op.elementwise mul %l, %r : tensor<8192xi32>
    func.return %j : tensor<8192xi32>
}
