//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// J = (A * B) * (C * D): the diamond -- two independent gemms joining in a
// third. Size class = bytes of each i32 weight (see 2mm_seq.mlir): B is
// K x M and C is M x K, so both hold K*M elements and the join contracts
// over M. The statics are those inner factors, one per branch: the outer
// factors are the skinny activations, which is what keeps both
// intermediates small (8 x M and M x 8) instead of materialising a big one.
// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mm_diam_1MB(%A: tensor<8x512xi32>, %B: tensor<512x512xi32>{cinm.static}, %C: tensor<512x512xi32>{cinm.static}, %D: tensor<512x8xi32>) -> tensor<8x8xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<8x512xi32>, tensor<512x512xi32> -> tensor<8x512xi32>
    %r = cinm.op.gemm %C, %D : tensor<512x512xi32>, tensor<512x8xi32> -> tensor<512x8xi32>
    %j = cinm.op.gemm %l, %r : tensor<8x512xi32>, tensor<512x8xi32> -> tensor<8x8xi32>
    func.return %j : tensor<8x8xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mm_diam_64MB(%A: tensor<8x4096xi32>, %B: tensor<4096x4096xi32>{cinm.static}, %C: tensor<4096x4096xi32>{cinm.static}, %D: tensor<4096x8xi32>) -> tensor<8x8xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<8x4096xi32>, tensor<4096x4096xi32> -> tensor<8x4096xi32>
    %r = cinm.op.gemm %C, %D : tensor<4096x4096xi32>, tensor<4096x8xi32> -> tensor<4096x8xi32>
    %j = cinm.op.gemm %l, %r : tensor<8x4096xi32>, tensor<4096x8xi32> -> tensor<8x8xi32>
    func.return %j : tensor<8x8xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mm_diam_256MB(%A: tensor<8x8192xi32>, %B: tensor<8192x8192xi32>{cinm.static}, %C: tensor<8192x8192xi32>{cinm.static}, %D: tensor<8192x8xi32>) -> tensor<8x8xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<8x8192xi32>, tensor<8192x8192xi32> -> tensor<8x8192xi32>
    %r = cinm.op.gemm %C, %D : tensor<8192x8192xi32>, tensor<8192x8xi32> -> tensor<8192x8xi32>
    %j = cinm.op.gemm %l, %r : tensor<8x8192xi32>, tensor<8192x8xi32> -> tensor<8x8xi32>
    func.return %j : tensor<8x8xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mm_diam_512MB(%A: tensor<8x8192xi32>, %B: tensor<8192x16384xi32>{cinm.static}, %C: tensor<16384x8192xi32>{cinm.static}, %D: tensor<8192x8xi32>) -> tensor<8x8xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<8x8192xi32>, tensor<8192x16384xi32> -> tensor<8x16384xi32>
    %r = cinm.op.gemm %C, %D : tensor<16384x8192xi32>, tensor<8192x8xi32> -> tensor<16384x8xi32>
    %j = cinm.op.gemm %l, %r : tensor<8x16384xi32>, tensor<16384x8xi32> -> tensor<8x8xi32>
    func.return %j : tensor<8x8xi32>
}
