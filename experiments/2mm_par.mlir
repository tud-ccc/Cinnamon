//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// (r1, r2) = (X * W1, X * W2): two independent gemms sharing the activation
// X -- the QKV pattern. Size class = bytes of each NxN i32 weight (see
// 2mm_seq.mlir); both weights are static.

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_par_1MB(%X: tensor<8x512xi32>, %W1: tensor<512x512xi32>{cinm.static}, %W2: tensor<512x512xi32>{cinm.static}) -> (tensor<8x512xi32>, tensor<8x512xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %r1 = cinm.op.gemm %X, %W1 : tensor<8x512xi32>, tensor<512x512xi32> -> tensor<8x512xi32>
    %r2 = cinm.op.gemm %X, %W2 : tensor<8x512xi32>, tensor<512x512xi32> -> tensor<8x512xi32>
    func.return %r1, %r2 : tensor<8x512xi32>, tensor<8x512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_par_16MB(%X: tensor<8x2048xi32>, %W1: tensor<2048x2048xi32>{cinm.static}, %W2: tensor<2048x2048xi32>{cinm.static}) -> (tensor<8x2048xi32>, tensor<8x2048xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %r1 = cinm.op.gemm %X, %W1 : tensor<8x2048xi32>, tensor<2048x2048xi32> -> tensor<8x2048xi32>
    %r2 = cinm.op.gemm %X, %W2 : tensor<8x2048xi32>, tensor<2048x2048xi32> -> tensor<8x2048xi32>
    func.return %r1, %r2 : tensor<8x2048xi32>, tensor<8x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_par_64MB(%X: tensor<8x4096xi32>, %W1: tensor<4096x4096xi32>{cinm.static}, %W2: tensor<4096x4096xi32>{cinm.static}) -> (tensor<8x4096xi32>, tensor<8x4096xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %r1 = cinm.op.gemm %X, %W1 : tensor<8x4096xi32>, tensor<4096x4096xi32> -> tensor<8x4096xi32>
    %r2 = cinm.op.gemm %X, %W2 : tensor<8x4096xi32>, tensor<4096x4096xi32> -> tensor<8x4096xi32>
    func.return %r1, %r2 : tensor<8x4096xi32>, tensor<8x4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_par_256MB(%X: tensor<8x8192xi32>, %W1: tensor<8192x8192xi32>{cinm.static}, %W2: tensor<8192x8192xi32>{cinm.static}) -> (tensor<8x8192xi32>, tensor<8x8192xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %r1 = cinm.op.gemm %X, %W1 : tensor<8x8192xi32>, tensor<8192x8192xi32> -> tensor<8x8192xi32>
    %r2 = cinm.op.gemm %X, %W2 : tensor<8x8192xi32>, tensor<8192x8192xi32> -> tensor<8x8192xi32>
    func.return %r1, %r2 : tensor<8x8192xi32>, tensor<8x8192xi32>
}
