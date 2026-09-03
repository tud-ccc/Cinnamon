//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// r2 = (X * W1) * W2: a two-gemm chain. The size class names the bytes of
// each NxN i32 weight (1MB -> N=512, 16MB -> 2048, 64MB -> 4096,
// 256MB -> 8192), the prim files' convention; the activation X stays skinny
// (8 rows). W1 and W2 are static.

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_1MB(%X: tensor<8x512xi32>, %W1: tensor<512x512xi32>{cinm.static}, %W2: tensor<512x512xi32>{cinm.static}) -> tensor<8x512xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %X, %W1 : tensor<8x512xi32>, tensor<512x512xi32> -> tensor<8x512xi32>
    %r2 = cinm.op.gemm %r, %W2 : tensor<8x512xi32>, tensor<512x512xi32> -> tensor<8x512xi32>
    func.return %r2 : tensor<8x512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_16MB(%X: tensor<8x2048xi32>, %W1: tensor<2048x2048xi32>{cinm.static}, %W2: tensor<2048x2048xi32>{cinm.static}) -> tensor<8x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %X, %W1 : tensor<8x2048xi32>, tensor<2048x2048xi32> -> tensor<8x2048xi32>
    %r2 = cinm.op.gemm %r, %W2 : tensor<8x2048xi32>, tensor<2048x2048xi32> -> tensor<8x2048xi32>
    func.return %r2 : tensor<8x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_64MB(%X: tensor<8x4096xi32>, %W1: tensor<4096x4096xi32>{cinm.static}, %W2: tensor<4096x4096xi32>{cinm.static}) -> tensor<8x4096xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %X, %W1 : tensor<8x4096xi32>, tensor<4096x4096xi32> -> tensor<8x4096xi32>
    %r2 = cinm.op.gemm %r, %W2 : tensor<8x4096xi32>, tensor<4096x4096xi32> -> tensor<8x4096xi32>
    func.return %r2 : tensor<8x4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_256MB(%X: tensor<8x8192xi32>, %W1: tensor<8192x8192xi32>{cinm.static}, %W2: tensor<8192x8192xi32>{cinm.static}) -> tensor<8x8192xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %X, %W1 : tensor<8x8192xi32>, tensor<8192x8192xi32> -> tensor<8x8192xi32>
    %r2 = cinm.op.gemm %r, %W2 : tensor<8x8192xi32>, tensor<8192x8192xi32> -> tensor<8x8192xi32>
    func.return %r2 : tensor<8x8192xi32>
}
