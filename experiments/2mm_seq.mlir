//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// r2 = (X * W1) * W2: a two-gemm chain. The size class names the bytes of
// each i32 weight, the prim files' convention: a weight is K x M and the
// classes are 1MB (512x512), 64MB (4096x4096), 256MB (8192x8192) and 512MB
// (8192x16384, the shape the prim suite's 512MB gemv uses -- no square i32
// matrix is 512MB). The second weight is M x K so both hold K*M elements
// and the chain closes back on K. The activation X stays skinny (8 rows).
// W1 and W2 are static.
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

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_512MB(%X: tensor<8x8192xi32>, %W1: tensor<8192x16384xi32>{cinm.static}, %W2: tensor<16384x8192xi32>{cinm.static}) -> tensor<8x8192xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %X, %W1 : tensor<8x8192xi32>, tensor<8192x16384xi32> -> tensor<8x16384xi32>
    %r2 = cinm.op.gemm %r, %W2 : tensor<8x16384xi32>, tensor<16384x8192xi32> -> tensor<8x8192xi32>
    func.return %r2 : tensor<8x8192xi32>
}
