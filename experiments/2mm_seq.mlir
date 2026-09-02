//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_d8(%A: tensor<8x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>,  %C: tensor<256x2048xi32>{cinm.static}) -> tensor<8x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x256xi32> -> tensor<8x256xi32>
    %r2 = cinm.op.gemm %r, %C : tensor<8x256xi32>, tensor<256x2048xi32> -> tensor<8x2048xi32>
    func.return %r2 : tensor<8x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_d16(%A: tensor<16x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>,  %C: tensor<256x2048xi32>{cinm.static}) -> tensor<16x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x256xi32> -> tensor<16x256xi32>
    %r2 = cinm.op.gemm %r, %C : tensor<16x256xi32>, tensor<256x2048xi32> -> tensor<16x2048xi32>
    func.return %r2 : tensor<16x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_d32(%A: tensor<32x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>,  %C: tensor<256x2048xi32>{cinm.static}) -> tensor<32x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %A, %B : tensor<32x1024xi32>, tensor<1024x256xi32> -> tensor<32x256xi32>
    %r2 = cinm.op.gemm %r, %C : tensor<32x256xi32>, tensor<256x2048xi32> -> tensor<32x2048xi32>
    func.return %r2 : tensor<32x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mm_seq_d64(%A: tensor<64x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>,  %C: tensor<256x2048xi32>{cinm.static}) -> tensor<64x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %r = cinm.op.gemm %A, %B : tensor<64x1024xi32>, tensor<1024x256xi32> -> tensor<64x256xi32>
    %r2 = cinm.op.gemm %r, %C : tensor<64x256xi32>, tensor<256x2048xi32> -> tensor<64x2048xi32>
    func.return %r2 : tensor<64x2048xi32>
}
