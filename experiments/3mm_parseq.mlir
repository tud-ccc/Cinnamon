//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>

    func.func @_3mm_d8(%A: tensor<8x128xi32> {cinm.static}, %B: tensor<128x256xi32>, %C: tensor<256x128xi32> {cinm.static},  %D: tensor<128x128xi32>) -> (tensor<8x128xi32>, tensor<8x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<8x128xi32>, tensor<128x256xi32> -> tensor<8x256xi32>
        %r2 = cinm.op.gemm %r, %C : tensor<8x256xi32>, tensor<256x128xi32> -> tensor<8x128xi32>
        %r3 = cinm.op.gemm %A, %D : tensor<8x128xi32>, tensor<128x128xi32> -> tensor<8x128xi32>
        func.return %r2, %r3 : tensor<8x128xi32>, tensor<8x128xi32>
    }

// -----
#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>

    func.func @_3mm_d16(%A: tensor<16x128xi32>{cinm.static}, %B: tensor<128x256xi32>, %C: tensor<256x128xi32>{cinm.static},  %D: tensor<128x128xi32>) -> (tensor<16x128xi32>, tensor<16x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<16x128xi32>, tensor<128x256xi32> -> tensor<16x256xi32>
        %r2 = cinm.op.gemm %r, %C : tensor<16x256xi32>, tensor<256x128xi32> -> tensor<16x128xi32>
        %r3 = cinm.op.gemm %A, %D : tensor<16x128xi32>, tensor<128x128xi32> -> tensor<16x128xi32>
        func.return %r2, %r3 : tensor<16x128xi32>, tensor<16x128xi32>
    }

// -----
#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>

    func.func @_3mm_d32(%A: tensor<32x128xi32>{cinm.static}, %B: tensor<128x256xi32>, %C: tensor<256x128xi32>{cinm.static},  %D: tensor<128x128xi32>) -> (tensor<32x128xi32>, tensor<32x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<32x128xi32>, tensor<128x256xi32> -> tensor<32x256xi32>
        %r2 = cinm.op.gemm %r, %C : tensor<32x256xi32>, tensor<256x128xi32> -> tensor<32x128xi32>
        %r3 = cinm.op.gemm %A, %D : tensor<32x128xi32>, tensor<128x128xi32> -> tensor<32x128xi32>
        func.return %r2, %r3 : tensor<32x128xi32>, tensor<32x128xi32>
    }

// -----
#upmem = #upmem.platform<type = v1A, dpus = 4096, tasklets = 1>

    func.func @_3mm_d64(%A: tensor<64x128xi32>{cinm.static}, %B: tensor<128x256xi32>, %C: tensor<256x128xi32>{cinm.static},  %D: tensor<128x128xi32>) -> (tensor<64x128xi32>, tensor<64x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<64x128xi32>, tensor<128x256xi32> -> tensor<64x256xi32>
        %r2 = cinm.op.gemm %r, %C : tensor<64x256xi32>, tensor<256x128xi32> -> tensor<64x128xi32>
        %r3 = cinm.op.gemm %A, %D : tensor<64x128xi32>, tensor<128x128xi32> -> tensor<64x128xi32>
        func.return %r2, %r3 : tensor<64x128xi32>, tensor<64x128xi32>
    }
