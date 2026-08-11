#upmem = #upmem.platform<type=v1A, dimensions = 32x128x1>

    func.func @_2mm_par_d8(%A: tensor<8x128xi32>, %B: tensor<128x256xi32>,  %C: tensor<128x128xi32>) -> (tensor<8x256xi32>, tensor<8x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<8x128xi32>, tensor<128x256xi32> -> tensor<8x256xi32>
        %r2 = cinm.op.gemm %A, %C : tensor<8x128xi32>, tensor<128x128xi32> -> tensor<8x128xi32>
        func.return %r, %r2 : tensor<8x256xi32>, tensor<8x128xi32>
    }

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>
    func.func @_2mm_par_d16(%A: tensor<16x128xi32>, %B: tensor<128x256xi32>,  %C: tensor<128x128xi32>) -> (tensor<16x256xi32>, tensor<16x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<16x128xi32>, tensor<128x256xi32> -> tensor<16x256xi32>
        %r2 = cinm.op.gemm %A, %C : tensor<16x128xi32>, tensor<128x128xi32> -> tensor<16x128xi32>
        func.return %r, %r2 : tensor<16x256xi32>, tensor<16x128xi32>
    }

// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

    func.func @_2mm_par_d32(%A: tensor<32x128xi32>, %B: tensor<128x256xi32>,  %C: tensor<128x128xi32>) -> (tensor<32x256xi32>, tensor<32x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<32x128xi32>, tensor<128x256xi32> -> tensor<32x256xi32>
        %r2 = cinm.op.gemm %A, %C : tensor<32x128xi32>, tensor<128x128xi32> -> tensor<32x128xi32>
        func.return %r, %r2 : tensor<32x256xi32>, tensor<32x128xi32>
    }
// -----
#upmem = #upmem.platform<type = v1A, dimensions = 32x64x24>

    func.func @_2mm_par_d64(%A: tensor<64x128xi32>, %B: tensor<128x256xi32>,  %C: tensor<128x128xi32>) -> (tensor<64x256xi32>, tensor<64x128xi32>)
    attributes { cinm.available_platforms = [#upmem] } {
        %r = cinm.op.gemm %A, %B : tensor<64x128xi32>, tensor<128x256xi32> -> tensor<64x256xi32>
        %r2 = cinm.op.gemm %A, %C : tensor<64x128xi32>, tensor<128x128xi32> -> tensor<64x128xi32>
        func.return %r, %r2 : tensor<64x256xi32>, tensor<64x128xi32>
    }
