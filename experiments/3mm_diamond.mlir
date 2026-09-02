//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// Diamond dependency structure, (A*B)*(C*D): two independent gemms join in a
// third. Only A and C are static.
func.func @_3mm_diam_d8(%A: tensor<8x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>, %C: tensor<256x512xi32>{cinm.static}, %D: tensor<512x2048xi32>) -> tensor<8x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x256xi32> -> tensor<8x256xi32>
    %r = cinm.op.gemm %C, %D : tensor<256x512xi32>, tensor<512x2048xi32> -> tensor<256x2048xi32>
    %j = cinm.op.gemm %l, %r : tensor<8x256xi32>, tensor<256x2048xi32> -> tensor<8x2048xi32>
    func.return %j : tensor<8x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// Diamond dependency structure, (A*B)*(C*D): two independent gemms join in a
// third. Only A and C are static.
func.func @_3mm_diam_d16(%A: tensor<16x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>, %C: tensor<256x512xi32>{cinm.static}, %D: tensor<512x2048xi32>) -> tensor<16x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x256xi32> -> tensor<16x256xi32>
    %r = cinm.op.gemm %C, %D : tensor<256x512xi32>, tensor<512x2048xi32> -> tensor<256x2048xi32>
    %j = cinm.op.gemm %l, %r : tensor<16x256xi32>, tensor<256x2048xi32> -> tensor<16x2048xi32>
    func.return %j : tensor<16x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// Diamond dependency structure, (A*B)*(C*D): two independent gemms join in a
// third. Only A and C are static.
func.func @_3mm_diam_d32(%A: tensor<32x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>, %C: tensor<256x512xi32>{cinm.static}, %D: tensor<512x2048xi32>) -> tensor<32x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<32x1024xi32>, tensor<1024x256xi32> -> tensor<32x256xi32>
    %r = cinm.op.gemm %C, %D : tensor<256x512xi32>, tensor<512x2048xi32> -> tensor<256x2048xi32>
    %j = cinm.op.gemm %l, %r : tensor<32x256xi32>, tensor<256x2048xi32> -> tensor<32x2048xi32>
    func.return %j : tensor<32x2048xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

// Diamond dependency structure, (A*B)*(C*D): two independent gemms join in a
// third. Only A and C are static.
func.func @_3mm_diam_d64(%A: tensor<64x1024xi32>{cinm.static}, %B: tensor<1024x256xi32>, %C: tensor<256x512xi32>{cinm.static}, %D: tensor<512x2048xi32>) -> tensor<64x2048xi32>
attributes { cinm.available_platforms = [#upmem] } {
    %l = cinm.op.gemm %A, %B : tensor<64x1024xi32>, tensor<1024x256xi32> -> tensor<64x256xi32>
    %r = cinm.op.gemm %C, %D : tensor<256x512xi32>, tensor<512x2048xi32> -> tensor<256x2048xi32>
    %j = cinm.op.gemm %l, %r : tensor<64x256xi32>, tensor<256x2048xi32> -> tensor<64x2048xi32>
    func.return %j : tensor<64x2048xi32>
}
