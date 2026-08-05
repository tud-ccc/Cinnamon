#upmem = #upmem.platform<type=v1A, dimensions = 32x128x1>
#upmem_4_128_1 = #upmem.array<4x128x1, #upmem>
#upmem_8_128_1 = #upmem.array<8x128x1, #upmem>
#upmem_16_64_1 = #upmem.array<16x64x1, #upmem>

module {

    func.func @mm_dimm4_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x256xi32>) -> tensor<8x256xi32> {

        %r0 = cinm.compute on accelerator #upmem_4_128_1 -> tensor<8x256xi32> {
            %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x256xi32> -> tensor<8x256xi32>
            cinm.yield %r : tensor<8x256xi32>
        }
        func.return %r0 : tensor<8x256xi32>
    }

    func.func @mm_dimm4_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<16x128xi32> {

        %r0 = cinm.compute on accelerator #upmem_4_128_1 -> tensor<16x128xi32> {
            %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x128xi32> -> tensor<16x128xi32>
            cinm.yield %r : tensor<16x128xi32>
        }
        func.return %r0 : tensor<16x128xi32>
    }

    func.func @mm_dimm8_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {

        %r0 = cinm.compute on accelerator #upmem_8_128_1 -> tensor<8x128xi32> {
            %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }

    func.func @mm_dimm8_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x64xi32>) -> tensor<16x64xi32> {

        %r0 = cinm.compute on accelerator #upmem_8_128_1 -> tensor<16x64xi32> {
            %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x64xi32> -> tensor<16x64xi32>
            cinm.yield %r : tensor<16x64xi32>
        }
        func.return %r0 : tensor<16x64xi32>
    }

    func.func @mm_dimm16_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x512xi32>) -> tensor<8x512xi32> {

        %r0 = cinm.compute on accelerator #upmem_16_64_1 -> tensor<8x512xi32> {
            %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x512xi32> -> tensor<8x512xi32>
            cinm.yield %r : tensor<8x512xi32>
        }
        func.return %r0 : tensor<8x512xi32>
    }

    func.func @mm_dimm16_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x512xi32>) -> tensor<16x512xi32> {

        %r0 = cinm.compute on accelerator #upmem_16_64_1 -> tensor<16x512xi32> {
            %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x512xi32> -> tensor<16x512xi32>
            cinm.yield %r : tensor<16x512xi32>
        }
        func.return %r0 : tensor<16x512xi32>
    }

}
