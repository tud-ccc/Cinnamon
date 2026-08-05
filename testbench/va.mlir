#upmem = #upmem.platform<type=v1A, dimensions = 32x128x1>
#upmem_32_64_8 = #upmem.array<32x64x8, #upmem>
#upmem_16_64_16 = #upmem.array<16x64x16, #upmem>

module {

    func.func @va_8(%A: tensor<8x2097152xi32>, %B: tensor<8x2097152xi32>) -> tensor<8x2097152xi32> {

        %res = cinm.compute on accelerator #upmem_16_64_16 -> tensor<8x2097152xi32> {
            %r = cinm.op.elementwise add %A, %B : tensor<8x2097152xi32>
            cinm.yield %r: tensor<8x2097152xi32>
        }

        func.return %res : tensor<8x2097152xi32>
    }
    func.func @va_16(%A: tensor<16x1048576xi32>, %B: tensor<16x1048576xi32>) -> tensor<16x1048576xi32> {

        %res = cinm.compute on accelerator #upmem_32_64_8 -> tensor<16x1048576xi32> {
            %r = cinm.op.elementwise add %A, %B {cinm.tile_sizes=array<i64: 65536>}: tensor<16x1048576xi32>
            cinm.yield %r: tensor<16x1048576xi32>
        }

        func.return %res : tensor<16x1048576xi32>
    }
}
