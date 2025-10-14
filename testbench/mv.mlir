module {

    // func.func @mv_dimm4_nopt(%A: tensor<128x65536xi32>, %B: tensor<65536xi32>) -> tensor<128xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<128xi32> {
    //         %r = cinm.op.gemv %A, %B: (tensor<128x65536xi32>, tensor<65536xi32>) -> tensor<128xi32>
    //         cinm.yield %r : tensor<128xi32>
    //     }
    //     return %r0 : tensor<128xi32>
    // }

    func.func @mv_dimm4_opt(%A: tensor<4096x2048xi32>, %B: tensor<2048xi32>) -> tensor<4096xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<4096xi32> {
            %r = cinm.op.gemv %A, %B: (tensor<4096x2048xi32>, tensor<2048xi32>) -> tensor<4096xi32>
            cinm.yield %r : tensor<4096xi32>
        }
        return %r0 : tensor<4096xi32>
    }

    func.func @mv_dimm8_nopt(%A: tensor<16384x512xi32>, %B: tensor<512xi32>) -> tensor<16384xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<16384xi32> {
            %r = cinm.op.gemv %A, %B: (tensor<16384x512xi32>, tensor<512xi32>) -> tensor<16384xi32>
            cinm.yield %r : tensor<16384xi32>
        }
        return %r0 : tensor<16384xi32>
    }

    // func.func @mv_dimm8_opt(%A: tensor<256x16384xi32>, %B: tensor<16384xi32>) -> tensor<256xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<256xi32> {
    //         %r = cinm.op.gemv %A, %B: (tensor<256x16384xi32>, tensor<16384xi32>) -> tensor<256xi32>
    //         cinm.yield %r : tensor<256xi32>
    //     }
    //     return %r0 : tensor<256xi32>
    // }

    // func.func @mv_dimm16_nopt(%A: tensor<512x4096xi32>, %B: tensor<4096xi32>) -> tensor<512xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 16, 128, 1> } -> tensor<512xi32> {
    //         %r = cinm.op.gemv %A, %B: (tensor<512x4096xi32>, tensor<4096xi32>) -> tensor<512xi32>
    //         cinm.yield %r : tensor<512xi32>
    //     }
    //     return %r0 : tensor<512xi32>
    // }

    func.func @mv_dimm16_opt(%A: tensor<16384x128xi32>, %B: tensor<128xi32>) -> tensor<16384xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 16, 128, 1> } -> tensor<16384xi32> {
            %r = cinm.op.gemv %A, %B: (tensor<16384x128xi32>, tensor<128xi32>) -> tensor<16384xi32>
            cinm.yield %r : tensor<16384xi32>
        }
        return %r0 : tensor<16384xi32>
    }

}