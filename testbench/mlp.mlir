module {
    func.func @mm_dimm1_nopt(%A: tensor<1x1024xi32>, %B: tensor<1024x512xi32>) -> tensor<1x512xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 1, 128, 1> } -> tensor<1x512xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<1x1024xi32>, tensor<1024x512xi32>) -> tensor<1x512xi32>
            cinm.yield %r : tensor<1x512xi32>
        }
        func.return %r0 : tensor<1x512xi32>
    }


    func.func @mm_dimm1_opt(%A: tensor<16x64xi32>, %B: tensor<64x512xi32>) -> tensor<16x512xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 1, 128, 1> } -> tensor<16x512xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<16x64xi32>, tensor<64x512xi32>) -> tensor<16x512xi32>
            cinm.yield %r : tensor<16x512xi32>
        }
        func.return %r0 : tensor<16x512xi32>
     }

    func.func @mm_dimm2_nopt(%A: tensor<1x1024xi32>, %B: tensor<1024x256xi32>) -> tensor<1x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 2, 128, 1> } -> tensor<1x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<1x1024xi32>, tensor<1024x256xi32>) -> tensor<1x256xi32>
            cinm.yield %r : tensor<1x256xi32>
        }
        func.return %r0 : tensor<1x256xi32>
     }

    func.func @mm_dimm2_opt(%A: tensor<16x64xi32>, %B: tensor<64x256xi32>) -> tensor<16x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 2, 128, 1> } -> tensor<16x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<16x64xi32>, tensor<64x256xi32>) -> tensor<16x256xi32>
            cinm.yield %r : tensor<16x256xi32>
        }
        func.return %r0 : tensor<16x256xi32>
     }

    // // too small for the WG
    // func.func @mm_dimm4_nopt(%A: tensor<1x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<1x128xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<1x128xi32> {
    //         %r = cinm.op.gemm %A, %B: (tensor<1x1024xi32>, tensor<1024x128xi32>) -> tensor<1x128xi32>
    //         cinm.yield %r : tensor<1x128xi32>
    //     }
    //     func.return %r0 : tensor<1x128xi32>
    //  }

    func.func @mm_dimm4_opt(%A: tensor<16x64xi32>, %B: tensor<64x128xi32>) -> tensor<16x128xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<16x128xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<16x64xi32>, tensor<64x128xi32>) -> tensor<16x128xi32>
            cinm.yield %r : tensor<16x128xi32>
        }
        func.return %r0 : tensor<16x128xi32>
     }
}
