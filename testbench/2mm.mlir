module {

    func.func @mm_dimm4_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x256xi32>) -> (tensor<8x256xi32>, tensor<8x256xi32>) {

        %r, %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<8x256xi32>, tensor<8x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x256xi32>) -> tensor<8x256xi32>
            %r2 = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x256xi32>) -> tensor<8x256xi32>
            cinm.yield %r, %r2 : tensor<8x256xi32>, tensor<8x256xi32>
        }
        func.return %r, %r0 : tensor<8x256xi32>, tensor<8x256xi32>
    }
	
    func.func @mm_dimm4_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<16x128xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<16x128xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<16x1024xi32>, tensor<1024x128xi32>) -> tensor<16x128xi32>
            %r2 = cinm.op.gemm %A, %B: (tensor<16x1024xi32>, tensor<1024x128xi32>) -> tensor<16x128xi32>
            cinm.yield %r : tensor<16x128xi32>
        }
        func.return %r0 : tensor<16x128xi32>
    }
	
    func.func @mm_dimm8_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<8x128xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x128xi32>) -> tensor<8x128xi32>
            %r2 = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x128xi32>) -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }
	
    // func.func @mm_dimm8_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x64xi32>) -> tensor<16x64xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<16x64xi32> {
    //         %r = cinm.op.gemm %A, %B: (tensor<16x1024xi32>, tensor<1024x64xi32>) -> tensor<16x64xi32>
    //         %r2 = cinm.op.gemm %A, %B: (tensor<16x1024xi32>, tensor<1024x64xi32>) -> tensor<16x64xi32>
    //         cinm.yield %r : tensor<16x64xi32>
    //     }
    //     func.return %r0 : tensor<16x64xi32>
    // }
	
    // func.func @mm_dimm16_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x64xi32>) -> tensor<8x64xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 16, 128, 1> } -> tensor<8x64xi32> {
    //         %r = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x64xi32>) -> tensor<8x64xi32>
    //         %r2 = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x64xi32>) -> tensor<8x64xi32>
    //         cinm.yield %r : tensor<8x64xi32>
    //     }
    //     func.return %r0 : tensor<8x64xi32>
    // }
	
    // func.func @mm_dimm16_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x32xi32>) -> tensor<16x32xi32> {

    //     %r0 = cinm.compute attributes { workgroupShape=array<i64: 16, 128, 1> } -> tensor<16x32xi32> {
    //         %r = cinm.op.gemm %A, %B: (tensor<16x1024xi32>, tensor<1024x32xi32>) -> tensor<16x32xi32>
    //         %r2 = cinm.op.gemm %A, %B: (tensor<16x1024xi32>, tensor<1024x32xi32>) -> tensor<16x32xi32>
    //         cinm.yield %r : tensor<16x32xi32>
    //     }
    //     func.return %r0 : tensor<16x32xi32>
    // }

}
