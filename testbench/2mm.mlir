#upmem = #upmem.platform<30x64>
module {

    func.func @mm_dimm4_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x256xi32>,  %C: tensor<256x2048xi32>) -> tensor<8x2048xi32>
    attributes { cinm.available_platforms = [#cinm.host_platform, #upmem] } {


        %r = cinm.compute_ -> tensor<8x2048xi32> 
        attributes { workgroupShape=array<i64: 2, 4, 16> } {
            %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x256xi32> -> tensor<8x256xi32>
            %r2 = cinm.op.gemm %r, %C : tensor<8x256xi32>, tensor<256x2048xi32> -> tensor<8x2048xi32>
            cinm.yield %r2 : tensor<8x2048xi32>
        }
        func.return %r : tensor<8x2048xi32>
    }
	
    func.func @mm_dimm4_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x128xi32>, %C: tensor<128x2048xi32>) -> tensor<16x2048xi32> {

        %r0 = cinm.compute_ -> tensor<16x2048xi32> attributes { workgroupShape=array<i64: 4, 128, 1> } {
            %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x128xi32> -> tensor<16x128xi32>
            %r2 = cinm.op.gemm %r, %C : tensor<16x128xi32>, tensor<128x2048xi32> -> tensor<16x2048xi32>
            cinm.yield %r2 : tensor<16x2048xi32>
        }
        func.return %r0 : tensor<16x2048xi32>
    }
	
    func.func @mm_dimm8_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>, %C: tensor<128x2048xi32>) -> tensor<8x2048xi32> {

        %r0 = cinm.compute_ -> tensor<8x2048xi32> attributes { workgroupShape=array<i64: 8, 128, 1> } {
            %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
            %r2 = cinm.op.gemm %r, %C : tensor<8x128xi32>, tensor<128x2048xi32> -> tensor<8x2048xi32>
            cinm.yield %r2 : tensor<8x2048xi32>
        }
        func.return %r0 : tensor<8x2048xi32>
    }
	
    // func.func @mm_dimm8_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x64xi32>) -> tensor<16x64xi32> {

    //     %r0 = cinm.compute_ -> tensor<16x64xi32> attributes { workgroupShape=array<i64: 8, 128, 1> } {
    //         %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x64xi32> -> tensor<16x64xi32>
    //         %r2 = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x64xi32> -> tensor<16x64xi32>
    //         cinm.yield %r : tensor<16x64xi32>
    //     }
    //     func.return %r0 : tensor<16x64xi32>
    // }
	
    // func.func @mm_dimm16_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x64xi32>) -> tensor<8x64xi32> {

    //     %r0 = cinm.compute_ -> tensor<8x64xi32> attributes { workgroupShape=array<i64: 16, 128, 1> } {
    //         %r = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x64xi32> -> tensor<8x64xi32>
    //         %r2 = cinm.op.gemm %A, %B : tensor<8x1024xi32>, tensor<1024x64xi32> -> tensor<8x64xi32>
    //         cinm.yield %r : tensor<8x64xi32>
    //     }
    //     func.return %r0 : tensor<8x64xi32>
    // }
	
    // func.func @mm_dimm16_opt(%A: tensor<16x1024xi32>, %B: tensor<1024x32xi32>) -> tensor<16x32xi32> {

    //     %r0 = cinm.compute_ -> tensor<16x32xi32> attributes { workgroupShape=array<i64: 16, 128, 1> } {
    //         %r = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x32xi32> -> tensor<16x32xi32>
    //         %r2 = cinm.op.gemm %A, %B : tensor<16x1024xi32>, tensor<1024x32xi32> -> tensor<16x32xi32>
    //         cinm.yield %r : tensor<16x32xi32>
    //     }
    //     func.return %r0 : tensor<16x32xi32>
    // }

}
