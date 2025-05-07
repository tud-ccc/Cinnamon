module {
    
    func.func @mm_dimm4_nopt(%A: tensor<25088x16xi32>, %B: tensor<16x256xi32>) -> tensor<25088x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<25088x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<25088x16xi32>, tensor<16x256xi32>) -> tensor<25088x256xi32>
            cinm.yield %r : tensor<25088x256xi32>
        }
        func.return %r0 : tensor<25088x256xi32>
    }

    func.func @mm_dimm4_opt(%A: tensor<6272x64xi32>, %B: tensor<64x256xi32>) -> tensor<6272x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 4, 128, 1> } -> tensor<6272x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<6272x64xi32>, tensor<64x256xi32>) -> tensor<6272x256xi32>
            cinm.yield %r : tensor<6272x256xi32>
        }
        func.return %r0 : tensor<6272x256xi32>
    }
	
    func.func @mm_dimm8_nopt(%A: tensor<25088x8xi32>, %B: tensor<8x256xi32>) -> tensor<25088x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<25088x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<25088x8xi32>, tensor<8x256xi32>) -> tensor<25088x256xi32>
            cinm.yield %r : tensor<25088x256xi32>
        }
        func.return %r0 : tensor<25088x256xi32>
    }
	
    func.func @mm_dimm8_opt(%A: tensor<3136x64xi32>, %B: tensor<64x256xi32>) -> tensor<3136x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<3136x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<3136x64xi32>, tensor<64x256xi32>) -> tensor<3136x256xi32>
            cinm.yield %r : tensor<3136x256xi32>
        }
        func.return %r0 : tensor<3136x256xi32>
    }
	
    func.func @mm_dimm16_nopt(%A: tensor<25088x4xi32>, %B: tensor<4x256xi32>) -> tensor<25088x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 16, 128, 1> } -> tensor<25088x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<25088x4xi32>, tensor<4x256xi32>) -> tensor<25088x256xi32>
            cinm.yield %r : tensor<25088x256xi32>
        }
        func.return %r0 : tensor<25088x256xi32>
    }

    func.func @mm_dimm16_opt(%A: tensor<1568x64xi32>, %B: tensor<64x256xi32>) -> tensor<1568x256xi32> {

        %r0 = cinm.compute attributes { workgroupShape=array<i64: 16, 128, 1> } -> tensor<1568x256xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<1568x64xi32>, tensor<64x256xi32>) -> tensor<1568x256xi32>
            cinm.yield %r : tensor<1568x256xi32>
        }
        func.return %r0 : tensor<1568x256xi32>
    }
}
