module {
    func.func @gemv_f(%A: tensor<4096x4096xi32>, %B: tensor<4096xi32>) {
        %res = cinm.compute  attributes { target_map = #space_partition,  
            workgroupShapes=[array<i64: 16, 64, 1, 8>, array<i64: 16, 64, 1, 8>,
                             array<i64: 16, 64, 1, 8>, array<i64: 16, 64, 1, 8>],  
            targets=[samsung, upmem, samsung, upmem], 
            cost = [exec = array<i64: 132000, 31231, 12431>, 
                    comm = array<i64: 123, 121, 1231, 412>]} -> tensor<4096xi32> {

            %r = cinm.op.gemv %A, %B: tensor<4096x4096xi32>, tensor<4096xi32>
            cinm.yield %r: tensor<4096xi32>
        }
        func.return
    }
}
