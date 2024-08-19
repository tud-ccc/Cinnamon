#space_partition = [
  affine_map<(d0, d1) -> (0.5 * d0, 0.75 * d1)>,
  affine_map<(d0, d1) -> (0.5 * d0, 0.25 * d1)>,
  affine_map<(d0, d1) -> (0.5 * d0, 0.25 * d1)>,
  affine_map<(d0, d1) -> (0.5 * d0, 0.75 * d1)>
]
// Notes: Outside mapping is heterogeneous, but the inside is homogeneous

module {
    func.func @gemv_f(%A: tensor<4096x4096xi32>, %B: tensor<4096xi32>) {
        %res = cinm.compute  attributes { target_map = #space_partition,  
            workgroupShapes=[array<i64: 16, 64, 1, 8>, array<i64: 16, 64, 1, 8>,
                             array<i64: 16, 64, 1, 8>, array<i64: 16, 64, 1, 8>],  
            targets=[samsung, upmem, samsung, upmem]} -> tensor<4096xi32> {

            %r = cinm.op.gemv %A, %B: tensor<4096x4096xi32>, tensor<4096xi32>
            cinm.yield %r: tensor<4096xi32>
        }
        func.return
    }
}
