func.func @simple() {
    cinm.compute () attributes {workgroupShape = array<i64: 8, 128, 1>, bufferSizesInBytes=array<i64: 0,0,512>} {
        %a00 = tensor.empty (): tensor<1024x1024xi32>
        %a01 = tensor.empty (): tensor<1024x1024xi32>
        %a02 = tensor.empty (): tensor<1024x1024xi32>
        %d4 = cinm.op.gemm %a00, %a01 : tensor<1024x1024xi32>, tensor<1024x1024xi32> -> tensor<1024x1024xi32>
        affine.for %i = 0 to 1024 step 1 {
          %d3 = cinm.op.gemm %a00, %a01 plus %a02 {cinm.notile} : tensor<1024x1024xi32>, tensor<1024x1024xi32> plus tensor<1024x1024xi32> -> tensor<1024x1024xi32>
        }
        cinm.yield
    }

    return
}
