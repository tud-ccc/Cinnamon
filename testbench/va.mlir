// cinm-opt $FILE --convert-cinm-to-cnm

module {

    func.func @va_8(%A: tensor<8x2097152xi32>, %B: tensor<8x2097152xi32>) {
        
        
        %res = cinm.compute_ -> tensor<8x2097152xi32> attributes { workgroupShape=array<i64: 16, 64, 16> } {
            %r = cinm.op.elementwise add %A, %B: tensor<8x2097152xi32>
            cinm.yield %r: tensor<8x2097152xi32>
        }

        func.return
    }
    func.func @va_16(%A: tensor<16x1048576xi32>, %B: tensor<16x1048576xi32>) {
        
        %res = cinm.compute_ -> tensor<16x1048576xi32> attributes { workgroupShape=array<i64: 32, 64, 8> } {
            %r = cinm.op.elementwise add %A, %B: tensor<16x1048576xi32>
            cinm.yield %r: tensor<16x1048576xi32>
        }

        func.return
    }
}
