// cinm-opt $FILE --convert-cinm-to-cnm

module {

    func.func @va_8(%A: tensor<8x2097152xi32>, %B: tensor<8x2097152xi32>) {
        
        
        %res = cinm.compute attributes { workgroupShape=array<i64: 16, 64, 16> } -> tensor<8x2097152xi32> {
            %r = cinm.op.add %A, %B: tensor<8x2097152xi32>
            cinm.yield %r: tensor<8x2097152xi32>
        }

        func.return
    }
    func.func @va_16(%A: tensor<16x1048576xi32>, %B: tensor<16x1048576xi32>) {
        
        %res = cinm.compute attributes { workgroupShape=array<i64: 32, 64, 8> } -> tensor<16x1048576xi32> {
            %r = cinm.op.add %A, %B: tensor<16x1048576xi32>
            cinm.yield %r: tensor<16x1048576xi32>
        }

        func.return
    }
}
