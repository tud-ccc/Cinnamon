func.func @gemv(%t0: tensor<1024xi1024>) -> tensor<1024xi1024>{
	%res = cinm.compute -> tensor<1024xi1024> {
        %x = cinm.op.add %t0, %t0: tensor<1024xi1024>
        cinm.yield %x: tensor<1024xi1024>
	}
	return %res: tensor<1024xi1024>
}