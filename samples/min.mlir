func.func @main() {
	%a = tensor.empty() : tensor<1024xi32>

	%x,%y = cinm.compute -> i32, i32 {
		%b = cinm.op.min %a : tensor<1024xi32>
		%c = cinm.op.max %a : tensor<1024xi32>
		cinm.yield %b, %c : i32, i32
	}
	return
}
