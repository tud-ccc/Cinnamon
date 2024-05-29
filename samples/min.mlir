func.func @main() {
	%a = tensor.empty() : tensor<1024xi32>
	%b = cinm.op.min %a : tensor<1024xi32>
	%c = cinm.op.max %a : tensor<1024xi32>
	return
}
