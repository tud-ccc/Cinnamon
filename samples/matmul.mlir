func.func @main() {
	%a = tensor.empty() : tensor<1024x1024xi32>
	%b = tensor.empty() : tensor<1024x1024xi32>
	%d = "cinm.gemm" (%a, %b) : (tensor<1024x1024xi32>, tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
	return
}
