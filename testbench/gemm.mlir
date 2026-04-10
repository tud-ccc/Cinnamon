func.func @main() {
	%a = tensor.empty() : tensor<64x64xi32>
	%b = tensor.empty() : tensor<64x64xi32>
	%res = cinm.compute_ -> tensor<64x64xi32> attributes{ workgroupShape = array<i64: 1, 16, 1> } {
		%d = cinm.op.gemm %a, %b : tensor<64x64xi32>, tensor<64x64xi32> -> tensor<64x64xi32>
		cinm.yield %d : tensor<64x64xi32>
	}
	return
}
