func.func @main() {
	%a = tensor.empty() : tensor<64x64xi32>
	%b = tensor.empty() : tensor<64x64xi32>
	%res = cinm.compute attributes{ workgroupShape = array<i64: 1, 16, 1> }-> tensor<64x64xi32> {
		%d = cinm.op.gemm %a, %b : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
		cinm.yield %d : tensor<64x64xi32>
	}
	return
}
