
func.func @softmax_1048576(%vec : tensor<1048576xf32>) -> tensor<1048576xf32> {
	%r = cinm.compute attributes { workgroupShape = array<i64: 4,64,16> } -> tensor<1048576xf32> {
		%max = cinm.op.reduce max (%vec) : tensor<1048576xf32> -> f32
		%t = cinm.op.subs %vec, %max : tensor<1048576xf32>
		%shape = tensor.empty() : tensor<1048576xf32>
		%e = cinm.op.element_wise exp (%t) : tensor<1048576xf32>
		%s = cinm.op.reduce add (%e) : tensor<1048576xf32> -> f32
		%r = cinm.op.divs %e, %s : tensor<1048576xf32>
		cinm.yield %r : tensor<1048576xf32>
	}

	return %r : tensor<1048576xf32>
}