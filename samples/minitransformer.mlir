

func.func @mha(%attn3: tensor<256xf32>, %kc: tensor<256x288xf32>, %vc: tensor<256x288xf32>, %pos: index, %hoff: index) -> tensor<48xf32> {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c6 = arith.constant 6 : index
	%c48 = arith.constant 48 : index
	%c0f = arith.constant 0.0 : f32
	%scale = arith.constant 6.92820323028 : f32 // sqrt(head_size)
	%ninf = arith.constant 0xFF800000 : f32
	%pos2 = arith.addi %pos, %c1 : index

		%xb_slice_init = tensor.generate {
		^bb0(%arg1: index):
			tensor.yield %c0f : f32
		} : tensor<48xf32>
	%xb_slice = scf.for %i = %c0 to %pos2 step %c1 iter_args(%xb_slice_i = %xb_slice_init) -> (tensor<48xf32>) {
			%v = tensor.extract_slice %vc [%i, %hoff] [1, 48] [1, 1] : tensor<256x288xf32> to tensor<48xf32>
			%a = tensor.extract %attn3 [%i] : tensor<256xf32>
			%xb_slice = cinm.compute -> tensor<48xf32> attributes { workgroupShape = array<i64: 1,1,8> } {
        %av = tensor.splat %a : tensor<48xf32>
        %e = tensor.empty() : tensor<48xf32>
				%0 = linalg.mul ins(%v, %av: tensor<48xf32>, tensor<48xf32>) outs (%e: tensor<48xf32>) -> tensor<48xf32>
				%1 = linalg.add ins(%xb_slice_i, %0: tensor<48xf32>, tensor<48xf32>) outs (%e: tensor<48xf32>) -> tensor<48xf32>
				cinm.yield %1 : tensor<48xf32>
			}
      // todo not equivalent.
      //  - Review bufferization implementations for compute.
      //    Make sure equivalence is preserved between args/bbargs and yield/results.
      //  - Review bufferization implementation for elementwise.
			scf.yield %xb_slice : tensor<48xf32>
		}
    return %xb_slice : tensor<48xf32>
}
