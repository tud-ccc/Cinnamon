// These dims and sizes aren't correct for this model
// dim: 512
// hidden_dim: 1536
// kv_dim: 512
// kv_mul: 1
// n_blocks: 18 (layers)
// n_heads: 8
// head_size: 64
// vocab_size: 32000 > should be deleted?
// seq_len T: 512

func.func @forward(%token : index, %pos : index,
	// state
	%kc : memref<18x512x512xf32>, // blocks, T, C
	%vc : memref<18x512x512xf32>, // blocks, T, C
	// weights
	%embedding_table : tensor<32000x512xf32>,
	%rms_att_weights : tensor<18x512xf32>,
	%wq : tensor<18x512x512xf32>,
	%wk : tensor<18x512x512xf32>,
	%wv : tensor<18x512x512xf32>,
	%wo : tensor<18x512x512xf32>,
	%w1 : tensor<18x1536x512xf32>,
	%w2 : tensor<18x512x1536xf32>,
	%w3 : tensor<18x1536x512xf32>,
	%rms_ffn_weights : tensor<18x512xf32>,
	%rms_final_weight : tensor<512xf32>,
	%wcls : tensor<32000x512xf32>
) -> tensor<32000xf32> {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c2 = arith.constant 2 : index
	%n_blocks = arith.constant 18 : index
	%head_size = arith.constant 64 : index
	%head_size_f32 = arith.constant 64.0 : f32
	%dim = arith.constant 512 : index
	%len = arith.constant 512 : index
	%hidden_dim = arith.constant 1536 : index

    // Required by language limitations
	%c0f = arith.constant 0.0 : f32
	%c1f = arith.constant 1.0 : f32
	%c10000f = arith.constant 10000.0 : f32

	%content_row = tensor.extract_slice %embedding_table [%token, 0] [1, 512] [1, 1] : tensor<32000x512xf32> to tensor<512xf32>

	%x = scf.for %layer = %c0 to %n_blocks step %c1 iter_args(%x = %content_row) -> (tensor<512xf32>) {
		%rms_att_weight = tensor.extract_slice %rms_att_weights [%layer, 0] [1, 512] [1, 1] : tensor<18x512xf32> to tensor<512xf32>
		%xb = func.call @rmsnorm(%x, %rms_att_weight) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>

		// qkv matmuls
		%wqs = tensor.extract_slice %wq [%layer, 0, 0] [1, 512, 512] [1, 1, 1] : tensor<18x512x512xf32> to tensor<512x512xf32>
		%wks = tensor.extract_slice %wk [%layer, 0, 0] [1, 512, 512] [1, 1, 1] : tensor<18x512x512xf32> to tensor<512x512xf32>
		%wvs = tensor.extract_slice %wv [%layer, 0, 0] [1, 512, 512] [1, 1, 1] : tensor<18x512x512xf32> to tensor<512x512xf32>
		%q, %k, %v = cinm.compute attributes { workgroupShape = array<i64: 1,8,8> } -> tensor<512xf32>, tensor<512xf32>, tensor<512xf32> {
			%q = cinm.op.gemv %wqs, %xb : (tensor<512x512xf32>, tensor<512xf32>) -> tensor<512xf32>
			%k = cinm.op.gemv %wks, %xb : (tensor<512x512xf32>, tensor<512xf32>) -> tensor<512xf32>
			%v = cinm.op.gemv %wvs, %xb : (tensor<512x512xf32>, tensor<512xf32>) -> tensor<512xf32>
			cinm.yield %q, %k, %v : tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
		}

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		%posi = arith.index_cast %pos : index to i64
		%posf = arith.uitofp %posi : i64 to f32
		%q2, %k2 = scf.for %i = %c0 to %dim step %c2 iter_args(%qi = %q, %ki = %k) -> (tensor<512xf32>, tensor<512xf32>) {
			%head_dim = arith.remui %i, %head_size : index
			%head_dimi = arith.index_cast %head_dim : index to i64
			%head_dimf = arith.uitofp %head_dimi : i64 to f32
			%0 = arith.divf %head_dimf, %head_size_f32 : f32
			%1 = math.powf %c10000f, %0 : f32
			%freq = arith.divf %c1f, %1 : f32
			%val = arith.mulf %posf, %freq : f32
			%fcr = math.cos %val : f32
			%fci = math.sin %val : f32

			%qr = func.call @rot(%qi, %i, %fcr, %fci) : (tensor<512xf32>, index, f32, f32) -> tensor<512xf32>

			%cond = arith.cmpi ult, %i, %dim : index
			%kr = scf.if %cond -> (tensor<512xf32>) {
				%kr = func.call @rot(%ki, %i, %fcr, %fci) : (tensor<512xf32>, index, f32, f32) -> tensor<512xf32>
				scf.yield %kr : tensor<512xf32>
			} else {
				scf.yield %ki : tensor<512xf32>
			}

			scf.yield %qr, %kr : tensor<512xf32>, tensor<512xf32>
		}

		%kmr = bufferization.to_memref %k2 : memref<512xf32>
		%vmr = bufferization.to_memref %v : memref<512xf32>

		%kcd = memref.subview %kc [%layer, %pos, 0] [1, 1, 512] [1, 1, 1] : memref<18x512x512xf32> to memref<512xf32, strided<[1], offset: ?>> // blocks, T, C
		%vcd = memref.subview %vc [%layer, %pos, 0] [1, 1, 512] [1, 1, 1] : memref<18x512x512xf32> to memref<512xf32, strided<[1], offset: ?>> // blocks, T, C

		memref.copy %vmr, %vcd : memref<512xf32> to memref<512xf32, strided<[1], offset: ?>>
		memref.copy %kmr, %kcd : memref<512xf32> to memref<512xf32, strided<[1], offset: ?>>

		// multi head attention
		%lkc = memref.subview %kc [%layer, 0, 0] [1, 512, 512] [1, 1, 1] : memref<18x512x512xf32> to memref<512x512xf32, strided<[512, 1], offset: ?>> // blocks, T, C
		%lkc2 = bufferization.to_tensor %lkc : memref<512x512xf32, strided<[512, 1], offset: ?>>
		%lvc = memref.subview %vc [%layer, 0, 0] [1, 512, 512] [1, 1, 1] : memref<18x512x512xf32> to memref<512x512xf32, strided<[512, 1], offset: ?>> // blocks, T, C
		%lvc2 = bufferization.to_tensor %lvc : memref<512x512xf32, strided<[512, 1], offset: ?>>
		%xb2 = func.call @mha(%q2, %lkc2, %lvc2, %pos) : (tensor<512xf32>, tensor<512x512xf32>, tensor<512x512xf32>, index) -> tensor<512xf32> // T, C

		%wo_slice = tensor.extract_slice %wo [%layer, 0, 0] [1, 512, 512] [1, 1, 1] : tensor<18x512x512xf32> to tensor<512x512xf32>
		%xb4 = cinm.compute attributes { workgroupShape = array<i64: 1,8,8> } -> tensor<512xf32> {
			// final matmul to get the output of the attention
			%xb3 = cinm.op.gemv %wo_slice, %xb2 : (tensor<512x512xf32>, tensor<512xf32>) -> tensor<512xf32>

			// residual connection back into x
			%xb4 = cinm.op.add %x, %xb3 : tensor<512xf32>
			cinm.yield %xb4 : tensor<512xf32>
		}

		// ffn rmsnorm
		%rms_ffn_weight = tensor.extract_slice %rms_ffn_weights [%layer, 0] [1, 512] [1, 1] : tensor<18x512xf32> to tensor<512xf32>
		%xb5 = func.call @rmsnorm(%xb4, %rms_ffn_weight) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		%w1_slice = tensor.extract_slice %w1 [%layer, 0, 0] [1, 1536, 512] [1, 1, 1] : tensor<18x1536x512xf32> to tensor<1536x512xf32>
		%w3_slice = tensor.extract_slice %w3 [%layer, 0, 0] [1, 1536, 512] [1, 1, 1] : tensor<18x1536x512xf32> to tensor<1536x512xf32>
		%hb1, %hb2 = cinm.compute attributes { workgroupShape = array<i64: 1,8,8> } -> tensor<1536xf32>, tensor<1536xf32> {
			%hb1 = cinm.op.gemv %w1_slice, %xb5 : (tensor<1536x512xf32>, tensor<512xf32>) -> tensor<1536xf32>
			%hb2 = cinm.op.gemv %w3_slice, %xb5 : (tensor<1536x512xf32>, tensor<512xf32>) -> tensor<1536xf32>
			cinm.yield %hb1, %hb2 : tensor<1536xf32>, tensor<1536xf32>
		}

		// SwiGLU non-linearity
		%hb3 = scf.for %i = %c0 to %hidden_dim step %c1 iter_args(%hb = %hb1) -> (tensor<1536xf32>) {
			%0 = tensor.extract %hb [%i] : tensor<1536xf32>
			%1 = tensor.extract %hb2 [%i] : tensor<1536xf32>
			%2 = math.exp %0 : f32
			%3 = arith.addf %c1f, %2 : f32
			%4 = arith.divf %c1f, %3 : f32
			%5 = arith.mulf %1, %4 : f32
			%hbr = tensor.insert %5 into %hb [%i] : tensor<1536xf32>
			scf.yield %hbr : tensor<1536xf32>
		}

		%w2_slice = tensor.extract_slice %w2 [%layer, 0, 0] [1, 512, 1536] [1, 1, 1] : tensor<18x512x1536xf32> to tensor<512x1536xf32>
		%xb7 = cinm.compute attributes { workgroupShape = array<i64: 1,8,8> } -> tensor<512xf32> {
			// final matmul to get the output of the ffn
			%xb6 = cinm.op.gemv %w2_slice, %hb3 : (tensor<512x1536xf32>, tensor<1536xf32>) -> tensor<512xf32>

			// residual connection
			%xb7 = cinm.op.add %x, %xb6 : tensor<512xf32>
			cinm.yield %xb7 : tensor<512xf32>
		}

		scf.yield %xb7 : tensor<512xf32>
	}

	%x2 = func.call @rmsnorm(%x, %rms_final_weight) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
	%logits = cinm.compute attributes { workgroupShape = array<i64: 2,8,16> } -> tensor<32000xf32> {
		%wcls2 = tensor.pad %wcls low[0,0] high[%hidden_dim,0] {
		^bb0(%arg1: index, %arg2: index):
			tensor.yield %c0f : f32
		} : tensor<32000x512xf32> to tensor<32768x512xf32>
		%logits = cinm.op.gemv %wcls2, %x2 : (tensor<32768x512xf32>, tensor<512xf32>) -> tensor<32768xf32>
		%logits2 = tensor.extract_slice %logits [0] [32000] [1] : tensor<32768xf32> to tensor<32000xf32>
		cinm.yield %logits2 : tensor<32000xf32>
	}

	return %logits : tensor<32000xf32>
}

func.func @rot(%v: tensor<512xf32>, %i: index, %fcr : f32, %fci : f32) -> tensor<512xf32> {
	%c1 = arith.constant 1 : index
	%i2 = arith.addi %i, %c1 : index
	%v0 = tensor.extract %v [%i] : tensor<512xf32>
	%v1 = tensor.extract %v [%i2] : tensor<512xf32>
	%0 = arith.mulf %v0, %fcr : f32
	%1 = arith.mulf %v1, %fci : f32
	%2 = arith.subf %0, %1 : f32
	%r0 = tensor.insert %2 into %v[%i] : tensor<512xf32>
	%3 = arith.mulf %v0, %fci : f32
	%4 = arith.mulf %v1, %fcr : f32
	%5 = arith.addf %3, %4 : f32
	%r1 = tensor.insert %2 into %r0[%i] : tensor<512xf32>
	return %r1 : tensor<512xf32>
}


// Q: features, KC: sequence length x features, VC: sequence length x features
func.func @mha(%q: tensor<512xf32>, %kc: tensor<512x512xf32>, %vc: tensor<512x512xf32>, %pos: index) -> tensor<512xf32> {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%nheads = arith.constant 8 : index
	%head_dim = arith.constant 64 : index
	%c0f = arith.constant 0.0 : f32
	%scale = arith.constant 8.0 : f32 // sqrt(head_dim)
	%ninf = arith.constant 0xFF800000 : f32

	%pos2 = arith.addi %pos, %c1 : index

	%attn_init = tensor.generate {
	^bb0(%arg1: index):
		tensor.yield %ninf : f32
	} : tensor<512xf32>

	%xb_init = tensor.empty() : tensor<512xf32>
	%xb = scf.for %head = %c0 to %nheads step %c1 iter_args(%xbi = %xb_init) -> (tensor<512xf32>) {
		%hoff = arith.muli %head, %head_dim : index

		%attn = scf.for %i = %c0 to %pos2 step %c1 iter_args(%attn_i = %attn_init) -> (tensor<512xf32>) {
			%qs = tensor.extract_slice %q [%hoff] [64] [1] : tensor<512xf32> to tensor<64xf32>
			%k = tensor.extract_slice %kc [%i, %hoff] [1, 64] [1, 1] : tensor<512x512xf32> to tensor<64xf32>
			%score = cinm.compute attributes { workgroupShape = array<i64: 1,1,8> } -> f32 {
				%0 = cinm.op.mul %qs, %k : tensor<64xf32>
				%1 = cinm.op.reduce add (%0) : tensor<64xf32>
				%2 = arith.divf %1, %scale : f32
				cinm.yield %2 : f32
			}
			%attn_i2 = tensor.insert %score into %attn_i [%i] : tensor<512xf32>
			scf.yield %attn_i2 : tensor<512xf32>
		}

		%attn3 = func.call @softmax(%attn) : (tensor<512xf32>) -> tensor<512xf32>

		%xb_slice_init = tensor.generate {
		^bb0(%arg1: index):
			tensor.yield %c0f : f32
		} : tensor<64xf32>

		%xb_slice = scf.for %i = %c0 to %pos2 step %c1 iter_args(%xb_slice_i = %xb_slice_init) -> (tensor<64xf32>) {
			%v = tensor.extract_slice %vc [%i, %hoff] [1, 64] [1, 1] : tensor<512x512xf32> to tensor<64xf32>
			%a = tensor.extract %attn3 [%i] : tensor<512xf32>
			%xb_slice = cinm.compute attributes { workgroupShape = array<i64: 1,1,8> } -> tensor<64xf32> {
				%0 = cinm.op.muls %v, %a : tensor<64xf32>
				%1 = cinm.op.add %xb_slice_i, %0 : tensor<64xf32>
				cinm.yield %1 : tensor<64xf32>
			}
			scf.yield %xb_slice : tensor<64xf32>
		}

		%xbr = tensor.insert_slice %xb_slice into %xbi [%hoff] [64] [1] : tensor<64xf32> into tensor<512xf32>
		scf.yield %xbr : tensor<512xf32>
	}

	return %xb : tensor<512xf32>
}

func.func @rmsnorm(%v : tensor<512xf32>, %w : tensor<512xf32>) -> tensor<512xf32> {
	%epsilon = arith.constant 1.0e-5 : f32
	%c1 = arith.constant 1.0 : f32
	%len = arith.constant 512.0 : f32

	%r = cinm.compute attributes { workgroupShape = array<i64: 1,1,16> } -> tensor<512xf32> {
		%0 = cinm.op.mul %v, %v : tensor<512xf32>
		%ss = cinm.op.reduce add (%0) : tensor<512xf32>
		%s0 = arith.divf %ss, %len : f32
		%s1 = arith.addf %s0, %epsilon : f32
		%s = math.rsqrt %s1 : f32
		%x = cinm.op.muls %v, %s : tensor<512xf32>
		%r = cinm.op.mul %x, %w : tensor<512xf32>
		cinm.yield %r : tensor<512xf32>
	}
	return %r : tensor<512xf32>
}

//func.func @rmsnorm(%v : tensor<262144xf32>, %w : tensor<262144xf32>) -> tensor<262144xf32> {
//	%epsilon = arith.constant 1.0e-5 : f32
//	%c1 = arith.constant 1.0 : f32
//	%len = arith.constant 262144.0 : f32
//
//	%r = cinm.compute attributes { workgroupShape = array<i64: 1,1,16> } -> tensor<262144xf32> {
//		%0 = cinm.op.mul %v, %v : tensor<262144xf32>
//		%ss = cinm.op.reduce add (%0) : tensor<262144xf32>
//		%s0 = arith.divf %ss, %len : f32
//		%s1 = arith.addf %s0, %epsilon : f32
//		%s = math.rsqrt %s1 : f32
//		%x = cinm.op.muls %v, %s : tensor<262144xf32>
//		%r = cinm.op.mul %x, %w : tensor<262144xf32>
//		cinm.yield %r : tensor<262144xf32>
//	}
//	return %r : tensor<262144xf32>
//}

func.func @softmax(%vec : tensor<512xf32>) -> tensor<512xf32> {
	%r = cinm.compute attributes { workgroupShape = array<i64: 1,8,16> } -> tensor<512xf32> {
		%max = cinm.op.reduce max (%vec) : tensor<512xf32>
		%t = cinm.op.subs %vec, %max : tensor<512xf32>
		%shape = tensor.empty() : tensor<512xf32>
		%e = linalg.exp ins(%t : tensor<512xf32>) outs(%shape : tensor<512xf32>) -> tensor<512xf32>
		%s = cinm.op.reduce add (%e) : tensor<512xf32>
		%r = cinm.op.divs %e, %s : tensor<512xf32>
		cinm.yield %r : tensor<512xf32>
	}

	return %r : tensor<512xf32>
}
