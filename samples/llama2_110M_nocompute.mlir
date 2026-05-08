// transformer model for https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
// Source here: https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/run.c#L231-L362
// dim: 768
// hidden_dim: 2048
// kv_dim: 768
// kv_mul: 1
// n_layers: 6
// n_heads: 6
// n_kv_heads: 6
// head_size: 48
// vocab_size: 32000
// seq_len: 1024

func.func @forward(%token : index, %pos : index,
	// state
	%kc : tensor<6x1024x768xf32> {bufferization.writable = true, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%vc : tensor<6x1024x768xf32> {bufferization.writable = true, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	// weights
	%embedding_table : tensor<32000x768xf32> {bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%rms_att_weights : tensor<6x768xf32>{bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%wq : tensor<6x768x768xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wk : tensor<6x768x768xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wv : tensor<6x768x768xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wo : tensor<6x768x768xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%w1 : tensor<6x2048x768xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%w2 : tensor<6x768x2048xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%w3 : tensor<6x2048x768xf32>{bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%rms_ffn_weights : tensor<6x768xf32>{bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%rms_final_weight : tensor<768xf32>{bufferization.buffer_layout = affine_map<(i) -> (i)>},
	%wcls : tensor<32000x768xf32>{bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>}
) -> tensor<32000xf32> {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c2 = arith.constant 2 : index
	%c6 = arith.constant 6 : index
	%c48 = arith.constant 48 : index
	%c768 = arith.constant 768 : index
	%c2048 = arith.constant 2048 : index
	%c0f = arith.constant 0.0 : f32
	%c1f = arith.constant 1.0 : f32
	%c48f = arith.constant 48.0 : f32
	%c10000f = arith.constant 10000.0 : f32

	%content_row = tensor.extract_slice %embedding_table [%token, 0] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<768xf32>

	%x, %kc2, %vc2 = scf.for %layer = %c0 to %c6 step %c1 iter_args(%x = %content_row, %kc0 = %kc, %vc0 = %vc) -> (tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>) {
		%rms_att_weight = tensor.extract_slice %rms_att_weights [%layer, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
		%xb = func.call @rmsnorm(%x, %rms_att_weight) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>

		// qkv matmuls
		%wqs = tensor.extract_slice %wq [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
		%wks = tensor.extract_slice %wk [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
		%wvs = tensor.extract_slice %wv [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>

		%kdest = tensor.extract_slice %kc0 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :   tensor<6x1024x768xf32> to tensor<768xf32>
		%vdest = tensor.extract_slice %vc0 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :   tensor<6x1024x768xf32> to tensor<768xf32>

    %q = cinm.op.gemv %wqs, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %k = cinm.op.gemv %wks, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %v = cinm.op.gemv %wvs, %xb : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
    %kc1 = tensor.insert_slice %k into %kc0[%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :  tensor<768xf32> into tensor<6x1024x768xf32> 
    //%k0 = bufferization.materialize_in_destination %k in %kdest : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %vc1 = tensor.insert_slice %v into %vc0[%layer, %pos, 0] [1, 1, 768] [1, 1, 1] :  tensor<768xf32> into tensor<6x1024x768xf32> 

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		%posi = arith.index_cast %pos : index to i64
		%posf = arith.uitofp %posi : i64 to f32


		%q2, %k2 = scf.for %i = %c0 to %c768 step %c2 iter_args(%qi = %q, %ki = %k) -> (tensor<768xf32>, tensor<768xf32>) {
			%head_dim = arith.remui %i, %c48 : index
			%head_dimi = arith.index_cast %head_dim : index to i64
			%head_dimf = arith.uitofp %head_dimi : i64 to f32
			%0 = arith.divf %head_dimf, %c48f : f32
			%1 = math.powf %c10000f, %0 : f32
			%freq = arith.divf %c1f, %1 : f32
			%val = arith.mulf %posf, %freq : f32
			%fcr = math.cos %val : f32
			%fci = math.sin %val : f32

			%qr = func.call @rot(%qi, %i, %fcr, %fci) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>

			%cond = arith.cmpi ult, %i, %c768 : index
			%kr = scf.if %cond -> (tensor<768xf32>) {
				%kr = func.call @rot(%ki, %i, %fcr, %fci) : (tensor<768xf32>, index, f32, f32) -> tensor<768xf32>
				scf.yield %kr : tensor<768xf32>
			} else {
				scf.yield %ki : tensor<768xf32>
			}

			scf.yield %qr, %kr : tensor<768xf32>, tensor<768xf32>
		}

		%kc2 = tensor.insert_slice %k2 into %kc1 [%layer, %pos, 0] [1, 1, 768] [1, 1, 1] : tensor<768xf32>  into tensor<6x1024x768xf32>

		// multi head attention
    %lkc = tensor.extract_slice %kc2[%layer, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
    %lvc = tensor.extract_slice %vc1[%layer, 0, 0] [1, 1024, 768] [1, 1, 1] : tensor<6x1024x768xf32> to tensor<1024x768xf32>
		%xb20 = func.call @mha(%q2, %lkc, %lvc, %pos) : (tensor<768xf32>, tensor<1024x768xf32>, tensor<1024x768xf32>, index) -> tensor<768xf32>
    %xb2 = bufferization.materialize_in_destination %xb20 in %q2 : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>

		%wo_slice = tensor.extract_slice %wo [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<6x768x768xf32> to tensor<768x768xf32>
    // final matmul to get the output of the attention
    %xb3 = cinm.op.gemv %wo_slice, %xb2 into %xb2 : tensor<768x768xf32>, tensor<768xf32> into tensor<768xf32>-> tensor<768xf32>

    // residual connection back into x
    %xb4 = cinm.op.elementwise add %x, %xb3 into %xb3 : tensor<768xf32> into tensor<768xf32>

		// ffn rmsnorm
		%rms_ffn_weight = tensor.extract_slice %rms_ffn_weights [%layer, 0] [1, 768] [1, 1] : tensor<6x768xf32> to tensor<768xf32>
		%xb5 = func.call @rmsnorm(%xb4, %rms_ffn_weight) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		%w1_slice = tensor.extract_slice %w1 [%layer, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
		%w3_slice = tensor.extract_slice %w3 [%layer, 0, 0] [1, 2048, 768] [1, 1, 1] : tensor<6x2048x768xf32> to tensor<2048x768xf32>
    %hb1 = cinm.op.gemv %w1_slice, %xb5 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>
    %hb2 = cinm.op.gemv %w3_slice, %xb5 : tensor<2048x768xf32>, tensor<768xf32> -> tensor<2048xf32>

		// SwiGLU non-linearity
    // hb[i] = hb2[i] * hb[i] * (1 / (1 + exp(-hb[i])))
    %hb11 = linalg.map ins(%hb1, %hb2 : tensor<2048xf32>, tensor<2048xf32>) outs(%hb1 : tensor<2048xf32>)
    (%hbi : f32, %hb2i : f32) {
			%2 = arith.negf %hbi : f32
			%3 = math.exp %2 : f32
			%4 = arith.addf %c1f, %3 : f32
			%5 = arith.divf %c1f, %4 : f32
			%6 = arith.mulf %hbi, %5 : f32
			%7 = arith.mulf %6, %hb2i : f32
      linalg.yield %7 : f32
    }

		%w2_slice = tensor.extract_slice %w2 [%layer, 0, 0] [1, 768, 2048] [1, 1, 1] : tensor<6x768x2048xf32> to tensor<768x2048xf32>
    // final matmul to get the output of the ffn
    %xb7 = cinm.op.gemv %w2_slice, %hb11 plus %xb5 into %xb5 : tensor<768x2048xf32>, tensor<2048xf32> plus tensor<768xf32> into tensor<768xf32> -> tensor<768xf32> 

		scf.yield %xb7, %kc2, %vc1 : tensor<768xf32>, tensor<6x1024x768xf32>, tensor<6x1024x768xf32>
	}

	%x2 = func.call @rmsnorm(%x, %rms_final_weight) : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>

  %wcls2 = tensor.pad %wcls low[0,0] high[2048,0] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %c0f : f32
  } : tensor<32000x768xf32> to tensor<34048x768xf32>
  %logits0 = cinm.op.gemv %wcls2, %x2 : tensor<34048x768xf32>, tensor<768xf32> -> tensor<34048xf32>
  %logits = tensor.extract_slice %logits0 [0] [32000] [1] : tensor<34048xf32> to tensor<32000xf32>
	return %logits : tensor<32000xf32>
}

func.func @rot(%v: tensor<768xf32> {bufferization.writable = true}, %i: index, %fcr : f32, %fci : f32) -> tensor<768xf32> {
	%c1 = arith.constant 1 : index
	%i2 = arith.addi %i, %c1 : index
	%v0 = tensor.extract %v [%i] : tensor<768xf32>
	%v1 = tensor.extract %v [%i2] : tensor<768xf32>
	%0 = arith.mulf %v0, %fcr : f32
	%1 = arith.mulf %v1, %fci : f32
	%2 = arith.subf %0, %1 : f32
	%r0 = tensor.insert %2 into %v[%i] : tensor<768xf32>
	%3 = arith.mulf %v0, %fci : f32
	%4 = arith.mulf %v1, %fcr : f32
	%5 = arith.addf %3, %4 : f32
	%r1 = tensor.insert %5 into %r0[%i2] : tensor<768xf32>
  %r2 = bufferization.materialize_in_destination %r1 in %v : (tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
	return %r2 : tensor<768xf32>
}

func.func @mha(%q: tensor<768xf32>, %kc: tensor<1024x768xf32>, %vc: tensor<1024x768xf32>, %pos: index) -> tensor<768xf32> {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c6 = arith.constant 6 : index
	%c48 = arith.constant 48 : index
	%c1024 = arith.constant 1024 : index
	%c0f = arith.constant 0.0 : f32
	%scale = arith.constant 6.92820323028 : f32 // sqrt(head_size)
	%ninf = arith.constant 0xFF800000 : f32

	%pos2 = arith.addi %pos, %c1 : index

	%xb_init = tensor.empty() : tensor<768xf32>
	%xb = scf.for %head = %c0 to %c6 step %c1 iter_args(%xbi = %xb_init) -> (tensor<768xf32>) {
		%hoff = arith.muli %head, %c48 : index

    %attn_init = tensor.empty() : tensor<1024xf32> 
    // %attn_init_zeroed = linalg.fill ins(%ninf : f32) outs(%attn_init: tensor<1024xf32>) -> tensor<1024xf32>

		%attn0 = scf.for %i = %c0 to %pos2 step %c1 iter_args(%attn_i = %attn_init) -> (tensor<1024xf32>) {
			%qs = tensor.extract_slice %q [%hoff] [48] [1] : tensor<768xf32> to tensor<48xf32>
			%k = tensor.extract_slice %kc [%i, %hoff] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
      %0 = cinm.op.elementwise mul %qs, %k : tensor<48xf32>
      %1 = cinm.op.reduce add (%0) : tensor<48xf32> -> f32
      %score = arith.divf %1, %scale : f32
			%attn_i2 = tensor.insert %score into %attn_i [%i] : tensor<1024xf32>
			scf.yield %attn_i2 : tensor<1024xf32>
		}

    // fill the rest with ninf
		%attn = scf.for %i = %pos2 to %c1024 step %c1 iter_args(%attn_i = %attn0) -> (tensor<1024xf32>) {
			%attn_i2 = tensor.insert %ninf into %attn_i [%i] : tensor<1024xf32>
			scf.yield %attn_i2 : tensor<1024xf32>
		}

		%attn3 = func.call @softmax(%attn) : (tensor<1024xf32>) -> tensor<1024xf32>

		%xb_slice_init =  tensor.extract_slice %xbi [%hoff] [48] [1] : tensor<768xf32> to tensor<48xf32>
    %init_zeroed = linalg.fill ins(%c0f : f32) outs(%xb_slice_init: tensor<48xf32>) -> tensor<48xf32>
		%xbi0 = tensor.insert_slice %init_zeroed into %xbi [%hoff] [48] [1] : tensor<48xf32> into tensor<768xf32>  

		%xb1 = scf.for %i = %c0 to %pos2 step %c1 iter_args(%xbi1 = %xbi0) -> (tensor<768xf32>) {
      %xb_slice_i =  tensor.extract_slice %xbi1 [%hoff] [48] [1] : tensor<768xf32> to tensor<48xf32>
			%v = tensor.extract_slice %vc [%i, %hoff] [1, 48] [1, 1] : tensor<1024x768xf32> to tensor<48xf32>
			%a = tensor.extract %attn3 [%i] : tensor<1024xf32>
      %av = tensor.splat %a : tensor<48xf32>
      %0 = cinm.op.elementwise mul %v, %av : tensor<48xf32>
      %1 = cinm.op.elementwise add %xb_slice_i, %0 into %xb_slice_i: tensor<48xf32> into tensor<48xf32>
      %xbr = tensor.insert_slice %1 into %xbi1 [%hoff] [48] [1] : tensor<48xf32> into tensor<768xf32>
			scf.yield %xbr : tensor<768xf32>
		}

		scf.yield %xb1 : tensor<768xf32>
	}

	return %xb : tensor<768xf32>
}

func.func @rmsnorm(%v : tensor<768xf32>, %w : tensor<768xf32>) -> tensor<768xf32> {
	%epsilon = arith.constant 1.0e-5 : f32
	%c1 = arith.constant 1.0 : f32
	%c768 = arith.constant 768.0 : f32

  %0 = cinm.op.elementwise mul %v, %v :tensor<768xf32>
  %ss = cinm.op.reduce add (%0) : tensor<768xf32> -> f32
  %s0 = arith.divf %ss, %c768 : f32
  %s1 = arith.addf %s0, %epsilon : f32
  %s = math.rsqrt %s1 : f32
  %sv = tensor.splat %s : tensor<768xf32>
  %x = cinm.op.elementwise mul %v, %sv : tensor<768xf32>
  %r = cinm.op.elementwise mul %x, %w : tensor<768xf32>
	return %r : tensor<768xf32>
}

func.func @softmax(%vec : tensor<1024xf32>{bufferization.writable=true}) -> tensor<1024xf32> {
  %max = cinm.op.reduce maxnumf (%vec) : tensor<1024xf32> -> f32
  %maxv = tensor.splat %max : tensor<1024xf32>
  %t = cinm.op.elementwise sub %vec, %maxv  into %vec: tensor<1024xf32> into tensor<1024xf32>
  %e = cinm.op.elementwise exp %t  into %vec: tensor<1024xf32> into tensor<1024xf32>
  %s = cinm.op.reduce add (%e) : tensor<1024xf32> -> f32
  %sumv = tensor.splat %s : tensor<1024xf32>
  %r = cinm.op.elementwise div %e, %sumv into %vec : tensor<1024xf32> into tensor<1024xf32>

	return %r : tensor<1024xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]}
                attributes{sym_name = "forward"} in %root
        : (!transform.any_op) -> !transform.any_op

    // Returns all 6 scf.for ops in pre-order; the layer loop is first
    %loops = transform.structured.match ops{["scf.for"]} in %func
        : (!transform.any_op) -> !transform.any_op

    %a_loop, %layer_loop  =
        transform.split_handle %loops {overflow_result = 1}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // transform.print %rest_loops: !transform.any_op

    // Full unroll: trip count is 6 (0 to 6 step 1)
    transform.loop.unroll %layer_loop { factor = 6 } : !transform.any_op
    transform.yield
  }
}