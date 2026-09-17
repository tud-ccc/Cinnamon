// Llama-2 7B, integer-only single-token decode, for the whole-program
// experiment: the decode-shaped end-to-end model at the size where the
// layer weights are worth keeping on the device. Same program as
// llama2_110M_w8a8.mlir with the dimensions of Llama-2 7B (Touvron et al.
// 2023); the two files differ in nothing but shapes.
//
// Model:
//   dim        (H)  4096
//   hidden     (F)  11008
//   layers     (L)  32
//   heads      (A)  32    -> head size 128 (kv heads = heads, no grouping)
//   vocab      (V)  32000
//   sequence   (N)  1024  (the KV cache length; the model's 4096 would be
//                         1 GB of i8 state per step, and the step's cost
//                         scales with it, not with the model)
// One call is one decode step: a token id and its position go in, the
// caches of every layer are updated in place at that position, and the
// logits of the next token come out. Attention runs over the whole cache
// with the positions after %pos masked, so the work of a step does not
// depend on %pos.
//
// Numerics: integer-only, after I-BERT (Kim et al., ICML 2021), with the
// same building blocks roberta_base.mlir uses:
//   RMSNorm   the Newton integer sqrt (I-BERT 3.4), via @irmsnorm
//   Softmax   the exp-by-shift        (I-BERT 3.3), via @isoftmax
//   SiLU      sigmoid from that same exp: x * e^x / (1 + e^x), via @iswiglu
//   RoPE      Q14 cos/sin tables per position (%rope_cos, %rope_sin),
//             rotated in i32 and narrowed back, via @irope
//
// Precision: W8A8. Weights, activations and both caches are i8; every
// matmul is i8 x i8 -> i32 and each accumulator is requantized back to i8
// by a multiply-shift with saturation (@requant_*). The residual stream is
// carried in i32 (a sum of int8 activations does not fit an int8), and the
// RMSNorm weights are i32 for the same reason RoBERTa's LayerNorm
// parameters are. The Q, K, V projections are one 12288x4096 weight, stored
// fused: a serving system lays the checkpoint out that way rather than
// concatenating three matrices per token. The classifier is stored with its
// rows padded to 32768 (a power of two the tiler can split evenly); the pad
// rows are zero in the checkpoint and the logits are the first 32000.
//
// Fixed point: quantized activations are Q4 (a stored q means q / 16), the
// nonlinearity polynomials evaluate in Q14, the RMSNorm weights are Q10 and
// the RoPE tables Q14. The requantization multiplier is the placeholder
// roberta_base.mlir uses, consistent with those conventions, not a value
// calibrated on a checkpoint; the run is faithful in shape, op sequence and
// data movement, not in what it computes.
//
// Weights: 131 MB of embeddings, 32 * (12288 + 4096 + 3 * 11008) * 4096 =
// 6.47 GB of layer weights, 134 MB of classifier, 16.8 MB of RoPE tables --
// 6.75 GB of i8 (and Q14 i32 tables) that stays resident on the device, 3.3
// MB per DPU of a 2048-DPU array; the caches (268 MB, i8) are per-call
// state. Per layer the QKV weight is 50 MB, w1/w2/w3 45 MB each and wo 17
// MB: at these sizes streaming a weight from host DRAM costs more than a
// device call, which is what makes this the decode model that offloads.
//
// The per-head loop is unrolled by the schedule in
// llama2_7B_w8a8.schedule.mlir (the front end preloads it); the layer loop
// stays rolled, each layer's weight being the slice of its stack at the
// loop index that the compute graph keeps resident for every layer.

#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @llama2_7B_w8a8(
	// per-step inputs. The token and its position are i32, as every runtime
	// hands them over; they become `index` only at the gather and cache
	// sites. The mask is additive on the i32 scores: 0 up to %pos, a large
	// negative after it.
	%token : i32,
	%pos : i32,
	%attn_mask : tensor<1024xi32>,
	// state: the key and value caches of every layer, updated in place
	%kc : tensor<32x1024x4096xi8> {bufferization.writable = true, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%vc : tensor<32x1024x4096xi8> {bufferization.writable = true, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	// RoPE, Q14: row p holds cos/sin(p * freq(j)) for the 2048 rotated pairs
	%rope_cos : tensor<1024x2048xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%rope_sin : tensor<1024x2048xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	// weights, stored [layer][out][in]
	%embedding_table : tensor<32000x4096xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%rms_att_weights : tensor<32x4096xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%wqkv : tensor<32x12288x4096xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wo : tensor<32x4096x4096xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%rms_ffn_weights : tensor<32x4096xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%w1 : tensor<32x11008x4096xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%w2 : tensor<32x4096x11008xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%w3 : tensor<32x11008x4096xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%rms_final_weight : tensor<4096xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i) -> (i)>},
	%wcls : tensor<32768x4096xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>}
) -> tensor<32000xi32> attributes {cinm.available_platforms = [#upmem]} {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c32 = arith.constant 32 : index

	%token_idx = arith.index_cast %token : i32 to index
	%pos_idx = arith.index_cast %pos : i32 to index

	// ---- embedding: one row, widened into the i32 residual stream ---------
	%x_row = tensor.extract_slice %embedding_table [%token_idx, 0] [1, 4096] [1, 1] : tensor<32000x4096xi8> to tensor<4096xi8>
	%x_in = func.call @widen_v(%x_row) : (tensor<4096xi8>) -> tensor<4096xi32>

	%cos = tensor.extract_slice %rope_cos [%pos_idx, 0] [1, 2048] [1, 1] : tensor<1024x2048xi32> to tensor<2048xi32>
	%sin = tensor.extract_slice %rope_sin [%pos_idx, 0] [1, 2048] [1, 1] : tensor<1024x2048xi32> to tensor<2048xi32>

	// ---- decoder layers ----------------------------------------------------
	%xL, %kcL, %vcL = scf.for %layer = %c0 to %c32 step %c1
			iter_args(%x = %x_in, %kc0 = %kc, %vc0 = %vc)
			-> (tensor<4096xi32>, tensor<32x1024x4096xi8>, tensor<32x1024x4096xi8>) {
		%rms_att_w = tensor.extract_slice %rms_att_weights [%layer, 0] [1, 4096] [1, 1] : tensor<32x4096xi32> to tensor<4096xi32>
		%xb = func.call @irmsnorm(%x, %rms_att_w) : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi8>

		// Q, K, V in one GEMV against the fused weight, requantized together
		%wqkv_l = tensor.extract_slice %wqkv [%layer, 0, 0] [1, 12288, 4096] [1, 1, 1] : tensor<32x12288x4096xi8> to tensor<12288x4096xi8>
		%qkv_acc = cinm.op.gemv %wqkv_l, %xb : tensor<12288x4096xi8>, tensor<4096xi8> -> tensor<12288xi32>
		%qkv = func.call @requant_qkv(%qkv_acc) : (tensor<12288xi32>) -> tensor<12288xi8>
		%q = tensor.extract_slice %qkv [0] [4096] [1] : tensor<12288xi8> to tensor<4096xi8>
		%k = tensor.extract_slice %qkv [4096] [4096] [1] : tensor<12288xi8> to tensor<4096xi8>
		%v = tensor.extract_slice %qkv [1536] [4096] [1] : tensor<12288xi8> to tensor<4096xi8>

		// RoPE on q and k, then the caches take this position's k and v
		%q2 = func.call @irope(%q, %cos, %sin) : (tensor<4096xi8>, tensor<2048xi32>, tensor<2048xi32>) -> tensor<4096xi8>
		%k2 = func.call @irope(%k, %cos, %sin) : (tensor<4096xi8>, tensor<2048xi32>, tensor<2048xi32>) -> tensor<4096xi8>
		%kc1 = tensor.insert_slice %k2 into %kc0 [%layer, %pos_idx, 0] [1, 1, 4096] [1, 1, 1] : tensor<4096xi8> into tensor<32x1024x4096xi8>
		%vc1 = tensor.insert_slice %v into %vc0 [%layer, %pos_idx, 0] [1, 1, 4096] [1, 1, 1] : tensor<4096xi8> into tensor<32x1024x4096xi8>

		// attention over the whole cache of this layer
		%lkc = tensor.extract_slice %kc1 [%layer, 0, 0] [1, 1024, 4096] [1, 1, 1] : tensor<32x1024x4096xi8> to tensor<1024x4096xi8>
		%lvc = tensor.extract_slice %vc1 [%layer, 0, 0] [1, 1024, 4096] [1, 1, 1] : tensor<32x1024x4096xi8> to tensor<1024x4096xi8>
		%ctx = func.call @imha(%q2, %lkc, %lvc, %attn_mask)
			: (tensor<4096xi8>, tensor<1024x4096xi8>, tensor<1024x4096xi8>, tensor<1024xi32>) -> tensor<4096xi8>

		// output projection and residual. The residual is added after the
		// projection is back at the activation scale, so it is requantized
		// and then widened rather than added as an accumulator.
		%wo_l = tensor.extract_slice %wo [%layer, 0, 0] [1, 4096, 4096] [1, 1, 1] : tensor<32x4096x4096xi8> to tensor<4096x4096xi8>
		%ao_acc = cinm.op.gemv %wo_l, %ctx : tensor<4096x4096xi8>, tensor<4096xi8> -> tensor<4096xi32>
		%ao = func.call @requant_v(%ao_acc) : (tensor<4096xi32>) -> tensor<4096xi8>
		%ao_w = func.call @widen_v(%ao) : (tensor<4096xi8>) -> tensor<4096xi32>
		%x1 = cinm.op.elementwise add %x, %ao_w : tensor<4096xi32>

		// feed-forward: w2(silu(w1 x) * w3 x), 4096 -> 11008 -> 4096
		%rms_ffn_w = tensor.extract_slice %rms_ffn_weights [%layer, 0] [1, 4096] [1, 1] : tensor<32x4096xi32> to tensor<4096xi32>
		%xf = func.call @irmsnorm(%x1, %rms_ffn_w) : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi8>
		%w1_l = tensor.extract_slice %w1 [%layer, 0, 0] [1, 11008, 4096] [1, 1, 1] : tensor<32x11008x4096xi8> to tensor<11008x4096xi8>
		%w3_l = tensor.extract_slice %w3 [%layer, 0, 0] [1, 11008, 4096] [1, 1, 1] : tensor<32x11008x4096xi8> to tensor<11008x4096xi8>
		%w2_l = tensor.extract_slice %w2 [%layer, 0, 0] [1, 4096, 11008] [1, 1, 1] : tensor<32x4096x11008xi8> to tensor<4096x11008xi8>
		%h1_acc = cinm.op.gemv %w1_l, %xf : tensor<11008x4096xi8>, tensor<4096xi8> -> tensor<11008xi32>
		%h3_acc = cinm.op.gemv %w3_l, %xf : tensor<11008x4096xi8>, tensor<4096xi8> -> tensor<11008xi32>
		%h1 = func.call @requant_f(%h1_acc) : (tensor<11008xi32>) -> tensor<11008xi8>
		%h3 = func.call @requant_f(%h3_acc) : (tensor<11008xi32>) -> tensor<11008xi8>
		%hb = func.call @iswiglu(%h1, %h3) : (tensor<11008xi8>, tensor<11008xi8>) -> tensor<11008xi8>
		%f_acc = cinm.op.gemv %w2_l, %hb : tensor<4096x11008xi8>, tensor<11008xi8> -> tensor<4096xi32>
		%f = func.call @requant_v(%f_acc) : (tensor<4096xi32>) -> tensor<4096xi8>
		%f_w = func.call @widen_v(%f) : (tensor<4096xi8>) -> tensor<4096xi32>
		%x2 = cinm.op.elementwise add %x1, %f_w : tensor<4096xi32>

		scf.yield %x2, %kc1, %vc1 : tensor<4096xi32>, tensor<32x1024x4096xi8>, tensor<32x1024x4096xi8>
	} {unroll_layers}

	// ---- classifier ----------------------------------------------------------
	// The logits are the one accumulator that is not requantized: nothing
	// consumes them as an int8 operand, and argmax does not care about scale.
	%xn = func.call @irmsnorm(%xL, %rms_final_weight) : (tensor<4096xi32>, tensor<4096xi32>) -> tensor<4096xi8>
	%logits_p = cinm.op.gemv %wcls, %xn : tensor<32768x4096xi8>, tensor<4096xi8> -> tensor<32768xi32>
	%logits = tensor.extract_slice %logits_p [0] [32000] [1] : tensor<32768xi32> to tensor<32000xi32>
	return %logits : tensor<32000xi32>
}

// ===========================================================================
// Multi-head attention for one query over the layer's cache
//
// Per head: scores = K_h q_h / sqrt(128) over all 1024 cache rows (a GEMV
// against the head's slice of the key cache), the additive mask, the
// integer softmax, and the weighted sum of the head's value rows as a
// 1 x 1024 by 1024 x 128 product. Heads accumulate into one i32 context and
// are requantized together after the loop.
// ===========================================================================
func.func @imha(%q : tensor<4096xi8>, %kc : tensor<1024x4096xi8>, %vc : tensor<1024x4096xi8>,
                %mask : tensor<1024xi32>) -> tensor<4096xi8> {
	%c0 = arith.constant 0 : index
	%c128 = arith.constant 128 : index
	%c4096 = arith.constant 4096 : index
	%init_ctx = tensor.empty() : tensor<4096xi32>

	%ctx_acc = scf.for %h = %c0 to %c4096 step %c128 iter_args(%ctx_i = %init_ctx) -> (tensor<4096xi32>) {
		%qs = tensor.extract_slice %q [%h] [128] [1] : tensor<4096xi8> to tensor<128xi8>
		%kh = tensor.extract_slice %kc [0, %h] [1024, 128] [1, 1] : tensor<1024x4096xi8> to tensor<1024x128xi8>
		%vh = tensor.extract_slice %vc [0, %h] [1024, 128] [1, 1] : tensor<1024x4096xi8> to tensor<1024x128xi8>

		%sc_acc = cinm.op.gemv %kh, %qs : tensor<1024x128xi8>, tensor<128xi8> -> tensor<1024xi32>
		%sc = func.call @rescale_s(%sc_acc) : (tensor<1024xi32>) -> tensor<1024xi32>
		%sc_m = cinm.op.elementwise add %sc, %mask : tensor<1024xi32>
		%p = func.call @isoftmax(%sc_m) : (tensor<1024xi32>) -> tensor<1024xi8>

		%p2d = tensor.expand_shape %p [[0, 1]] output_shape [1, 1024] : tensor<1024xi8> into tensor<1x1024xi8>
		%ch2d = cinm.op.gemm %p2d, %vh : tensor<1x1024xi8>, tensor<1024x128xi8> -> tensor<1x128xi32>
		%ch = tensor.collapse_shape %ch2d [[0, 1]] : tensor<1x128xi32> into tensor<128xi32>
		%ctx_o = tensor.insert_slice %ch into %ctx_i [%h] [128] [1] : tensor<128xi32> into tensor<4096xi32>
		scf.yield %ctx_o : tensor<4096xi32>
	} {unroll_heads}

	%ctx = func.call @requant_v(%ctx_acc) : (tensor<4096xi32>) -> tensor<4096xi8>
	return %ctx : tensor<4096xi8>
}

// ===========================================================================
// RoPE
//
// Adjacent pairs (2j, 2j+1) of the vector are rotated by the angle of pair
// j at this position, read from the Q14 tables:
//   r0 = (q0 cos - q1 sin) >> 14      r1 = (q0 sin + q1 cos) >> 14
// The even and odd halves are strided slices, rotated as two maps, and
// written back into place. A rotation preserves the norm, so the result
// narrows back to i8 with at most a rounding excursion for the clamp.
// ===========================================================================
func.func @irope(%q : tensor<4096xi8>, %cos : tensor<2048xi32>, %sin : tensor<2048xi32>) -> tensor<4096xi8> {
	%q0 = tensor.extract_slice %q [0] [2048] [2] : tensor<4096xi8> to tensor<2048xi8>
	%q1 = tensor.extract_slice %q [1] [2048] [2] : tensor<4096xi8> to tensor<2048xi8>
	%init = tensor.empty() : tensor<2048xi8>
	%r0 = linalg.map ins(%q0, %q1, %cos, %sin : tensor<2048xi8>, tensor<2048xi8>, tensor<2048xi32>, tensor<2048xi32>) outs(%init : tensor<2048xi8>)
		(%a : i8, %b : i8, %c : i32, %s : i32, %o : i8) {
			%q14 = arith.constant 14 : i32
			%a32 = arith.extsi %a : i8 to i32
			%b32 = arith.extsi %b : i8 to i32
			%ac = arith.muli %a32, %c : i32
			%bs = arith.muli %b32, %s : i32
			%d = arith.subi %ac, %bs : i32
			%sh = arith.shrsi %d, %q14 : i32
			%r = func.call @narrow_i8(%sh) : (i32) -> i8
			linalg.yield %r : i8
		}
	%r1 = linalg.map ins(%q0, %q1, %cos, %sin : tensor<2048xi8>, tensor<2048xi8>, tensor<2048xi32>, tensor<2048xi32>) outs(%init : tensor<2048xi8>)
		(%a : i8, %b : i8, %c : i32, %s : i32, %o : i8) {
			%q14 = arith.constant 14 : i32
			%a32 = arith.extsi %a : i8 to i32
			%b32 = arith.extsi %b : i8 to i32
			%as = arith.muli %a32, %s : i32
			%bc = arith.muli %b32, %c : i32
			%d = arith.addi %as, %bc : i32
			%sh = arith.shrsi %d, %q14 : i32
			%r = func.call @narrow_i8(%sh) : (i32) -> i8
			linalg.yield %r : i8
		}
	%o0 = tensor.insert_slice %r0 into %q [0] [2048] [2] : tensor<2048xi8> into tensor<4096xi8>
	%o1 = tensor.insert_slice %r1 into %o0 [1] [2048] [2] : tensor<2048xi8> into tensor<4096xi8>
	return %o1 : tensor<4096xi8>
}

// ===========================================================================
// I-RMSNorm  (the sqrt of I-BERT 3.4, without the mean)
//
//   ss    = sum((x >> 3)^2)            -- pre-shifted so 4096 squares fit i32
//   sigma = int_sqrt(ss / H)           -- ~ rms(x) / 8, Newton, no float
//   y     = x * 2^7 / sigma            -- x / rms(x), held in Q10
//   out   = (y * w) >> 10              -- w is Q10; then narrowed to i8
//
// Takes the i32 residual stream and produces the i8 the next GEMV consumes,
// so it is both the normalization and the requantization at that point.
// ===========================================================================
func.func @irmsnorm(%x : tensor<4096xi32>, %w : tensor<4096xi32>) -> tensor<4096xi8> {
	%init = tensor.empty() : tensor<4096xi32>
	%init_o = tensor.empty() : tensor<4096xi8>
	%pre = arith.constant 3 : i32
	%h = arith.constant 4096 : i32
	%one = arith.constant 1 : i32
	%two = arith.constant 2 : i32
	%up = arith.constant 7 : i32
	%down = arith.constant 10 : i32
	%i0 = arith.constant 0 : index
	%i1 = arith.constant 1 : index
	%i24 = arith.constant 24 : index

	%xs = linalg.map ins(%x : tensor<4096xi32>) outs(%init : tensor<4096xi32>)
		(%v : i32, %o : i32) {
			%r = arith.shrsi %v, %pre : i32
			linalg.yield %r : i32
		}
	%sq = cinm.op.elementwise mul %xs, %xs : tensor<4096xi32>
	%ss = cinm.op.reduce add (%sq) : tensor<4096xi32> -> i32
	%var = arith.divsi %ss, %h : i32

	// Newton's integer sqrt. y_{k+1} = (y_k + n/y_k) / 2 starting from n+1,
	// which keeps y >= 1 so the divide is always defined and converges from
	// above to floor(sqrt(n)). 24 steps covers the whole i32 range: each
	// early step roughly halves y, and the tail is quadratic.
	%n1 = arith.addi %var, %one : i32
	%sigma0 = scf.for %k = %i0 to %i24 step %i1 iter_args(%y = %n1) -> (i32) {
		%qd = arith.divsi %n1, %y : i32
		%sm = arith.addi %y, %qd : i32
		%hf = arith.divsi %sm, %two : i32
		scf.yield %hf : i32
	}
	// A zero vector has sigma 0; dividing by 1 then maps it to zero.
	%sigma = arith.maxsi %sigma0, %one : i32

	%out = linalg.map ins(%x, %w : tensor<4096xi32>, tensor<4096xi32>) outs(%init_o : tensor<4096xi8>)
		(%v : i32, %g : i32, %o : i8) {
			%vu = arith.shli %v, %up : i32
			%y = arith.divsi %vu, %sigma : i32
			%yg = arith.muli %y, %g : i32
			%ygs = arith.shrsi %yg, %down : i32
			%r = func.call @narrow_i8(%ygs) : (i32) -> i8
			linalg.yield %r : i8
		}
	return %out : tensor<4096xi8>
}

// ===========================================================================
// I-Softmax  (I-BERT 3.3), over one 1024-entry score row
//
// exp is never evaluated: with x~ = x - max(x) <= 0, write x~ = -z*ln2 + p
// for integer z and p in [-ln2, 0], so exp(x~) = 2^-z * exp(p), and exp(p)
// over that one interval is the second-order polynomial L below. The 2^-z
// is an arithmetic shift.
//
//   L(p) = 0.3585 (p + 1.353)^2 + 0.344      evaluated in Q14
//
// Produces the i8 probabilities the value product multiplies.
// ===========================================================================
func.func @isoftmax(%x : tensor<1024xi32>) -> tensor<1024xi8> {
	%init = tensor.empty() : tensor<1024xi32>
	%init_o = tensor.empty() : tensor<1024xi8>

	%mx = cinm.op.reduce maxsi (%x) : tensor<1024xi32> -> i32
	%mx_v = tensor.splat %mx : tensor<1024xi32>
	%xt = cinm.op.elementwise sub %x, %mx_v : tensor<1024xi32>

	%e = linalg.map ins(%xt : tensor<1024xi32>) outs(%init : tensor<1024xi32>)
		(%v : i32, %o : i32) {
			// ln2 in the Q4 activation scale: round(0.693147 * 16)
			%ln2 = arith.constant 11 : i32
			// Q14 polynomial coefficients: 0.3585, 1.353, 0.344
			%pa = arith.constant 5874 : i32
			%pb = arith.constant 22168 : i32
			%pc = arith.constant 5636 : i32
			%q14 = arith.constant 14 : i32
			// Q4 -> Q14 is a shift of 10
			%up = arith.constant 10 : i32
			// Past 30 halvings exp is 0 in Q14, so the input is floored at
			// -30*ln2 rather than the shift count being clamped after the
			// fact. It has to be the input: a masked position arrives here
			// around -1e4, and clamping only z would leave p that far
			// outside [-ln2, 0], overflowing the square below.
			%floor = arith.constant -330 : i32

			%zero = arith.constant 0 : i32
			%vc = arith.maxsi %v, %floor : i32
			%negv = arith.subi %zero, %vc : i32
			%z = arith.divsi %negv, %ln2 : i32
			%zl = arith.muli %z, %ln2 : i32
			%p = arith.addi %vc, %zl : i32

			%p14 = arith.shli %p, %up : i32
			%t = arith.addi %p14, %pb : i32
			%t2 = arith.muli %t, %t : i32
			%t2s = arith.shrsi %t2, %q14 : i32
			%at = arith.muli %pa, %t2s : i32
			%ats = arith.shrsi %at, %q14 : i32
			%l = arith.addi %ats, %pc : i32
			%ex = arith.shrsi %l, %z : i32
			linalg.yield %ex : i32
		}

	// Normalize into the int8 range the value product expects: p = 127 * e
	// / sum. e is at most 2^14 so the numerator stays inside i32, and the
	// row's own maximum contributes L(0) = 0.344 in Q14, so the sum is never
	// zero.
	%s = cinm.op.reduce add (%e) : tensor<1024xi32> -> i32
	%s_v = tensor.splat %s : tensor<1024xi32>
	%q127 = arith.constant 127 : i32
	%q127_v = tensor.splat %q127 : tensor<1024xi32>
	%en = cinm.op.elementwise mul %e, %q127_v : tensor<1024xi32>
	%p32 = cinm.op.elementwise div %en, %s_v : tensor<1024xi32>

	%p8 = linalg.map ins(%p32 : tensor<1024xi32>) outs(%init_o : tensor<1024xi8>)
		(%v : i32, %o : i8) {
			%r = func.call @narrow_i8(%v) : (i32) -> i8
			linalg.yield %r : i8
		}
	return %p8 : tensor<1024xi8>
}

// ===========================================================================
// SwiGLU: silu(h1) * h3, with sigmoid from the exp-by-shift above
//
//   e        = exp(-|x|)               in Q14, exponent as in @isoftmax
//   sigmoid  = x >= 0 ? 2^14 / (1 + e) : e / (1 + e)      in Q14
//   silu     = (x * sigmoid) >> 14     back in Q4
//   out      = (silu * h3) >> 4        Q4 * Q4 -> Q4, then narrowed
//
// Both inputs are the requantized i8 activations of the w1 and w3 GEMVs.
// ===========================================================================
func.func @iswiglu(%h1 : tensor<11008xi8>, %h3 : tensor<11008xi8>) -> tensor<11008xi8> {
	%init_o = tensor.empty() : tensor<11008xi8>
	%r = linalg.map ins(%h1, %h3 : tensor<11008xi8>, tensor<11008xi8>) outs(%init_o : tensor<11008xi8>)
		(%a : i8, %g : i8, %o : i8) {
			%ln2 = arith.constant 11 : i32
			%pa = arith.constant 5874 : i32
			%pb = arith.constant 22168 : i32
			%pc = arith.constant 5636 : i32
			%q14 = arith.constant 14 : i32
			%one14 = arith.constant 16384 : i32
			%up = arith.constant 10 : i32
			%floor = arith.constant -330 : i32
			%zero = arith.constant 0 : i32
			%four = arith.constant 4 : i32

			%x = arith.extsi %a : i8 to i32
			%nx = arith.subi %zero, %x : i32
			%ax = arith.maxsi %x, %nx : i32
			%neg = arith.subi %zero, %ax : i32

			// exp(-|x|) in Q14
			%vc = arith.maxsi %neg, %floor : i32
			%negv = arith.subi %zero, %vc : i32
			%z = arith.divsi %negv, %ln2 : i32
			%zl = arith.muli %z, %ln2 : i32
			%p = arith.addi %vc, %zl : i32
			%p14 = arith.shli %p, %up : i32
			%t = arith.addi %p14, %pb : i32
			%t2 = arith.muli %t, %t : i32
			%t2s = arith.shrsi %t2, %q14 : i32
			%at = arith.muli %pa, %t2s : i32
			%ats = arith.shrsi %at, %q14 : i32
			%l = arith.addi %ats, %pc : i32
			%e = arith.shrsi %l, %z : i32

			// sigmoid(x) in Q14, from e = exp(-|x|)
			%den = arith.addi %one14, %e : i32
			%nonneg = arith.cmpi sge, %x, %zero : i32
			%num = arith.select %nonneg, %one14, %e : i32
			%num14 = arith.shli %num, %q14 : i32
			%sig = arith.divsi %num14, %den : i32

			%sx = arith.muli %x, %sig : i32
			%silu = arith.shrsi %sx, %q14 : i32
			%g32 = arith.extsi %g : i8 to i32
			%prod = arith.muli %silu, %g32 : i32
			%out = arith.shrsi %prod, %four : i32
			%r = func.call @narrow_i8(%out) : (i32) -> i8
			linalg.yield %r : i8
		}
	return %r : tensor<11008xi8>
}

// ===========================================================================
// Crossing between the operand and accumulator domains
//
// Every GEMV widens on the way in and every requantization narrows on the
// way out, so these are the only places an activation changes width.
// ===========================================================================

// Saturating narrow. Clamping rather than wrapping is what keeps a single
// outlier from flipping sign.
func.func private @narrow_i8(%v : i32) -> i8 {
	%lo = arith.constant -128 : i32
	%hi = arith.constant 127 : i32
	%a = arith.maxsi %v, %lo : i32
	%b = arith.minsi %a, %hi : i32
	%r = arith.trunci %b : i32 to i8
	return %r : i8
}

// (acc * M) >> SH, the multiply-shift that returns an accumulator to the
// int8 range the next GEMV expects. M and SH are the placeholders the
// header describes. The product genuinely needs 64 bits: an 11008-deep int8
// accumulation reaches ~1.8e8, and M is ~2e4.
func.func private @requant_scalar(%v : i32) -> i8 {
	%m = arith.constant 21475 : i64
	%sh = arith.constant 31 : i64
	%v64 = arith.extsi %v : i32 to i64
	%p = arith.muli %v64, %m : i64
	%q = arith.shrsi %p, %sh : i64
	%t = arith.trunci %q : i64 to i32
	%r = func.call @narrow_i8(%t) : (i32) -> i8
	return %r : i8
}

func.func @requant_v(%x : tensor<4096xi32>) -> tensor<4096xi8> {
	%init = tensor.empty() : tensor<4096xi8>
	%r = linalg.map ins(%x : tensor<4096xi32>) outs(%init : tensor<4096xi8>)
		(%v : i32, %o : i8) {
			%y = func.call @requant_scalar(%v) : (i32) -> i8
			linalg.yield %y : i8
		}
	return %r : tensor<4096xi8>
}

func.func @requant_qkv(%x : tensor<12288xi32>) -> tensor<12288xi8> {
	%init = tensor.empty() : tensor<12288xi8>
	%r = linalg.map ins(%x : tensor<12288xi32>) outs(%init : tensor<12288xi8>)
		(%v : i32, %o : i8) {
			%y = func.call @requant_scalar(%v) : (i32) -> i8
			linalg.yield %y : i8
		}
	return %r : tensor<12288xi8>
}

func.func @requant_f(%x : tensor<11008xi32>) -> tensor<11008xi8> {
	%init = tensor.empty() : tensor<11008xi8>
	%r = linalg.map ins(%x : tensor<11008xi32>) outs(%init : tensor<11008xi8>)
		(%v : i32, %o : i8) {
			%y = func.call @requant_scalar(%v) : (i32) -> i8
			linalg.yield %y : i8
		}
	return %r : tensor<11008xi8>
}

func.func @widen_v(%x : tensor<4096xi8>) -> tensor<4096xi32> {
	%init = tensor.empty() : tensor<4096xi32>
	%r = linalg.map ins(%x : tensor<4096xi8>) outs(%init : tensor<4096xi32>)
		(%v : i8, %o : i32) {
			%y = arith.extsi %v : i8 to i32
			linalg.yield %y : i32
		}
	return %r : tensor<4096xi32>
}

// The attention scores keep the accumulator width but change scale: this
// folds in 1/sqrt(128) ~ 1/11, taken as 1/8 -- a placeholder like the
// multiplier itself -- so it shifts three places further than
// @requant_scalar and does not narrow.
func.func @rescale_s(%x : tensor<1024xi32>) -> tensor<1024xi32> {
	%init = tensor.empty() : tensor<1024xi32>
	%m = arith.constant 21475 : i64
	%sh = arith.constant 34 : i64
	%r = linalg.map ins(%x : tensor<1024xi32>) outs(%init : tensor<1024xi32>)
		(%v : i32, %o : i32) {
			%v64 = arith.extsi %v : i32 to i64
			%p = arith.muli %v64, %m : i64
			%q = arith.shrsi %p, %sh : i64
			%y = arith.trunci %q : i64 to i32
			linalg.yield %y : i32
		}
	return %r : tensor<1024xi32>
}
