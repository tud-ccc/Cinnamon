// RoBERTa-base, integer-only inference, for the whole-program experiment.
//
// Model: roberta-base (Liu et al. 2019) -- the checkpoint I-BERT quantizes.
//   hidden       (H)  768
//   layers       (L)  12
//   heads        (A)  12    -> head dim 64
//   intermediate (I)  3072
//   vocab        (V)  50265
//   positions    (P)  514   (RoBERTa offsets positions by padding_idx + 1)
//   sequence     (S)  128   (the GLUE/SQuAD evaluation length)
// Post-LayerNorm, GELU, bidirectional: padding is the only masking, and it
// arrives as the additive %attn_mask, so there is no causal triangle.
//
// Numerics: integer-only, after I-BERT (Kim et al., ICML 2021). Nothing
// leaves the integer domain between the embedding lookup and the logits:
//   GELU      the erf polynomial of I-BERT  3.2, via @igelu
//   Softmax   the exp-by-shift of I-BERT    3.3, via @isoftmax
//   LayerNorm the Newton integer sqrt of    3.4, via @ilayernorm
// tanh in the pooler reuses the erf polynomial through the standard
// tanh(x) ~= erf(sqrt(pi)/2 * x) identity (max error ~0.018), so the head
// needs no second approximation.
//
// Precision: W8A8. Weights and activations are i8; every matmul is
// i8 x i8 -> i32, biases are i32 because they are added in the accumulator
// domain, and each accumulator is requantized back to i8 by a multiply-shift
// with saturation (@requant_*). LayerNorm parameters are i32 for the same
// reason. Two activations are only ever added after both are widened to i32
// (@widen_*): a sum of two int8 activations does not fit in an int8.
//
// This is the layout an int8 checkpoint has, and the reason it matters here
// is the transfer: the encoder is 12 * (4*768*768 + 2*768*3072) = 85 MB of
// i8 weights against 340 MB of i32, plus 39 MB of embeddings against 156 MB,
// so 124 MB total -- what a real roberta-base occupies. Every weight scatter
// and every MRAM read moves a quarter of what the i32 spelling would, and
// the tile-size space widens to match because a leaf holds four times the
// operand elements.
//
// Fixed point: quantized activations are Q4 (a stored q means q / 16), and
// the nonlinearity polynomials evaluate in Q14. The requantization
// multiplier below is a placeholder consistent with that convention, not a
// calibration of a trained checkpoint: this program is faithful in shape,
// op sequence and data movement, which is what the evaluation measures, and
// is not expected to reproduce GLUE accuracy until real scales are loaded.

#upmem = #upmem.platform<type = v1A, dpus = 2560, tasklets = 24>

func.func @roberta_base(
	// per-inference inputs. Token and position ids are i32, as every runtime
	// hands them over; they become `index` only at the two gather sites.
	%input_ids : tensor<128xi32>,
	%position_ids : tensor<128xi32>,
	// additive padding mask, applied to the i32 scores before the softmax
	%attn_mask : tensor<128x128xi32>,
	// embeddings
	%word_emb : tensor<50265x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%pos_emb : tensor<514x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%tok_type_emb : tensor<768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i) -> (i)>},
	%emb_ln_g : tensor<768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i) -> (i)>},
	%emb_ln_b : tensor<768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i) -> (i)>},
	// attention: weights are stored [layer][in][out], the layout a serving
	// deployment packs once, so no transpose is needed at inference
	%wq : tensor<12x768x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wk : tensor<12x768x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wv : tensor<12x768x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%wo : tensor<12x768x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%bq : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%bk : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%bv : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%bo : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%attn_ln_g : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%attn_ln_b : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	// feed-forward
	%w1 : tensor<12x768x3072xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%b1 : tensor<12x3072xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%w2 : tensor<12x3072x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j,k) -> (i,j,k)>},
	%b2 : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%ffn_ln_g : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%ffn_ln_b : tensor<12x768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	// GLUE classification head: pooler over <s>, then the 2-way projection
	%pool_w : tensor<768x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%pool_b : tensor<768xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i) -> (i)>},
	%cls_w : tensor<2x768xi8> {cinm.static, bufferization.buffer_layout = affine_map<(i,j) -> (i,j)>},
	%cls_b : tensor<2xi32> {cinm.static, bufferization.buffer_layout = affine_map<(i) -> (i)>}
) -> tensor<2xi32> attributes {cinm.available_platforms = [#upmem]} {
	%c0 = arith.constant 0 : index
	%c1 = arith.constant 1 : index
	%c12 = arith.constant 12 : index
	%c64 = arith.constant 64 : index
	%c128 = arith.constant 128 : index
	%c768 = arith.constant 768 : index

	// ---- embeddings -------------------------------------------------------
	// The two gathers move rows, they do not compute: keeping the cinm ops
	// out of the loop bodies is what lets --cinm-complete-compute-graph see
	// one connected graph after the layer loop is unrolled.
	%emb_init = tensor.empty() : tensor<128x768xi8>

	%we_g = scf.for %t = %c0 to %c128 step %c1 iter_args(%acc = %emb_init) -> (tensor<128x768xi8>) {
		%id32 = tensor.extract %input_ids[%t] : tensor<128xi32>
		%id = arith.index_cast %id32 : i32 to index
		%row = tensor.extract_slice %word_emb [%id, 0] [1, 768] [1, 1] : tensor<50265x768xi8> to tensor<768xi8>
		%acc2 = tensor.insert_slice %row into %acc [%t, 0] [1, 768] [1, 1] : tensor<768xi8> into tensor<128x768xi8>
		scf.yield %acc2 : tensor<128x768xi8>
	}

	%pe_g = scf.for %t = %c0 to %c128 step %c1 iter_args(%acc = %emb_init) -> (tensor<128x768xi8>) {
		%pid32 = tensor.extract %position_ids[%t] : tensor<128xi32>
		%pid = arith.index_cast %pid32 : i32 to index
		%row = tensor.extract_slice %pos_emb [%pid, 0] [1, 768] [1, 1] : tensor<514x768xi8> to tensor<768xi8>
		%acc2 = tensor.insert_slice %row into %acc [%t, 0] [1, 768] [1, 1] : tensor<768xi8> into tensor<128x768xi8>
		scf.yield %acc2 : tensor<128x768xi8>
	}

	// RoBERTa has a single token type, so its embedding is one row added to
	// every position rather than a second gather. Three int8 rows summed
	// overflow an int8, so the sum is taken in i32 and LayerNorm brings it
	// back down.
	%tt_b = linalg.broadcast ins(%tok_type_emb : tensor<768xi8>) outs(%emb_init : tensor<128x768xi8>) dimensions = [0]
	%we_w = func.call @widen_h(%we_g) : (tensor<128x768xi8>) -> tensor<128x768xi32>
	%pe_w = func.call @widen_h(%pe_g) : (tensor<128x768xi8>) -> tensor<128x768xi32>
	%tt_w = func.call @widen_h(%tt_b) : (tensor<128x768xi8>) -> tensor<128x768xi32>
	%e0 = cinm.op.elementwise add %we_w, %pe_w : tensor<128x768xi32>
	%e1 = cinm.op.elementwise add %e0, %tt_w : tensor<128x768xi32>
	%x0 = func.call @ilayernorm(%e1, %emb_ln_g, %emb_ln_b)
		: (tensor<128x768xi32>, tensor<768xi32>, tensor<768xi32>) -> tensor<128x768xi8>

	// ---- encoder ----------------------------------------------------------
	%xL = scf.for %layer = %c0 to %c12 step %c1 iter_args(%x = %x0) -> (tensor<128x768xi8>) {
		%wq_l = tensor.extract_slice %wq [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xi8> to tensor<768x768xi8>
		%wk_l = tensor.extract_slice %wk [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xi8> to tensor<768x768xi8>
		%wv_l = tensor.extract_slice %wv [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xi8> to tensor<768x768xi8>
		%wo_l = tensor.extract_slice %wo [%layer, 0, 0] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xi8> to tensor<768x768xi8>
		%bq_l = tensor.extract_slice %bq [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%bk_l = tensor.extract_slice %bk [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%bv_l = tensor.extract_slice %bv [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%bo_l = tensor.extract_slice %bo [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>

		%init_acc = tensor.empty() : tensor<128x768xi32>
		%bq_b = linalg.broadcast ins(%bq_l : tensor<768xi32>) outs(%init_acc : tensor<128x768xi32>) dimensions = [0]
		%bk_b = linalg.broadcast ins(%bk_l : tensor<768xi32>) outs(%init_acc : tensor<128x768xi32>) dimensions = [0]
		%bv_b = linalg.broadcast ins(%bv_l : tensor<768xi32>) outs(%init_acc : tensor<128x768xi32>) dimensions = [0]
		%bo_b = linalg.broadcast ins(%bo_l : tensor<768xi32>) outs(%init_acc : tensor<128x768xi32>) dimensions = [0]

		// Q, K, V projections. Each accumulates 768 int8 products into i32,
		// adds the i32 bias in that same domain, and is requantized back to
		// the int8 range the next matmul expects.
		%q_acc = cinm.op.gemm %x, %wq_l plus %bq_b : tensor<128x768xi8>, tensor<768x768xi8> plus tensor<128x768xi32> -> tensor<128x768xi32>
		%k_acc = cinm.op.gemm %x, %wk_l plus %bk_b : tensor<128x768xi8>, tensor<768x768xi8> plus tensor<128x768xi32> -> tensor<128x768xi32>
		%v_acc = cinm.op.gemm %x, %wv_l plus %bv_b : tensor<128x768xi8>, tensor<768x768xi8> plus tensor<128x768xi32> -> tensor<128x768xi32>
		%q = func.call @requant_h(%q_acc) : (tensor<128x768xi32>) -> tensor<128x768xi8>
		%k = func.call @requant_h(%k_acc) : (tensor<128x768xi32>) -> tensor<128x768xi8>
		%v = func.call @requant_h(%v_acc) : (tensor<128x768xi32>) -> tensor<128x768xi8>

		// One transpose of the whole K rather than twelve of its head slices:
		// the head's K^T is then a slice of rows.
		%init_kt = tensor.empty() : tensor<768x128xi8>
		%kT = linalg.transpose ins(%k : tensor<128x768xi8>) outs(%init_kt : tensor<768x128xi8>) permutation = [1, 0]

		// Heads accumulate into one i32 context and are requantized together
		// after the loop, so the concatenation costs one pass rather than
		// twelve.
		%ctx_acc = scf.for %h = %c0 to %c768 step %c64 iter_args(%ctx_i = %init_acc) -> (tensor<128x768xi32>) {
			%qh = tensor.extract_slice %q [0, %h] [128, 64] [1, 1] : tensor<128x768xi8> to tensor<128x64xi8>
			%khT = tensor.extract_slice %kT [%h, 0] [64, 128] [1, 1] : tensor<768x128xi8> to tensor<64x128xi8>
			%vh = tensor.extract_slice %v [0, %h] [128, 64] [1, 1] : tensor<128x768xi8> to tensor<128x64xi8>

			// scores = Q_h K_h^T / sqrt(64), the 1/8 folded into @rescale_s.
			// They stay i32: the mask is additive in that domain, and the
			// pre-max range of a score does not fit in an int8 anyway.
			%sc_acc = cinm.op.gemm %qh, %khT : tensor<128x64xi8>, tensor<64x128xi8> -> tensor<128x128xi32>
			%sc = func.call @rescale_s(%sc_acc) : (tensor<128x128xi32>) -> tensor<128x128xi32>
			%sc_m = cinm.op.elementwise add %sc, %attn_mask : tensor<128x128xi32>
			%p = func.call @isoftmax(%sc_m) : (tensor<128x128xi32>) -> tensor<128x128xi8>

			%ch = cinm.op.gemm %p, %vh : tensor<128x128xi8>, tensor<128x64xi8> -> tensor<128x64xi32>
			%ctx_o = tensor.insert_slice %ch into %ctx_i [0, %h] [128, 64] [1, 1] : tensor<128x64xi32> into tensor<128x768xi32>
			scf.yield %ctx_o : tensor<128x768xi32>
		} {unroll_heads}

		// attention output projection, residual, post-LayerNorm. The residual
		// is added after both sides are back at the activation scale, which
		// is why the projection is requantized before the widening rather
		// than after the add.
		%ctx = func.call @requant_h(%ctx_acc) : (tensor<128x768xi32>) -> tensor<128x768xi8>
		%ao_acc = cinm.op.gemm %ctx, %wo_l plus %bo_b : tensor<128x768xi8>, tensor<768x768xi8> plus tensor<128x768xi32> -> tensor<128x768xi32>
		%ao = func.call @requant_h(%ao_acc) : (tensor<128x768xi32>) -> tensor<128x768xi8>
		%ao_w = func.call @widen_h(%ao) : (tensor<128x768xi8>) -> tensor<128x768xi32>
		%x_w = func.call @widen_h(%x) : (tensor<128x768xi8>) -> tensor<128x768xi32>
		%ar = cinm.op.elementwise add %ao_w, %x_w : tensor<128x768xi32>
		%aln_g = tensor.extract_slice %attn_ln_g [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%aln_b = tensor.extract_slice %attn_ln_b [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%xa = func.call @ilayernorm(%ar, %aln_g, %aln_b)
			: (tensor<128x768xi32>, tensor<768xi32>, tensor<768xi32>) -> tensor<128x768xi8>

		// feed-forward: 768 -> 3072 -> 768, GELU between
		%w1_l = tensor.extract_slice %w1 [%layer, 0, 0] [1, 768, 3072] [1, 1, 1] : tensor<12x768x3072xi8> to tensor<768x3072xi8>
		%w2_l = tensor.extract_slice %w2 [%layer, 0, 0] [1, 3072, 768] [1, 1, 1] : tensor<12x3072x768xi8> to tensor<3072x768xi8>
		%b1_l = tensor.extract_slice %b1 [%layer, 0] [1, 3072] [1, 1] : tensor<12x3072xi32> to tensor<3072xi32>
		%b2_l = tensor.extract_slice %b2 [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%init_i = tensor.empty() : tensor<128x3072xi32>
		%b1_b = linalg.broadcast ins(%b1_l : tensor<3072xi32>) outs(%init_i : tensor<128x3072xi32>) dimensions = [0]
		%b2_b = linalg.broadcast ins(%b2_l : tensor<768xi32>) outs(%init_acc : tensor<128x768xi32>) dimensions = [0]

		%f1_acc = cinm.op.gemm %xa, %w1_l plus %b1_b : tensor<128x768xi8>, tensor<768x3072xi8> plus tensor<128x3072xi32> -> tensor<128x3072xi32>
		%f1 = func.call @requant_i(%f1_acc) : (tensor<128x3072xi32>) -> tensor<128x3072xi8>
		%g1 = func.call @igelu(%f1) : (tensor<128x3072xi8>) -> tensor<128x3072xi8>
		%f2_acc = cinm.op.gemm %g1, %w2_l plus %b2_b : tensor<128x3072xi8>, tensor<3072x768xi8> plus tensor<128x768xi32> -> tensor<128x768xi32>
		%f2 = func.call @requant_h(%f2_acc) : (tensor<128x768xi32>) -> tensor<128x768xi8>
		%f2_w = func.call @widen_h(%f2) : (tensor<128x768xi8>) -> tensor<128x768xi32>
		%xa_w = func.call @widen_h(%xa) : (tensor<128x768xi8>) -> tensor<128x768xi32>
		%fr = cinm.op.elementwise add %f2_w, %xa_w : tensor<128x768xi32>
		%fln_g = tensor.extract_slice %ffn_ln_g [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%fln_b = tensor.extract_slice %ffn_ln_b [%layer, 0] [1, 768] [1, 1] : tensor<12x768xi32> to tensor<768xi32>
		%xf = func.call @ilayernorm(%fr, %fln_g, %fln_b)
			: (tensor<128x768xi32>, tensor<768xi32>, tensor<768xi32>) -> tensor<128x768xi8>

		scf.yield %xf : tensor<128x768xi8>
	} {unroll_layers}

	// ---- GLUE classification head over the <s> token -----------------------
	// The logits are the one accumulator that is not requantized: nothing
	// consumes them as an int8 operand, and argmax does not care about scale.
	%cls_tok = tensor.extract_slice %xL [0, 0] [1, 768] [1, 1] : tensor<128x768xi8> to tensor<768xi8>
	%pool_acc = cinm.op.gemv %pool_w, %cls_tok plus %pool_b : tensor<768x768xi8>, tensor<768xi8> plus tensor<768xi32> -> tensor<768xi32>
	%pool_q = func.call @requant_v(%pool_acc) : (tensor<768xi32>) -> tensor<768xi8>
	%pooled = func.call @itanh(%pool_q) : (tensor<768xi8>) -> tensor<768xi8>
	%logits = cinm.op.gemv %cls_w, %pooled plus %cls_b : tensor<2x768xi8>, tensor<768xi8> plus tensor<2xi32> -> tensor<2xi32>
	return %logits : tensor<2xi32>
}

// ===========================================================================
// Crossing between the operand and accumulator domains
//
// Every matmul widens on the way in and every requantization narrows on the
// way out, so these two are the only places an activation changes width.
// ===========================================================================

// Saturating narrow. Clamping rather than wrapping is what keeps a single
// outlier from flipping sign and poisoning the next matmul.
func.func private @narrow_i8(%v : i32) -> i8 {
	%lo = arith.constant -128 : i32
	%hi = arith.constant 127 : i32
	%a = arith.maxsi %v, %lo : i32
	%b = arith.minsi %a, %hi : i32
	%r = arith.trunci %b : i32 to i8
	return %r : i8
}

// (acc * M) >> SH, the multiply-shift that returns an accumulator to the
// int8 range the next matmul expects. M and SH are the placeholders the
// header describes. The product genuinely needs 64 bits: a 768-deep int8
// accumulation reaches ~1.2e7, and M is ~2e4.
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

func.func @requant_h(%x : tensor<128x768xi32>) -> tensor<128x768xi8> {
	%init = tensor.empty() : tensor<128x768xi8>
	%r = linalg.map ins(%x : tensor<128x768xi32>) outs(%init : tensor<128x768xi8>)
		(%v : i32, %o : i8) {
			%y = func.call @requant_scalar(%v) : (i32) -> i8
			linalg.yield %y : i8
		}
	return %r : tensor<128x768xi8>
}

func.func @requant_i(%x : tensor<128x3072xi32>) -> tensor<128x3072xi8> {
	%init = tensor.empty() : tensor<128x3072xi8>
	%r = linalg.map ins(%x : tensor<128x3072xi32>) outs(%init : tensor<128x3072xi8>)
		(%v : i32, %o : i8) {
			%y = func.call @requant_scalar(%v) : (i32) -> i8
			linalg.yield %y : i8
		}
	return %r : tensor<128x3072xi8>
}

func.func @requant_v(%x : tensor<768xi32>) -> tensor<768xi8> {
	%init = tensor.empty() : tensor<768xi8>
	%r = linalg.map ins(%x : tensor<768xi32>) outs(%init : tensor<768xi8>)
		(%v : i32, %o : i8) {
			%y = func.call @requant_scalar(%v) : (i32) -> i8
			linalg.yield %y : i8
		}
	return %r : tensor<768xi8>
}

func.func @widen_h(%x : tensor<128x768xi8>) -> tensor<128x768xi32> {
	%init = tensor.empty() : tensor<128x768xi32>
	%r = linalg.map ins(%x : tensor<128x768xi8>) outs(%init : tensor<128x768xi32>)
		(%v : i8, %o : i32) {
			%y = arith.extsi %v : i8 to i32
			linalg.yield %y : i32
		}
	return %r : tensor<128x768xi32>
}

// The attention scores keep the accumulator width but change scale: this
// folds in 1/sqrt(64) = 1/8, so it shifts three places further than
// @requant_scalar and does not narrow.
func.func @rescale_s(%x : tensor<128x128xi32>) -> tensor<128x128xi32> {
	%init = tensor.empty() : tensor<128x128xi32>
	%m = arith.constant 21475 : i64
	%sh = arith.constant 34 : i64
	%r = linalg.map ins(%x : tensor<128x128xi32>) outs(%init : tensor<128x128xi32>)
		(%v : i32, %o : i32) {
			%v64 = arith.extsi %v : i32 to i64
			%p = arith.muli %v64, %m : i64
			%q = arith.shrsi %p, %sh : i64
			%y = arith.trunci %q : i64 to i32
			linalg.yield %y : i32
		}
	return %r : tensor<128x128xi32>
}

// ===========================================================================
// I-LayerNorm  (I-BERT 3.4)
//
//   mu    = sum(x) / H
//   var   = sum((x - mu)^2) / H
//   sigma = int_sqrt(var)              -- Newton, no float, no rsqrt
//   y     = (x - mu) * 2^10 / sigma    -- normalized, held in Q10
//   out   = (y * gamma) >> 10 + beta   -- then narrowed to the i8 activation
//
// Takes the i32 residual sum and produces the i8 the next matmul consumes,
// so it is both the normalization and the requantization at that point.
// Everything but the sqrt is a cinm op over the full 128x768 tile; the sqrt
// is 128 scalars, one per token row.
// ===========================================================================

func.func @ilayernorm(%x : tensor<128x768xi32>, %g : tensor<768xi32>, %b : tensor<768xi32>) -> tensor<128x768xi8> {
	%ch = arith.constant 768 : i32
	%scale = arith.constant 1024 : i32
	%unscale = arith.constant 1024 : i32
	%init_h = tensor.empty() : tensor<128x768xi32>
	%init_r = tensor.empty() : tensor<128xi32>
	%init_o = tensor.empty() : tensor<128x768xi8>

	%hv = tensor.splat %ch : tensor<128xi32>
	%sum = cinm.op.reduce add (%x) : tensor<128x768xi32> -> tensor<128xi32>
	%mu = cinm.op.elementwise div %sum, %hv : tensor<128xi32>
	%mu_b = linalg.broadcast ins(%mu : tensor<128xi32>) outs(%init_h : tensor<128x768xi32>) dimensions = [1]
	%d = cinm.op.elementwise sub %x, %mu_b : tensor<128x768xi32>

	%dd = cinm.op.elementwise mul %d, %d : tensor<128x768xi32>
	%vsum = cinm.op.reduce add (%dd) : tensor<128x768xi32> -> tensor<128xi32>
	%var = cinm.op.elementwise div %vsum, %hv : tensor<128xi32>

	// Newton's integer sqrt. y_{k+1} = (y_k + n/y_k) / 2 starting from n+1,
	// which keeps y >= 1 so the divide is always defined and converges from
	// above to floor(sqrt(n)). 24 steps covers the whole i32 range: each
	// early step roughly halves y, and the tail is quadratic.
	%sigma = linalg.map ins(%var : tensor<128xi32>) outs(%init_r : tensor<128xi32>)
		(%n : i32, %o : i32) {
			%one = arith.constant 1 : i32
			%two = arith.constant 2 : i32
			%i0 = arith.constant 0 : index
			%i1 = arith.constant 1 : index
			%i24 = arith.constant 24 : index
			%n1 = arith.addi %n, %one : i32
			%s = scf.for %k = %i0 to %i24 step %i1 iter_args(%y = %n1) -> (i32) {
				%qd = arith.divsi %n1, %y : i32
				%sm = arith.addi %y, %qd : i32
				%hf = arith.divsi %sm, %two : i32
				scf.yield %hf : i32
			}
			linalg.yield %s : i32
		}

	%sigma_b = linalg.broadcast ins(%sigma : tensor<128xi32>) outs(%init_h : tensor<128x768xi32>) dimensions = [1]
	%scale_v = tensor.splat %scale : tensor<128x768xi32>
	%d_up = cinm.op.elementwise mul %d, %scale_v : tensor<128x768xi32>
	%y = cinm.op.elementwise div %d_up, %sigma_b : tensor<128x768xi32>

	%g_b = linalg.broadcast ins(%g : tensor<768xi32>) outs(%init_h : tensor<128x768xi32>) dimensions = [0]
	%b_b = linalg.broadcast ins(%b : tensor<768xi32>) outs(%init_h : tensor<128x768xi32>) dimensions = [0]
	%yg = cinm.op.elementwise mul %y, %g_b : tensor<128x768xi32>
	%unscale_v = tensor.splat %unscale : tensor<128x768xi32>
	%ygs = cinm.op.elementwise div %yg, %unscale_v : tensor<128x768xi32>
	%out = cinm.op.elementwise add %ygs, %b_b : tensor<128x768xi32>

	%narrowed = linalg.map ins(%out : tensor<128x768xi32>) outs(%init_o : tensor<128x768xi8>)
		(%v : i32, %o : i8) {
			%r = func.call @narrow_i8(%v) : (i32) -> i8
			linalg.yield %r : i8
		}
	return %narrowed : tensor<128x768xi8>
}

// ===========================================================================
// I-Softmax  (I-BERT 3.3)
//
// exp is never evaluated: with x~ = x - max(x) <= 0, write x~ = -z*ln2 + p
// for integer z and p in [-ln2, 0], so exp(x~) = 2^-z * exp(p), and exp(p)
// over that one interval is the second-order polynomial L below. The 2^-z
// is an arithmetic shift.
//
//   L(p) = 0.3585 (p + 1.353)^2 + 0.344      evaluated in Q14
//
// Row-wise over a 128x128 i32 score block, producing the i8 probabilities
// the value matmul multiplies. The max and the sum are cinm reductions, the
// exponent itself is one pointwise pass.
// ===========================================================================

func.func @isoftmax(%x : tensor<128x128xi32>) -> tensor<128x128xi8> {
	%init = tensor.empty() : tensor<128x128xi32>
	%init_o = tensor.empty() : tensor<128x128xi8>

	%mx = cinm.op.reduce maxsi (%x) : tensor<128x128xi32> -> tensor<128xi32>
	%mx_b = linalg.broadcast ins(%mx : tensor<128xi32>) outs(%init : tensor<128x128xi32>) dimensions = [1]
	%xt = cinm.op.elementwise sub %x, %mx_b : tensor<128x128xi32>

	%e = linalg.map ins(%xt : tensor<128x128xi32>) outs(%init : tensor<128x128xi32>)
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

	// Normalize into the int8 range the V matmul expects: p = 127 * e / sum.
	// e is at most 2^14 so the numerator stays inside i32, and the row's own
	// maximum contributes L(0) = 0.344 in Q14, so the sum is never zero.
	%s = cinm.op.reduce add (%e) : tensor<128x128xi32> -> tensor<128xi32>
	%s_b = linalg.broadcast ins(%s : tensor<128xi32>) outs(%init : tensor<128x128xi32>) dimensions = [1]
	%q127 = arith.constant 127 : i32
	%q127_v = tensor.splat %q127 : tensor<128x128xi32>
	%en = cinm.op.elementwise mul %e, %q127_v : tensor<128x128xi32>
	%p32 = cinm.op.elementwise div %en, %s_b : tensor<128x128xi32>

	%p8 = linalg.map ins(%p32 : tensor<128x128xi32>) outs(%init_o : tensor<128x128xi8>)
		(%v : i32, %o : i8) {
			%r = func.call @narrow_i8(%v) : (i32) -> i8
			linalg.yield %r : i8
		}
	return %p8 : tensor<128x128xi8>
}

// ===========================================================================
// I-GELU and integer tanh  (I-BERT 3.2)
//
//   erf(y) ~= sgn(y) [ a (clip(|y|, -b) + b)^2 + 1 ],  a = -0.2888, b = -1.769
//   GELU(x) = x/2 (1 + erf(x/sqrt2))
//
// and tanh(x) ~= erf(sqrt(pi)/2 x), so the pooler's tanh is the same
// polynomial with one scaling constant changed -- no second approximation.
// Both take and return the i8 activation: the Q4 input widens to i32, the
// polynomial evaluates in Q14, and the result narrows back.
// ===========================================================================

func.func @igelu(%x : tensor<128x3072xi8>) -> tensor<128x3072xi8> {
	%init = tensor.empty() : tensor<128x3072xi8>
	%r = linalg.map ins(%x : tensor<128x3072xi8>) outs(%init : tensor<128x3072xi8>)
		(%v8 : i8, %o : i8) {
			%up = arith.constant 10 : i32
			%q14 = arith.constant 14 : i32
			%q15 = arith.constant 15 : i32
			// 1/sqrt2 and -b = 1.769, both Q14
			%inv_r2 = arith.constant 11585 : i32
			%bq = arith.constant 28983 : i32
			%aq = arith.constant -4732 : i32
			%one = arith.constant 16384 : i32
			%zero = arith.constant 0 : i32

			%v = arith.extsi %v8 : i8 to i32
			%x14 = arith.shli %v, %up : i32
			%ax = math.absi %x14 : i32
			%y0 = arith.muli %ax, %inv_r2 : i32
			%y = arith.shrsi %y0, %q14 : i32
			%u = arith.minsi %y, %bq : i32
			%d = arith.subi %u, %bq : i32
			%d2 = arith.muli %d, %d : i32
			%d2s = arith.shrsi %d2, %q14 : i32
			%ad = arith.muli %aq, %d2s : i32
			%ads = arith.shrsi %ad, %q14 : i32
			%mag = arith.addi %ads, %one : i32
			%isneg = arith.cmpi slt, %v, %zero : i32
			%nmag = arith.subi %zero, %mag : i32
			%erf = arith.select %isneg, %nmag, %mag : i32
			// x/2 * (1 + erf): the +1 in Q14, then >>15 for the Q14 undo and
			// the halving together
			%sum = arith.addi %erf, %one : i32
			%prod = arith.muli %v, %sum : i32
			%g = arith.shrsi %prod, %q15 : i32
			%res = func.call @narrow_i8(%g) : (i32) -> i8
			linalg.yield %res : i8
		}
	return %r : tensor<128x3072xi8>
}

func.func @itanh(%x : tensor<768xi8>) -> tensor<768xi8> {
	%init = tensor.empty() : tensor<768xi8>
	%r = linalg.map ins(%x : tensor<768xi8>) outs(%init : tensor<768xi8>)
		(%v8 : i8, %o : i8) {
			%up = arith.constant 10 : i32
			%q14 = arith.constant 14 : i32
			// sqrt(pi)/2 in Q14, the only constant that differs from @igelu
			%k = arith.constant 14519 : i32
			%bq = arith.constant 28983 : i32
			%aq = arith.constant -4732 : i32
			%one = arith.constant 16384 : i32
			%zero = arith.constant 0 : i32

			%v = arith.extsi %v8 : i8 to i32
			%x14 = arith.shli %v, %up : i32
			%ax = math.absi %x14 : i32
			%y0 = arith.muli %ax, %k : i32
			%y = arith.shrsi %y0, %q14 : i32
			%u = arith.minsi %y, %bq : i32
			%d = arith.subi %u, %bq : i32
			%d2 = arith.muli %d, %d : i32
			%d2s = arith.shrsi %d2, %q14 : i32
			%ad = arith.muli %aq, %d2s : i32
			%ads = arith.shrsi %ad, %q14 : i32
			%mag = arith.addi %ads, %one : i32
			%isneg = arith.cmpi slt, %v, %zero : i32
			%nmag = arith.subi %zero, %mag : i32
			%erf = arith.select %isneg, %nmag, %mag : i32
			// tanh is in [-1, 1] in Q14; bring it back to the Q4 activation
			// scale the classifier matmul expects
			%t = arith.shrsi %erf, %up : i32
			%res = func.call @narrow_i8(%t) : (i32) -> i8
			linalg.yield %res : i8
		}
	return %r : tensor<768xi8>
}

// ===========================================================================
// Unrolling
//
// The layer loop and the per-head loop carry {unroll_layers} / {unroll_heads}
// and are fully unrolled by roberta_base.schedule.mlir, which the front end
// preloads rather than embedding here: a payload that still carries its
// schedule cannot reach host code generation. See that file.
// ===========================================================================
