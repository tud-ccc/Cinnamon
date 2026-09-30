// RoBERTa-base, W8A8, one 128-token sequence: the end-to-end model of the
// evaluation's E2E subsection (the model beside this file). The compiled
// function is the whole model; this driver only materialises its operands
// and times calls to it.
//
// What is different from the operator drivers next to this one:
//
//   - Operands are mixed-width. Weights and activations are int8, biases and
//     LayerNorm parameters int32, so the single DTY that common.hpp assumes
//     for its operand helpers does not cover the inputs. rnd<T>() below is
//     the typed equivalent of bench::random_vector; DTY stays int32 and is
//     what the logits come back in.
//   - There is no golden reference, and there cannot be one: the weights are
//     random and the fixed-point scales in the model are placeholders, so
//     the program is faithful in shape, op sequence and data movement, not
//     in what it computes (see the model's header). The run has to be
//     started with BENCH_CHECK=0; reference() refuses otherwise, so a run
//     that was expecting verification stops with a message rather than
//     comparing against nothing.
//   - The argument list is long and its order is the model's signature. It
//     is spelled out once, in ARGS, and the extern declaration and the call
//     are both generated from it, so the two cannot drift apart.
//
// 25 of the 28 inputs are `cinm.static`: their transfer amortizes over the
// serving lifetime and the runtime's residency cache (UPMEM_RT_CACHE=1) is
// what makes that physical. Iteration 0 pays the one-time costs and is
// dropped at assembly, like the RQ4 stack.

#include "../../common.hpp"

namespace {

constexpr size_t S = 128, H = 768, I = 3072, L = 12, V = 50265, P = 514;
constexpr size_t NUM_LABELS = 2;

template <class T> std::vector<T> rnd(size_t n) {
  return bench::interleaved_pages([n] {
    std::vector<T> v(n);
    for (auto &x : v)
      x = (T)(rand() % bench::kOperandRange);
    return v;
  });
}

// (name, C type, element count). Order is the function's signature; the
// result is appended by the lowering as a trailing out-parameter.
#define ROBERTA_ARGS(X)                                                        \
  X(input_ids, int32_t, S)                                                     \
  X(position_ids, int32_t, S)                                                  \
  X(attn_mask, int32_t, S *S)                                                  \
  X(word_emb, int8_t, V *H)                                                    \
  X(pos_emb, int8_t, P *H)                                                     \
  X(tok_type_emb, int8_t, H)                                                   \
  X(emb_ln_g, int32_t, H)                                                      \
  X(emb_ln_b, int32_t, H)                                                      \
  X(wq, int8_t, L * H * H)                                                     \
  X(wk, int8_t, L * H * H)                                                     \
  X(wv, int8_t, L * H * H)                                                     \
  X(wo, int8_t, L * H * H)                                                     \
  X(bq, int32_t, L *H)                                                         \
  X(bk, int32_t, L *H)                                                         \
  X(bv, int32_t, L *H)                                                         \
  X(bo, int32_t, L *H)                                                         \
  X(attn_ln_g, int32_t, L *H)                                                  \
  X(attn_ln_b, int32_t, L *H)                                                  \
  X(w1, int8_t, L * H * I)                                                     \
  X(b1, int32_t, L *I)                                                         \
  X(w2, int8_t, L * I * H)                                                     \
  X(b2, int32_t, L *H)                                                         \
  X(ffn_ln_g, int32_t, L *H)                                                   \
  X(ffn_ln_b, int32_t, L *H)                                                   \
  X(pool_w, int8_t, H *H)                                                      \
  X(pool_b, int32_t, H)                                                        \
  X(cls_w, int8_t, NUM_LABELS *H)                                              \
  X(cls_b, int32_t, NUM_LABELS)

} // namespace

#define DECL_PTR(name, ty, n) ty *,
extern "C" void BENCH_FN(ROBERTA_ARGS(DECL_PTR) int32_t * /*logits*/);
#undef DECL_PTR

struct RobertaBase {
  static constexpr bench::Size kSizes[] = {
      {"roberta_base", {S, H, 0}},
  };

#define MEMBER(name, ty, n) std::vector<ty> name;
  ROBERTA_ARGS(MEMBER)
#undef MEMBER
  std::vector<DTY> out;

  void setup(const size_t *) {
#define FILL(name, ty, n) name = rnd<ty>(n);
    ROBERTA_ARGS(FILL)
#undef FILL
    // Ids must index their tables; the mask is 0 (attend) or a large
    // negative (padding) in the accumulator domain, and a fixed sequence
    // needs no padding, so it is all zero.
    for (auto &t : input_ids)
      t = rand() % V;
    // RoBERTa offsets positions by padding_idx + 1 = 2.
    for (size_t i = 0; i < S; i++)
      position_ids[i] = (int32_t)(i + 2);
    std::fill(attn_mask.begin(), attn_mask.end(), 0);
    out = bench::output_vector(NUM_LABELS);
    printf("%s  S=%zu H=%zu L=%zu (W8A8)", TOSTR(BENCH_FN), S, H, L);
  }

  void run() {
#define PASS(name, ty, n) name.data(),
    BENCH_FN(ROBERTA_ARGS(PASS) out.data());
#undef PASS
  }

  const std::vector<DTY> &output() const { return out; }

  std::vector<double> reference() const {
    fprintf(stderr,
            "%s: this model has no golden reference (random weights, "
            "placeholder scales); run with BENCH_CHECK=0\n",
            TOSTR(BENCH_FN));
    exit(1);
  }
};

int main(int argc, char **argv) { return bench::run<RobertaBase>(argc, argv); }
