// One decode step of a W8A8 llama, for the end-to-end models of the
// evaluation's E2E subsection. The compiled function is the whole step; this
// driver only materialises its operands and times calls to it. It follows
// roberta_base.cpp: mixed-width operands, an X-macro for the argument list,
// no golden reference (random weights, placeholder scales -- run with
// BENCH_CHECK=0).
//
// The model's dimensions come from the including file, which is the driver
// of one size (llama2_110M_w8a8.cpp, llama2_7B_w8a8.cpp): everything below
// is spelled in terms of them, as the .mlir beside each is.
//
// What is specific to a decode step:
//
//   - Two scalar inputs, the token and its position, come before the
//     tensors. They are passed by value and so sit outside the X-macro.
//   - The caches are state: the step writes this position's k and v into
//     them in place. They are random here (a cache mid-generation), and
//     their contents do not change what the step costs.
//   - The position is fixed at the middle of the cache. The work does not
//     depend on it -- attention runs over all N rows and masks the rest --
//     so any position would time the same; the mask and the RoPE tables are
//     the only things derived from it.
//
// The weights are `cinm.static`: their transfer amortizes over the serving
// lifetime and the runtime's residency cache (UPMEM_RT_CACHE=1) is what
// makes that physical.

#include "../../common.hpp"

#include <cmath>
#include <cstdint>

namespace {

constexpr size_t QKV = 3 * H, PAIRS = H / 2, HEAD = H / A;
constexpr int32_t POS = N / 2;

// Filled by a xorshift rather than rand(): the 7B model materialises 7 GB
// of operands, and libc's generator would make that the slowest part of
// the run by far. The values are in the same range rand() would give.
template <class T> std::vector<T> rnd(size_t n) {
  return bench::interleaved_pages([n] {
    std::vector<T> v(n);
    uint32_t s = 0x9e3779b9u;
    for (auto &x : v) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      x = (T)(s % bench::kOperandRange);
    }
    return v;
  });
}

// (name, C type, element count). Order is the function's signature after
// the two scalars; the result is appended by the lowering as a trailing
// out-parameter.
#define LLAMA_ARGS(X)                                                          \
  X(attn_mask, int32_t, N)                                                     \
  X(kc, int8_t, L * N * H)                                                     \
  X(vc, int8_t, L * N * H)                                                     \
  X(rope_cos, int32_t, N *PAIRS)                                               \
  X(rope_sin, int32_t, N *PAIRS)                                               \
  X(embedding_table, int8_t, V *H)                                             \
  X(rms_att_weights, int32_t, L *H)                                            \
  X(wqkv, int8_t, L * QKV * H)                                                 \
  X(wo, int8_t, L * H * H)                                                     \
  X(rms_ffn_weights, int32_t, L *H)                                            \
  X(w1, int8_t, L * F * H)                                                     \
  X(w2, int8_t, L * H * F)                                                     \
  X(w3, int8_t, L * F * H)                                                     \
  X(rms_final_weight, int32_t, H)                                              \
  X(wcls, int8_t, VPAD *H)

} // namespace

#define DECL_PTR(name, ty, n) ty *,
extern "C" void BENCH_FN(int32_t /*token*/, int32_t /*pos*/,
                         LLAMA_ARGS(DECL_PTR) int32_t * /*logits*/);
#undef DECL_PTR

struct LlamaDecode {
  static constexpr bench::Size kSizes[] = {
      {TOSTR(BENCH_FN), {1, H, 0}},
  };

  int32_t token = 0;
#define MEMBER(name, ty, n) std::vector<ty> name;
  LLAMA_ARGS(MEMBER)
#undef MEMBER
  std::vector<DTY> out;

  void setup(const size_t *) {
#define FILL(name, ty, n) name = rnd<ty>(n);
    LLAMA_ARGS(FILL)
#undef FILL
    token = rand() % V;
    // Additive causal mask in the accumulator domain: 0 up to the position,
    // a large negative after it (the softmax floors its input anyway).
    for (size_t i = 0; i < N; i++)
      attn_mask[i] = (int32_t)i <= POS ? 0 : -10000;
    // RoPE tables in Q14: pair j of a head rotates by pos * 10000^(-2j/HEAD),
    // the same angle in every head. The model reads row `pos`; the whole
    // table is filled because it is a static operand of the model.
    for (size_t p = 0; p < N; p++)
      for (size_t j = 0; j < PAIRS; j++) {
        double head_dim = (double)((2 * j) % HEAD);
        double freq = 1.0 / std::pow(10000.0, head_dim / (double)HEAD);
        double angle = (double)p * freq;
        rope_cos[p * PAIRS + j] =
            (int32_t)std::lround(std::cos(angle) * 16384.0);
        rope_sin[p * PAIRS + j] =
            (int32_t)std::lround(std::sin(angle) * 16384.0);
      }
    // The classifier's pad rows are zero in the checkpoint.
    std::fill(wcls.begin() + V * H, wcls.end(), 0);
    out = bench::output_vector(V);
    printf("%s  decode pos=%d H=%zu F=%zu L=%zu (W8A8)", TOSTR(BENCH_FN), POS,
           H, F, L);
  }

  void run() {
#define PASS(name, ty, n) name.data(),
    BENCH_FN(token, POS, LLAMA_ARGS(PASS) out.data());
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

int main(int argc, char **argv) { return bench::run<LlamaDecode>(argc, argv); }
