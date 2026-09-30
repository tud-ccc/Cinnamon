// One decode step of a W8A8 llama, for the end-to-end models of the
// evaluation's E2E subsection. The compiled function is the whole step; this
// driver only materialises its operands and times calls to it. It follows
// roberta_base.cpp: mixed-width operands, an X-macro for the argument list,
// no golden reference (random weights, placeholder scales -- run with
// BENCH_CHECK=0).
//
// The model's dimensions come from the including file, which is the driver
// of one size (llama2_110M_w8a8.cpp, llama2_7B_w8a8.cpp, llama3_8B_w8a8_*):
// everything below is spelled in terms of them, as the .mlir beside each is.
// KVH is the number of key/value heads: A for multi-head attention (the
// Llama-2 models), fewer for grouped-query attention (Llama 3), where each
// key/value head serves A / KVH query heads and the caches and the K and V
// projections are KVH / A the width of the queries.
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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <thread>

// An operand buffer whose elements start uninitialised, so that the threads
// filling it are the ones faulting its pages in.
template <class T> struct Uninit : std::allocator<T> {
  template <class U> struct rebind {
    using other = Uninit<U>;
  };
  template <class U> void construct(U *p) { ::new (static_cast<void *>(p)) U; }
};
template <class T> using Buffer = std::vector<T, Uninit<T>>;

namespace {

constexpr size_t HEAD = H / A, KV = KVH * HEAD, QKV = H + 2 * KV, PAIRS = H / 2;
constexpr int32_t POS = N / 2;

// Filled by a xorshift per chunk, on every core: a model materialises
// gigabytes of operands, which one thread takes tens of seconds to fill.
// The values are in the same range rand() would give.
template <class T> Buffer<T> rnd(size_t n) {
  return bench::interleaved_pages([n] {
    Buffer<T> v(n);
    constexpr size_t kChunk = size_t(1) << 22;
    size_t chunks = (n + kChunk - 1) / kChunk;
    std::atomic<size_t> next{0};
    auto work = [&] {
      for (size_t c; (c = next++) < chunks;) {
        uint32_t s = (0x9e3779b9u ^ (uint32_t(c) * 0x85ebca6bu)) | 1u;
        for (size_t i = c * kChunk, e = std::min(n, i + kChunk); i < e; i++) {
          s ^= s << 13;
          s ^= s >> 17;
          s ^= s << 5;
          v[i] = (T)(s % bench::kOperandRange);
        }
      }
    };
    std::vector<std::thread> pool(
        std::max(1u, std::thread::hardware_concurrency()));
    for (auto &t : pool)
      t = std::thread(work);
    for (auto &t : pool)
      t.join();
    return v;
  });
}

// (name, C type, element count). Order is the function's signature after
// the two scalars; the result is appended by the lowering as a trailing
// out-parameter.
#define LLAMA_ARGS(X)                                                          \
  X(attn_mask, int32_t, N)                                                     \
  X(kc, int8_t, L * N * KV)                                                    \
  X(vc, int8_t, L * N * KV)                                                    \
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
#define MEMBER(name, ty, n) Buffer<ty> name;
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
    printf("%s  decode pos=%d H=%zu F=%zu L=%zu KVH=%zu N=%zu (W8A8)",
           TOSTR(BENCH_FN), POS, H, F, L, KVH, N);
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
