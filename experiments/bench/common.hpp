// Shared harness for every bench driver in this directory.
//
// A driver supplies only what is specific to its primitive -- the kernel
// signature, a table of problem sizes, how to fill the operands, and a golden
// reference -- and calls bench::run<Op>(). Everything else (operand
// distribution, timing loop, stats dump, CSV format, verification) lives here
// so those decisions are made once.
//
// Every driver is built as a single translation unit with
// -DBENCH_FN=<function_name>; the problem size is looked up from that name at
// runtime, so the Makefile needs nothing beyond the function name.
//
// Binary interface: bench_<fn> <output_dir> [<iters>]
// Environment: BENCH_CHECK=0 skips the golden-reference check (see
// check_enabled()); unset or 1 verifies.

#pragma once

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <strings.h>
#include <vector>

#include <cblas.h>
#include <numa.h>

#ifndef BENCH_FN
#error "BENCH_FN must be defined at compile time (-DBENCH_FN=<function_name>)"
#endif

#ifndef DTY
#define DTY int32_t
#endif
#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

extern "C" {
void upmemrt_start_stat_collection(int iter);
void upmemrt_dump_stats(const char *prefix);
}

namespace bench {

// ─── Operand data ────────────────────────────────────────────────────────────

/// Upper bound (exclusive) on generated operand values.
///
/// Matches the ATiM evaluation's distribution (its evaluation/base.py uses
/// intdist=50) so latency comparisons against it are like for like.
///
/// This is not a free parameter. On this hardware an integer multiply is a
/// __mulsi3 call costing one step per significant bit of its smaller operand,
/// so the range directly sets the price of the hottest instruction in every
/// multiply-bearing kernel. The cost model's kMulOperandBits is derived from
/// it and has to move with it. The range also has to keep reductions inside
/// DTY -- see check_representable(), which enforces that per benchmark rather
/// than leaving it to be assumed.
inline constexpr int kOperandRange = 50;

inline DTY next_operand() { return (DTY)(rand() % kOperandRange); }

inline std::vector<DTY> random_vector(size_t n) {
  std::vector<DTY> v(n);
  for (size_t i = 0; i < n; i++)
    v[i] = next_operand();
  return v;
}

/// Like random_vector, with operands drawn from [0, range) instead of
/// [0, kOperandRange). The chained-matmul benchmarks (2mm/3mm) need this:
/// with default-range operands the second matmul stage overflows i32, so
/// they draw 0/1 inputs, which keep every stage exact in both the i32
/// kernel and the double reference.
inline std::vector<DTY> random_vector(size_t n, int range) {
  std::vector<DTY> v(n);
  for (size_t i = 0; i < n; i++)
    v[i] = (DTY)(rand() % range);
  return v;
}

/// Zero-initialised output buffer with its pages interleaved across NUMA
/// nodes. Every driver's gather target must be allocated through this.
///
/// The gather (dpu_push_xfer FROM_DPU) is executed by per-rank SDK worker
/// threads that libnuma pins to their rank's socket, and its bandwidth is set
/// by where the destination pages live: faulted from the main thread (what a
/// plain vector constructor does) they all land on one node, half the ranks
/// write cross-socket, and a 256 MiB gather runs at ~6.5 MiB/ms; interleaved
/// it runs at ~10.7-11.1 MiB/ms (measured on the 2-node bench machine, 2048
/// DPUs x 128 KiB, matching the raw-SDK probe both ways). ATiM's harness
/// gathers at ~10.3, so single-node placement here would charge our gather
/// term a ~1.6x penalty that is page placement, not the compiler under test.
///
/// Interleave rather than first-touch-per-rank because the driver does not
/// know the rank -> host-offset mapping; interleaving is within noise of the
/// measured optimum. The policy is scoped to this allocation: operands stay
/// on the default policy, since the scatter direction measures the same
/// single-node and interleaved (~12 MiB/ms).
inline std::vector<DTY> output_vector(size_t n) {
  if (numa_available() < 0 || numa_max_node() == 0)
    return std::vector<DTY>(n, 0);
  numa_set_interleave_mask(numa_all_nodes_ptr);
  std::vector<DTY> v(n, 0); // faulted here, under the interleave policy
  numa_set_localalloc();
  return v;
}

// ─── Problem sizes ───────────────────────────────────────────────────────────

/// One row of a driver's size table: the exported function name and up to
/// three dimensions, interpreted by that driver.
struct Size {
  const char *fn;
  size_t dims[3];
};

/// The dimensions for BENCH_FN. Exits with a pointed message when the name is
/// absent: a new benchmark size needs a table entry, and silently guessing a
/// shape would mean benchmarking something other than what was compiled.
template <size_t N> inline const size_t *lookup(const Size (&table)[N]) {
  const char *fn = TOSTR(BENCH_FN);
  for (const Size &s : table)
    if (!strcmp(fn, s.fn))
      return s.dims;
  fprintf(stderr,
          "%s: no size entry for '%s' -- add one to this driver's kSizes\n",
          TOSTR(BENCH_FN), fn);
  exit(1);
}

// ─── Golden references ───────────────────────────────────────────────────────
//
// References are computed in double, which holds every value these benchmarks
// can produce exactly (operands are small non-negative integers and the
// longest reduction is well inside 2^53), so results compare for equality
// rather than tolerance. Matrix references go through CBLAS -- an
// implementation with nothing in common with the code under test, which is the
// point of having a reference at all.

/// Largest double working buffer a reference may allocate, in bytes. Matrices
/// here reach hundreds of megabytes as int32, so a reference converts them a
/// row block at a time rather than materialising a whole double copy.
inline constexpr size_t kRefBlockBytes = 64u << 20;

/// out[0..m) = A * x, with A an m×n row-major matrix.
inline void gemv_ref(const DTY *A, const std::vector<DTY> &x, size_t m,
                     size_t n, double *out) {
  std::vector<double> xd(x.begin(), x.end());
  size_t rows_per_block =
      std::max<size_t>(1, kRefBlockBytes / (n * sizeof(double)));
  std::vector<double> block;
  for (size_t r0 = 0; r0 < m; r0 += rows_per_block) {
    size_t rows = std::min(rows_per_block, m - r0);
    block.assign(A + r0 * n, A + (r0 + rows) * n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)rows, (int)n, 1.0,
                block.data(), (int)n, xd.data(), 1, 0.0, out + r0, 1);
  }
}

inline std::vector<double> gemv_ref(const std::vector<DTY> &A,
                                    const std::vector<DTY> &x, size_t m,
                                    size_t n) {
  std::vector<double> out(m);
  gemv_ref(A.data(), x, m, n, out.data());
  return out;
}

/// out = A * B, with A m×k and B k×n, both row-major. Templated over the
/// element types so a chained reference can feed one stage's double output
/// into the next (2mm/3mm); everything accumulates in double, which is
/// exact for these benchmarks' operand ranges (see the note above).
template <class TA, class TB>
inline void matmul_ref(const TA *A, const TB *B, size_t m, size_t k, size_t n,
                       double *out) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++)
      out[i * n + j] = 0.0;
    for (size_t l = 0; l < k; l++) {
      const double a = (double)A[i * k + l];
      for (size_t j = 0; j < n; j++)
        out[i * n + j] += a * (double)B[l * n + j];
    }
  }
}

/// out[i] = alpha*u[i] + beta*v[i], elementwise over n.
inline std::vector<double> axpby_ref(const std::vector<DTY> &u,
                                     const std::vector<DTY> &v, double alpha,
                                     double beta, size_t n) {
  std::vector<double> out(u.begin(), u.begin() + n);
  std::vector<double> vd(v.begin(), v.begin() + n);
  cblas_dscal((int)n, alpha, out.data(), 1);
  cblas_daxpy((int)n, beta, vd.data(), 1, out.data(), 1);
  return out;
}

/// Sum of the first n elements. Accumulated straight from the int32 input so
/// the reduction needs no double copy of a buffer that can reach 256 MB.
inline double sum_ref(const std::vector<DTY> &v, size_t n) {
  double acc = 0;
  for (size_t i = 0; i < n; i++)
    acc += (double)v[i];
  return acc;
}

// ─── Verification ────────────────────────────────────────────────────────────

/// Whether run() checks the kernel output against the golden reference, read
/// from BENCH_CHECK so the same binary can do either.
///
/// Verification is the default, so a measurement campaign cannot quietly
/// produce unverified numbers by omission. It is worth turning off for the
/// search stacks, where the reference costs more than the kernel it checks:
/// a CBLAS dgemv over a 512 MB operand runs on one host core against a
/// kernel spread over 2048 DPUs.
///
/// Accepted spellings, case-insensitively: 1/on/true/yes and 0/off/false/no.
/// Anything else is a mistake worth stopping for -- guessing would mean
/// benchmarking under a verification setting other than the one asked for,
/// which is exactly what this knob exists to make explicit.
inline bool check_enabled() {
  const char *v = getenv("BENCH_CHECK");
  if (!v || !*v)
    return true;
  auto is = [v](const char *s) { return !strcasecmp(v, s); };
  if (is("1") || is("on") || is("true") || is("yes"))
    return true;
  if (is("0") || is("off") || is("false") || is("no"))
    return false;
  fprintf(stderr,
          "%s: BENCH_CHECK='%s' is not a boolean -- use 1/on/true/yes or "
          "0/off/false/no\n",
          TOSTR(BENCH_FN), v);
  exit(1);
}

/// Reports whether every golden value fits in DTY. A benchmark whose exact
/// result overflows its own element type cannot be verified and is not
/// measuring anything meaningful, so this is a sizing error in the benchmark
/// rather than a fault in the kernel, and it is reported as such.
inline bool check_representable(const std::vector<double> &want) {
  constexpr double lo = (double)std::numeric_limits<DTY>::min();
  constexpr double hi = (double)std::numeric_limits<DTY>::max();
  for (size_t i = 0; i < want.size(); i++) {
    if (want[i] < lo || want[i] > hi) {
      fprintf(stderr,
              "\n%s: OVERFLOW -- exact result[%zu] = %.0f does not fit in a "
              "%zu-bit element type.\nThe benchmark size and operand range "
              "(bench::kOperandRange = %d) are incompatible; the kernel cannot "
              "produce this value.\n",
              TOSTR(BENCH_FN), i, want[i], 8 * sizeof(DTY), kOperandRange);
      return false;
    }
  }
  return true;
}

/// Compares kernel output against the golden result, reporting the first few
/// differences. Exact equality: both sides are integers held exactly.
inline bool check(const std::vector<DTY> &got,
                  const std::vector<double> &want) {
  assert(got.size() == want.size() && "reference length mismatch");
  if (!check_representable(want))
    return false;

  size_t bad = 0;
  for (size_t i = 0; i < got.size(); i++) {
    if ((double)got[i] == want[i])
      continue;
    if (++bad <= 5)
      fprintf(stderr, "  [%zu] got %lld, expected %.0f\n", i, (long long)got[i],
              want[i]);
  }
  if (bad == 0) {
    printf("%s: verified %zu element(s) against reference\n", TOSTR(BENCH_FN),
           got.size());
    return true;
  }
  fprintf(stderr, "\n%s: MISMATCH -- %zu of %zu elements differ\n",
          TOSTR(BENCH_FN), bad, got.size());
  return false;
}

// ─── Driver ──────────────────────────────────────────────────────────────────

inline uint64_t now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

inline void write_totals(const char *out_dir,
                         const std::vector<uint64_t> &elapsed_ns) {
  char prefix[1024];
  snprintf(prefix, sizeof(prefix), "%s/" TOSTR(BENCH_FN), out_dir);
  upmemrt_dump_stats(prefix);

  char path[1024];
  snprintf(path, sizeof(path), "%s/" TOSTR(BENCH_FN) "_total.csv", out_dir);
  FILE *f = fopen(path, "w");
  assert(f && "failed to open total CSV");
  fprintf(f, "iter,elapsed_ns\n");
  for (size_t i = 0; i < elapsed_ns.size(); i++)
    fprintf(f, "%zu,%" PRIu64 "\n", i, elapsed_ns[i]);
  fclose(f);
}

/// Runs one benchmark end to end.
///
/// `Op` supplies:
///   static constexpr Size kSizes[]   -- problem sizes by function name
///   void setup(const size_t *dims)   -- allocate and fill operands
///   void run()                       -- one call of BENCH_FN
///   const std::vector<DTY> &output() const -- what the kernel produced
///   std::vector<double> reference() const -- what it should have produced
///
/// Verification happens after the timed loop but before anything is written,
/// so a run that computed the wrong answer leaves no timing data behind to be
/// plotted later. It can be switched off per run -- see check_enabled() -- in
/// which case the run is timed and written like any other.
template <class Op> int run(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output_dir> [<iters>]\n", argv[0]);
    return 1;
  }
  const char *out_dir = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 5;

  srand(0);
  Op op;
  op.setup(lookup(Op::kSizes));
  printf("  iters=%d\n", iters);
  fflush(stdout);

  std::vector<uint64_t> elapsed_ns(iters);
  for (int iter = 0; iter < iters; iter++) {
    upmemrt_start_stat_collection(iter);
    uint64_t t0 = now_ns();
    op.run();
    elapsed_ns[iter] = now_ns() - t0;
    printf("  iter %d  %.3f ms\n", iter, elapsed_ns[iter] / 1e6);
    fflush(stdout);
  }

  // The reference is only built when it is going to be used: it is the
  // expensive half of a skipped check, not the comparison.
  if (check_enabled()) {
    if (!check(op.output(), op.reference()))
      return 1;
  } else {
    printf("%s: verification skipped (BENCH_CHECK)\n", TOSTR(BENCH_FN));
  }

  write_totals(out_dir, elapsed_ns);
  return 0;
}

} // namespace bench
