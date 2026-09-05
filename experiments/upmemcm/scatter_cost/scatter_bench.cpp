// Benchmark harness for UPMEM SDK host->DPU transfer APIs.
//
// Sweeps (num_dpus, blocks_per_dpu, block_size), timing a host-to-DPU
// transfer for each combination, and appends one CSV row per timed
// repetition to SCATTER_CSV_OUT (default results.csv) as it goes, so an
// interrupted run still leaves usable data.
//
// Which transfer API is benchmarked is chosen at compile time via XFER_MODE
// (see Makefile, which builds one binary per mode):
//   XFER_SG        (default) - dpu_push_sg_xfer, per-block scatter/gather.
//                    Sweeps blocks_per_dpu 1..MAX_BLOCKS_PER_DPU.
//   XFER_BLOCK     - dpu_push_xfer, one contiguous block per DPU
//                    (dpu_prepare_xfer'd from a distinct host offset per
//                    DPU). blocks_per_dpu is fixed at 1 (kept as a CSV
//                    column for compatibility with analyze.py).
//   XFER_BROADCAST - dpu_broadcast_to, the same host block copied to every
//                    DPU. blocks_per_dpu is fixed at 1.
//   XFER_GATHER    - dpu_push_xfer(DPU_XFER_FROM_DPU), one contiguous block
//                    per DPU read back to a distinct host offset per DPU
//                    (the reverse direction of XFER_BLOCK). blocks_per_dpu
//                    is fixed at 1.
//
// Env vars:
//   SCATTER_DENSE    - if set, sweep the full "multiples of 32" ranges in
//                       addition to the powers-of-2/1-10 default (~464k
//                       configs instead of ~4.8k; expect a multi-hour run).
//   SCATTER_ITERS    - timed repetitions per config (default 3).
//   SCATTER_WARMUP   - untimed warmup repetitions per config (default 1).
//   SCATTER_CSV_OUT  - output CSV path (default "results.csv").

extern "C" {
#include <dpu.h>
}

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

#define XFER_SG 0
#define XFER_BLOCK 1
#define XFER_BROADCAST 2
#define XFER_GATHER 3

#ifndef XFER_MODE
#define XFER_MODE XFER_SG
#endif

namespace {

// Only XFER_SG exercises more than one block per DPU; the other two APIs
// each move exactly one contiguous block per DPU.
#if XFER_MODE == XFER_SG
constexpr int MAX_BLOCKS_PER_DPU = 4096;
#else
constexpr int MAX_BLOCKS_PER_DPU = 1;
#endif
constexpr int MAX_BLOCK_SIZE = 8192;
// Gap inserted after every block so consecutive blocks are never adjacent in
// host memory -- otherwise the SDK could coalesce them into one contiguous
// copy and understate true per-block scatter overhead.
constexpr size_t BLOCK_PAD = 64;
constexpr int MAX_DPUS = 2048;
constexpr const char *DPU_BINARY = "bin/scatter_dpu";
constexpr const char *MRAM_SYMBOL = "buffer";

int envInt(const char *name, int def) {
  const char *v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

std::string envStr(const char *name, const char *def) {
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string(def);
}

/// In dense mode, this samples all points
///   - k * align for k in [1, sampling)
///   - k * align for k in [1, sampling)
static std::vector<int> sampleFairLog2(bool dense, int max, int align,
                                       int sampling) {
  std::vector<int> s;
  for (int i = align; i < align * sampling; i += align)
    s.push_back(i);

  if (dense) {
    for (int lo = align * sampling; lo < max; lo *= 2) {
      // we take `sampling` samples in every bucket
      // between 2^n and 2^(n+1)
      for (int i = lo; i < lo * 2; i += lo / sampling) {
        s.push_back(i);
      }
    }
    // The loop above is strict-< throughout (bucket boundary and outer
    // `lo < max`), so it always stops one step short of `max` itself (e.g.
    // max=4096 -> last value 3584) -- `max` is a corner of the swept space
    // just as much as the smallest values are, so make sure it's present.
    if (s.empty() || s.back() != max)
      s.push_back(max);
  } else {
    for (int p = align * sampling; p <= max; p *= 2)
      s.push_back(p);
  }
  return std::move(s);
}

std::vector<int> genDpuCounts(bool dense) {
  return sampleFairLog2(dense, MAX_DPUS, 1, 16);
}

// {powers of 2, 8..MAX_BLOCK_SIZE}, plus {multiples of 32} when dense.
std::vector<int> genBlockSizes(bool dense) {
  return sampleFairLog2(dense, MAX_BLOCK_SIZE, 8, 16);
}

std::vector<int> genBlocksPerDpu(bool dense) {
#if XFER_MODE == XFER_SG
  return sampleFairLog2(dense, MAX_BLOCKS_PER_DPU, 1, 4);
  // std::vector<int> v;
  // for (int i = 1; i <= MAX_BLOCKS_PER_DPU; i++)
  //   v.push_back(i);
  // return v;
#else
  return {1};
#endif
}

#if XFER_MODE == XFER_SG

// Arguments closed over by the get_block callback. The SDK copies this
// struct internally (get_block_t.args/args_size), so it's safe to keep on
// the stack of the calling function -- mirrors sg_xfer_context in
// runtime/Upmem/upmem_rt.c.
struct ScatterCtx {
  const uint8_t *arena;
  size_t stride; // block_size + BLOCK_PAD
  int blocks_per_dpu;
  uint32_t block_size;
};

bool getBlock(struct sg_block_info *out, uint32_t dpu_index,
              uint32_t block_index, void *args) {
  const auto *ctx = static_cast<const ScatterCtx *>(args);
  if (static_cast<int>(block_index) >= ctx->blocks_per_dpu)
    return false;
  size_t idx =
      static_cast<size_t>(dpu_index) * ctx->blocks_per_dpu + block_index;
  out->addr = const_cast<uint8_t *>(ctx->arena + idx * ctx->stride);
  out->length = ctx->block_size;
  return true;
}

// Times a dpu_push_sg_xfer of `blocksPerDpu` blocks of `blockSize` bytes to
// every DPU in `set`.
dpu_error_t runTransfer(struct dpu_set_t set, const uint8_t *arena,
                        size_t stride, int blocksPerDpu, uint32_t blockSize,
                        size_t length) {
  ScatterCtx ctx{arena, stride, blocksPerDpu, blockSize};
  get_block_t getBlockInfo{getBlock, &ctx, sizeof(ctx)};
  return dpu_push_sg_xfer(set, DPU_XFER_TO_DPU, MRAM_SYMBOL, 0, length,
                          &getBlockInfo, DPU_SG_XFER_DEFAULT);
}

#elif XFER_MODE == XFER_BLOCK

// Times a dpu_push_xfer of one contiguous `blockSize`-byte block per DPU,
// each read from a distinct offset in `arena` (dpu_prepare_xfer'd per DPU) --
// mirrors do_dpu_transfer in runtime/Upmem/upmem_rt.c.
dpu_error_t runTransfer(struct dpu_set_t set, const uint8_t *arena,
                        size_t stride, int /*blocksPerDpu*/,
                        uint32_t /*blockSize*/, size_t length) {
  struct dpu_set_t dpu;
  size_t i = 0;
  DPU_FOREACH(set, dpu, i) {
    dpu_error_t err =
        dpu_prepare_xfer(dpu, const_cast<uint8_t *>(arena + i * stride));
    if (err != DPU_OK)
      return err;
  }
  return dpu_push_xfer(set, DPU_XFER_TO_DPU, MRAM_SYMBOL, 0, length,
                       DPU_XFER_DEFAULT);
}

#elif XFER_MODE == XFER_BROADCAST

// Times a dpu_broadcast_to of one `blockSize`-byte block, copied to every
// DPU in `set`.
dpu_error_t runTransfer(struct dpu_set_t set, const uint8_t *arena,
                        size_t /*stride*/, int /*blocksPerDpu*/,
                        uint32_t /*blockSize*/, size_t length) {
  return dpu_broadcast_to(set, MRAM_SYMBOL, 0, arena, length, DPU_XFER_DEFAULT);
}

#elif XFER_MODE == XFER_GATHER

// Times a dpu_push_xfer(DPU_XFER_FROM_DPU) of one contiguous `blockSize`-byte
// block per DPU, each written back to a distinct offset in `arena`
// (dpu_prepare_xfer'd per DPU) -- the read-back mirror of XFER_BLOCK.
dpu_error_t runTransfer(struct dpu_set_t set, const uint8_t *arena,
                        size_t stride, int /*blocksPerDpu*/,
                        uint32_t /*blockSize*/, size_t length) {
  struct dpu_set_t dpu;
  size_t i = 0;
  DPU_FOREACH(set, dpu, i) {
    dpu_error_t err =
        dpu_prepare_xfer(dpu, const_cast<uint8_t *>(arena + i * stride));
    if (err != DPU_OK)
      return err;
  }
  return dpu_push_xfer(set, DPU_XFER_FROM_DPU, MRAM_SYMBOL, 0, length,
                       DPU_XFER_DEFAULT);
}

#endif

// Three stacked progress bars (dpus / blocks-per-dpu / block-size), rendered
// by a single background thread reading atomics -- same pattern as
// SimpleProgressBar in lib/Dialect/Cinm/AcceleratorInference/Progress.h.
struct SweepProgress {
  using Bar = indicators::ProgressBar;
  bool active;
  std::vector<std::unique_ptr<Bar>> bars;
  std::unique_ptr<indicators::DynamicProgress<Bar>> dyn;
  std::atomic<size_t> dpuDone_{0}, blockDone_{0}, sizeDone_{0};
  std::atomic<bool> stop_{false};
  std::thread printer_;

  SweepProgress(size_t nDpuCounts, size_t nBlocks, size_t nSizes)
      : active(isatty(fileno(stdout))) {
    if (!active)
      return;
    auto makeBar = [](size_t maxProgress, const std::string &prefix) {
      return std::make_unique<Bar>(indicators::option::BarWidth{30},
                                   indicators::option::MaxProgress{maxProgress},
                                   indicators::option::PrefixText{prefix},
                                   indicators::option::ShowPercentage{true},
                                   indicators::option::ShowElapsedTime{true},
                                   indicators::option::ShowRemainingTime{true});
    };
    bars.push_back(makeBar(nDpuCounts, "dpus       "));
    bars.push_back(makeBar(nBlocks, "blocks/dpu "));
    bars.push_back(makeBar(nSizes, "block size "));
    dyn = std::make_unique<indicators::DynamicProgress<Bar>>();
    for (auto &b : bars)
      dyn->push_back(*b);

    printer_ = std::thread([this] {
      while (!stop_.load(std::memory_order_relaxed)) {
        bars[0]->set_progress(dpuDone_.load(std::memory_order_relaxed));
        bars[1]->set_progress(blockDone_.load(std::memory_order_relaxed));
        bars[2]->set_progress(sizeDone_.load(std::memory_order_relaxed));
        std::cout << "\033[?2026h";
        dyn->print_progress();
        std::cout << "\033[?2026l" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
      }
    });
  }

  void resetBlocks() { blockDone_.store(0, std::memory_order_relaxed); }
  void resetSizes() { sizeDone_.store(0, std::memory_order_relaxed); }
  void tickDpu() { dpuDone_.fetch_add(1, std::memory_order_relaxed); }
  void tickBlock() { blockDone_.fetch_add(1, std::memory_order_relaxed); }
  void tickSize() { sizeDone_.fetch_add(1, std::memory_order_relaxed); }

  void finish() {
    if (!active)
      return;
    active = false;
    stop_.store(true, std::memory_order_relaxed);
    if (printer_.joinable())
      printer_.join();
    for (auto &b : bars)
      b->mark_as_completed();
  }
  ~SweepProgress() { finish(); }
};

bool ispow2(int n) { return (n & (n - 1)) == 0; }

// Corner-point filter: past MAX_BLOCK_SIZE*16 total bytes/DPU, most of the
// (blocks_per_dpu, block_size) plane is unrealistic (real kernels don't
// exercise it) and prohibitively slow to sweep exhaustively -- only sample
// the "regular" grid points there (power-of-2 blocks_per_dpu/block_size,
// dpu count a multiple of 64) instead of every combination.
bool sampleInExpensiveRegion(int blocksPerDpu, int blockSize, int numDpus) {
  bool result = true;
  result &= ispow2(blocksPerDpu) || ispow2(blocksPerDpu - 1);
  result &= ispow2(blockSize) || ispow2(blockSize - 1);
  result &= numDpus % 64 == 0 || numDpus % 64 == 1;
  return result;
}
bool shouldSkipExpensive(int blocksPerDpu, int blockSize, int numDpus) {
  return static_cast<long>(blocksPerDpu) * blockSize > MAX_BLOCK_SIZE * 16 &&
         !sampleInExpensiveRegion(blocksPerDpu, blockSize, numDpus);
}

// Reads (num_dpus, blocks_per_dpu, block_size) triples already present in
// `path` (if it exists), for incremental resume -- a config is "already
// measured" as soon as it has at least one row, regardless of iters count.
std::set<std::tuple<int, int, int>>
loadExistingConfigs(const std::string &path) {
  std::set<std::tuple<int, int, int>> done;
  std::ifstream in(path);
  if (!in)
    return done;
  std::string line;
  std::getline(in, line); // header
  int numDpus, blocksPerDpu, blockSize, iter;
  long long ns;
  while (std::getline(in, line)) {
    if (std::sscanf(line.c_str(), "%d,%d,%d,%d,%lld", &numDpus, &blocksPerDpu,
                    &blockSize, &iter, &ns) == 5)
      done.emplace(numDpus, blocksPerDpu, blockSize);
  }
  return done;
}
} // namespace

int main() {
  bool dense = std::getenv("SCATTER_DENSE") != nullptr;
  int iters = envInt("SCATTER_ITERS", 10);
  int warmup = envInt("SCATTER_WARMUP", 2);
  std::string csvPath = envStr("SCATTER_CSV_OUT", "results.csv");

  std::vector<int> dpuCounts = genDpuCounts(dense);
  std::vector<int> blocksPerDpuList = genBlocksPerDpu(dense);
  std::vector<int> blockSizes = genBlockSizes(dense);

  size_t totalConfigs =
      dpuCounts.size() * blocksPerDpuList.size() * blockSizes.size();
  std::cerr << "scatter_bench: " << dpuCounts.size() << " dpu counts x "
            << blocksPerDpuList.size() << " blocks/dpu x " << blockSizes.size()
            << " block sizes = " << totalConfigs << " configs, " << iters
            << " iters each" << (dense ? " [dense]" : " [default]") << "\n";

  // Incremental resume: a prior (possibly interrupted) run's results.csv is
  // read for (num_dpus, blocks_per_dpu, block_size) triples already
  // present, and every such config is skipped below -- so re-running after
  // an interruption (or after widening the swept ranges) only measures what
  // isn't already on disk, instead of redoing an hour-long sweep from
  // scratch.
  std::set<std::tuple<int, int, int>> alreadyDone =
      loadExistingConfigs(csvPath);
  bool resuming = !alreadyDone.empty();
  if (resuming)
    std::cerr << "scatter_bench: resuming -- " << alreadyDone.size()
              << " configs already in " << csvPath << "\n";

  // One padded host arena, sized for the worst case, allocated once so
  // per-config allocation cost never pollutes the timing loop.
  size_t maxStride = static_cast<size_t>(MAX_BLOCK_SIZE) + BLOCK_PAD;
  size_t arenaBytes =
      static_cast<size_t>(MAX_DPUS) * MAX_BLOCKS_PER_DPU * maxStride;
  std::vector<uint8_t> arena(arenaBytes);
  for (size_t i = 0; i < arenaBytes; i++)
    arena[i] = static_cast<uint8_t>(i);

  std::ofstream csv(csvPath, resuming ? (std::ios::out | std::ios::app)
                                      : (std::ios::out | std::ios::trunc));
  if (!csv) {
    std::cerr << "scatter_bench: failed to open " << csvPath
              << " for writing\n";
    return 1;
  }
  if (!resuming) {
    csv << "num_dpus,blocks_per_dpu,block_size,iter,ns\n";
    csv.flush();
  }

  SweepProgress progress(dpuCounts.size(), blocksPerDpuList.size(),
                         blockSizes.size());

  // start with the biggest ones first because they're the slowest
  std::reverse(dpuCounts.begin(), dpuCounts.end());
  for (int numDpus : dpuCounts) {
    // Every config for this dpu count is either already measured or would
    // be skipped by the corner filter anyway -- skip dpu_alloc/dpu_load
    // entirely rather than pay for a rank allocation with nothing to do.
    bool allDoneForDpu = true;
    for (int blocksPerDpu : blocksPerDpuList) {
      for (int blockSize : blockSizes) {
        if (shouldSkipExpensive(blocksPerDpu, blockSize, numDpus))
          continue;
        if (!alreadyDone.count({numDpus, blocksPerDpu, blockSize})) {
          allDoneForDpu = false;
          break;
        }
      }
      if (!allDoneForDpu)
        break;
    }
    if (allDoneForDpu) {
      progress.tickDpu();
      continue;
    }

    struct dpu_set_t set;
    dpu_error_t err;
#if XFER_MODE == XFER_SG
    char profile[128];
    std::snprintf(profile, sizeof(profile),
                  "sgXferEnable=true,sgXferMaxBlocksPerDpu=%d",
                  MAX_BLOCKS_PER_DPU);
    err = dpu_alloc(numDpus, profile, &set);
#else
    err = dpu_alloc(numDpus, NULL, &set);
#endif
    if (err != DPU_OK) {
      std::cerr << "scatter_bench: dpu_alloc(" << numDpus
                << ") failed: " << dpu_error_to_string(err)
                << " -- skipping this DPU count\n";
      progress.tickDpu();
      continue;
    }
    err = dpu_load(set, DPU_BINARY, NULL);
    if (err != DPU_OK) {
      std::cerr << "scatter_bench: dpu_load failed for " << numDpus
                << " dpus: " << dpu_error_to_string(err) << "\n";
      dpu_free(set);
      progress.tickDpu();
      continue;
    }

    progress.resetBlocks();
    for (int blocksPerDpu : blocksPerDpuList) {
      progress.resetSizes();
      for (int blockSize : blockSizes) {
        if (shouldSkipExpensive(blocksPerDpu, blockSize, numDpus)) {
          // This is in the "expensive region".
          // We only sample here if we are exactly on a
          // "regular" config (power of 2 params).
          progress.tickSize();
          continue;
        }
        if (alreadyDone.count({numDpus, blocksPerDpu, blockSize})) {
          progress.tickSize();
          continue;
        }

        size_t stride = static_cast<size_t>(blockSize) + BLOCK_PAD;
        size_t length =
            static_cast<size_t>(blocksPerDpu) * static_cast<size_t>(blockSize);

        for (int w = 0; w < warmup; w++)
          runTransfer(set, arena.data(), stride, blocksPerDpu,
                      static_cast<uint32_t>(blockSize), length);

        for (int it = 0; it < iters; it++) {
          auto t0 = std::chrono::steady_clock::now();
          err = runTransfer(set, arena.data(), stride, blocksPerDpu,
                            static_cast<uint32_t>(blockSize), length);
          auto t1 = std::chrono::steady_clock::now();
          if (err != DPU_OK) {
            std::cerr << "scatter_bench: transfer failed (dpus=" << numDpus
                      << " blocks=" << blocksPerDpu << " size=" << blockSize
                      << "): " << dpu_error_to_string(err) << "\n";
            continue;
          }
          int64_t ns =
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                  .count();
          csv << numDpus << "," << blocksPerDpu << "," << blockSize << "," << it
              << "," << ns << "\n";
          csv.flush();
        }
        progress.tickSize();
      }
      progress.tickBlock();
    }
    dpu_free(set);
    progress.tickDpu();
  }

  progress.finish();
  std::cerr << "scatter_bench: done, results in " << csvPath << "\n";
  return 0;
}
