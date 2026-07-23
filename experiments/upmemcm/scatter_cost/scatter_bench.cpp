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

#include <indicators/progress_bar.hpp>

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
#include <unistd.h>
#include <vector>

#define XFER_SG 0
#define XFER_BLOCK 1
#define XFER_BROADCAST 2

#ifndef XFER_MODE
#define XFER_MODE XFER_SG
#endif

namespace {

// Only XFER_SG exercises more than one block per DPU; the other two APIs
// each move exactly one contiguous block per DPU.
#if XFER_MODE == XFER_SG
constexpr int MAX_BLOCKS_PER_DPU = 24;
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

static std::vector<int> sampleFairLog2(bool dense, int max, int align,
                                       int sampling) {
  std::vector<int> s;
  for (int i = align; i < align * sampling; i += align)
    s.push_back(i);

  if (dense) {
    for (int lo = align * sampling; lo < MAX_DPUS; lo *= 2) {
      // between 16 and max dpus, we
      // take 16 samples in every bucket
      // between 2^n and 2^(n+1)
      for (int i = lo; i < lo * 2; i += lo / sampling) {
        s.push_back(i);
      }
    }
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

std::vector<int> genBlocksPerDpu() {
#if XFER_MODE == XFER_SG
  std::vector<int> v;
  for (int i = 1; i <= MAX_BLOCKS_PER_DPU; i++)
    v.push_back(i);
  return v;
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

#endif

// Single progress bar over all (num_dpus, blocks_per_dpu, block_size)
// configs, ticked once per config from the hot loop via a plain atomic;
// a background thread reads it and redraws every 150ms -- same pattern as
// SimpleProgressBar in lib/Dialect/Cinm/AcceleratorInference/Progress.h.
struct SweepProgress {
  using Bar = indicators::ProgressBar;
  bool active;
  std::unique_ptr<Bar> bar;
  std::atomic<size_t> done_{0};
  std::atomic<bool> stop_{false};
  std::thread printer_;

  explicit SweepProgress(size_t totalConfigs)
      : active(isatty(fileno(stdout))) {
    if (!active)
      return;
    bar = std::make_unique<Bar>(
        indicators::option::BarWidth{30},
        indicators::option::MaxProgress{totalConfigs},
        indicators::option::PrefixText{"configs "},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true});

    printer_ = std::thread([this] {
      while (!stop_.load(std::memory_order_relaxed)) {
        bar->set_progress(done_.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
      }
    });
  }

  void tick(size_t n = 1) { done_.fetch_add(n, std::memory_order_relaxed); }

  void finish() {
    if (!active)
      return;
    active = false;
    stop_.store(true, std::memory_order_relaxed);
    if (printer_.joinable())
      printer_.join();
    bar->mark_as_completed();
  }
  ~SweepProgress() { finish(); }
};

} // namespace

int main() {
  bool dense = std::getenv("SCATTER_DENSE") != nullptr;
  int iters = envInt("SCATTER_ITERS", 10);
  int warmup = envInt("SCATTER_WARMUP", 2);
  std::string csvPath = envStr("SCATTER_CSV_OUT", "results.csv");

  std::vector<int> dpuCounts = genDpuCounts(dense);
  std::vector<int> blocksPerDpuList = genBlocksPerDpu();
  std::vector<int> blockSizes = genBlockSizes(dense);

  size_t totalConfigs =
      dpuCounts.size() * blocksPerDpuList.size() * blockSizes.size();
  std::cerr << "scatter_bench: " << dpuCounts.size() << " dpu counts x "
            << blocksPerDpuList.size() << " blocks/dpu x " << blockSizes.size()
            << " block sizes = " << totalConfigs << " configs, " << iters
            << " iters each" << (dense ? " [dense]" : " [default]") << "\n";

  // One padded host arena, sized for the worst case, allocated once so
  // per-config allocation cost never pollutes the timing loop.
  size_t maxStride = static_cast<size_t>(MAX_BLOCK_SIZE) + BLOCK_PAD;
  size_t arenaBytes =
      static_cast<size_t>(MAX_DPUS) * MAX_BLOCKS_PER_DPU * maxStride;
  std::vector<uint8_t> arena(arenaBytes);
  for (size_t i = 0; i < arenaBytes; i++)
    arena[i] = static_cast<uint8_t>(i);

  std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
  if (!csv) {
    std::cerr << "scatter_bench: failed to open " << csvPath
              << " for writing\n";
    return 1;
  }
  csv << "num_dpus,blocks_per_dpu,block_size,iter,ns\n";
  csv.flush();

  SweepProgress progress(totalConfigs);
  size_t configsPerDpuCount = blocksPerDpuList.size() * blockSizes.size();

  // start with the biggest ones first because they're the slowest
  std::reverse(dpuCounts.begin(), dpuCounts.end());
  for (int numDpus : dpuCounts) {
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
      progress.tick(configsPerDpuCount);
      continue;
    }
    err = dpu_load(set, DPU_BINARY, NULL);
    if (err != DPU_OK) {
      std::cerr << "scatter_bench: dpu_load failed for " << numDpus
                << " dpus: " << dpu_error_to_string(err) << "\n";
      dpu_free(set);
      progress.tick(configsPerDpuCount);
      continue;
    }

    for (int blocksPerDpu : blocksPerDpuList) {
      for (int blockSize : blockSizes) {
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
        progress.tick();
      }
    }
    dpu_free(set);
  }

  progress.finish();
  std::cerr << "scatter_bench: done, results in " << csvPath << "\n";
  return 0;
}
