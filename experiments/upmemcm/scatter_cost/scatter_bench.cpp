// Benchmark harness for the UPMEM SDK's dpu_push_sg_xfer scatter API.
//
// Sweeps (num_dpus, blocks_per_dpu, block_size), timing a host-to-DPU
// scatter transfer for each combination, and appends one CSV row per timed
// repetition to SCATTER_CSV_OUT (default results.csv) as it goes, so an
// interrupted run still leaves usable data.
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

namespace {

constexpr int MAX_BLOCKS_PER_DPU = 24;
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

// {1..10} ∪ {powers of 2 up to MAX_DPUS}, plus {multiples of 32 up to
// MAX_DPUS} when dense is requested.
std::vector<int> genDpuCounts(bool dense) {
  std::set<int> s;
  for (int i = 1; i <= 10; i++)
    s.insert(i);
  for (int p = 1; p <= MAX_DPUS; p *= 2)
    s.insert(p);
  if (dense)
    for (int m = 32; m <= MAX_DPUS; m += 32)
      s.insert(m);
  return {s.begin(), s.end()};
}

// {powers of 2, 8..MAX_BLOCK_SIZE}, plus {multiples of 32} when dense.
std::vector<int> genBlockSizes(bool dense) {
  std::set<int> s;
  for (int p = 8; p <= MAX_BLOCK_SIZE; p *= 2)
    s.insert(p);
  if (dense)
    for (int m = 32; m <= MAX_BLOCK_SIZE; m += 32)
      s.insert(m);
  return {s.begin(), s.end()};
}

std::vector<int> genBlocksPerDpu() {
  std::vector<int> v;
  for (int i = 1; i <= MAX_BLOCKS_PER_DPU; i++)
    v.push_back(i);
  return v;
}

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

// Three stacked progress bars (dpus / blocks-per-dpu / block-size), rendered
// by a single background thread reading atomics -- same pattern as
// SimpleProgressBar/MultiSeedProgress in
// lib/Dialect/Cinm/AcceleratorInference/Progress.h.
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
      return std::make_unique<Bar>(
          indicators::option::BarWidth{30},
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

} // namespace

int main() {
  bool dense = std::getenv("SCATTER_DENSE") != nullptr;
  int iters = envInt("SCATTER_ITERS", 3);
  int warmup = envInt("SCATTER_WARMUP", 1);
  std::string csvPath = envStr("SCATTER_CSV_OUT", "results.csv");

  std::vector<int> dpuCounts = genDpuCounts(dense);
  std::vector<int> blocksPerDpuList = genBlocksPerDpu();
  std::vector<int> blockSizes = genBlockSizes(dense);

  size_t totalConfigs =
      dpuCounts.size() * blocksPerDpuList.size() * blockSizes.size();
  std::cerr << "scatter_bench: " << dpuCounts.size() << " dpu counts x "
            << blocksPerDpuList.size() << " blocks/dpu x "
            << blockSizes.size() << " block sizes = " << totalConfigs
            << " configs, " << iters << " iters each"
            << (dense ? " [dense]" : " [default]") << "\n";

  // One padded host arena, sized for the worst case, allocated once so
  // per-config allocation cost never pollutes the timing loop.
  size_t stride = static_cast<size_t>(MAX_BLOCK_SIZE) + BLOCK_PAD;
  size_t arenaBytes =
      static_cast<size_t>(MAX_DPUS) * MAX_BLOCKS_PER_DPU * stride;
  std::vector<uint8_t> arena(arenaBytes);
  for (size_t i = 0; i < arenaBytes; i++)
    arena[i] = static_cast<uint8_t>(i);

  std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
  if (!csv) {
    std::cerr << "scatter_bench: failed to open " << csvPath << " for writing\n";
    return 1;
  }
  csv << "num_dpus,blocks_per_dpu,block_size,iter,ns\n";
  csv.flush();

  SweepProgress progress(dpuCounts.size(), blocksPerDpuList.size(),
                          blockSizes.size());

  for (int numDpus : dpuCounts) {
    struct dpu_set_t set;
    char profile[128];
    std::snprintf(profile, sizeof(profile),
                  "sgXferEnable=true,sgXferMaxBlocksPerDpu=%d",
                  MAX_BLOCKS_PER_DPU);
    dpu_error_t err = dpu_alloc(numDpus, profile, &set);
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
        ScatterCtx ctx{arena.data(), static_cast<size_t>(blockSize) + BLOCK_PAD,
                       blocksPerDpu, static_cast<uint32_t>(blockSize)};
        get_block_t getBlockInfo{getBlock, &ctx, sizeof(ctx)};
        size_t length =
            static_cast<size_t>(blocksPerDpu) * static_cast<size_t>(blockSize);

        for (int w = 0; w < warmup; w++)
          dpu_push_sg_xfer(set, DPU_XFER_TO_DPU, MRAM_SYMBOL, 0, length,
                            &getBlockInfo, DPU_SG_XFER_DEFAULT);

        for (int it = 0; it < iters; it++) {
          auto t0 = std::chrono::steady_clock::now();
          err = dpu_push_sg_xfer(set, DPU_XFER_TO_DPU, MRAM_SYMBOL, 0, length,
                                  &getBlockInfo, DPU_SG_XFER_DEFAULT);
          auto t1 = std::chrono::steady_clock::now();
          if (err != DPU_OK) {
            std::cerr << "scatter_bench: dpu_push_sg_xfer failed (dpus="
                       << numDpus << " blocks=" << blocksPerDpu
                       << " size=" << blockSize
                       << "): " << dpu_error_to_string(err) << "\n";
            continue;
          }
          int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           t1 - t0)
                           .count();
          csv << numDpus << "," << blocksPerDpu << "," << blockSize << ","
              << it << "," << ns << "\n";
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
