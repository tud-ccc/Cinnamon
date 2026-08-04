#pragma once

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

/// True when progress bars can render to stdout without conflicting with IR
/// output — i.e. the result IR is diverted to a file and stdout is a terminal.
bool canRenderProgress();

/// A single progress bar driven by a background printer thread.
/// Workers call tick(); the printer calls set_progress() every ~150 ms.
/// Active only when canRenderProgress() is true.
struct SimpleProgressBar {
  using Bar = indicators::ProgressBar;
  bool active;
  std::unique_ptr<Bar> bar;
  std::atomic<size_t> done_{0};
  std::atomic<bool> stop_{false};
  std::thread printer_;

  SimpleProgressBar(size_t maxProgress, std::string prefix)
      : active(canRenderProgress()) {
    if (!active)
      return;
    bar = std::make_unique<Bar>(
        indicators::option::BarWidth{30},
        indicators::option::MaxProgress{maxProgress},
        indicators::option::PrefixText{std::move(prefix)},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true});
    printer_ = std::thread([this] {
      while (!stop_.load(std::memory_order_relaxed)) {
        std::cout << "\033[?2026h";
        bar->set_progress(done_.load(std::memory_order_relaxed));
        std::cout << "\033[?2026l" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    });
  }

  void tick() {
    if (active)
      done_.fetch_add(1, std::memory_order_relaxed);
  }

  void finish() {
    if (!active)
      return;
    active = false;
    stop_.store(true, std::memory_order_relaxed);
    if (printer_.joinable())
      printer_.join();
    // Printer thread has stopped; one final render to mark completion.
    // Do NOT call set_progress() here — that would render a second line.
    bar->mark_as_completed();
  }
  ~SimpleProgressBar() { finish(); }
};

/// Live multi-bar progress for concurrent seeds: one overall bar (seeds
/// completed) plus one bar per worker slot (current seed's evaluations).
///
/// Worker threads ONLY write to atomic counters. A single printer thread owns
/// all bar mutations so that DynamicProgress cursor-up tracking is never
/// disrupted by concurrent renders triggered from worker threads.
struct MultiSeedProgress {
  using Bar = indicators::ProgressBar;
  bool active;
  int maxEvals;
  unsigned cap_;
  std::vector<std::unique_ptr<Bar>> bars; // [0]=overall, [1..cap_]=slots
  std::unique_ptr<indicators::DynamicProgress<Bar>> dyn;
  std::thread printer;
  std::atomic<bool> stop{false};
  std::atomic<bool> finished{false};

  // Written by worker threads, read exclusively by the printer thread.
  std::atomic<int> seedsDone_{0};
  std::unique_ptr<std::atomic<int>[]> slotProgress_;  // [0..cap_)
  std::unique_ptr<std::atomic<int>[]> slotSeedValue_; // [0..cap_), -1 = idle

  MultiSeedProgress(int nSeeds, unsigned cap, int maxEvals)
      : active(canRenderProgress()), maxEvals(maxEvals), cap_(cap),
        slotProgress_(std::make_unique<std::atomic<int>[]>(cap)),
        slotSeedValue_(std::make_unique<std::atomic<int>[]>(cap)) {
    for (unsigned i = 0; i < cap; ++i)
      slotSeedValue_[i].store(-1, std::memory_order_relaxed);

    if (!active)
      return;

    auto makeBar = [](size_t maxProgress, const std::string &prefix) {
      return std::make_unique<Bar>(indicators::option::BarWidth{30},
                                   indicators::option::MaxProgress{maxProgress},
                                   indicators::option::PrefixText{prefix},
                                   indicators::option::ShowPercentage{true},
                                   indicators::option::ShowElapsedTime{true});
    };
    bars.push_back(
        makeBar(static_cast<size_t>(std::max(1, nSeeds)), "seeds        "));
    for (unsigned t = 0; t < cap; ++t)
      bars.push_back(
          makeBar(static_cast<size_t>(std::max(1, maxEvals)), "  slot idle  "));
    dyn = std::make_unique<indicators::DynamicProgress<Bar>>();
    for (auto &b : bars)
      dyn->push_back(*b);

    printer = std::thread([this] {
      while (!stop.load(std::memory_order_relaxed)) {
        // Only the printer thread calls set_option/set_progress on bars.
        // DynamicProgress re-renders (cursor-up + reprint) on each
        // set_progress call, so keeping all mutations here prevents racing
        // renders that would break the cursor position.
        bars[0]->set_progress(
            static_cast<size_t>(seedsDone_.load(std::memory_order_relaxed)));
        for (unsigned i = 0; i < cap_; ++i) {
          int sv = slotSeedValue_[i].load(std::memory_order_relaxed);
          char buf[32];
          if (sv < 0)
            std::strncpy(buf, "  slot idle  ", sizeof(buf));
          else
            std::snprintf(buf, sizeof(buf), "  seed %6d", sv);
          bars[i + 1]->set_option(indicators::option::PrefixText{buf});
          bars[i + 1]->set_progress(static_cast<size_t>(
              std::max(0, slotProgress_[i].load(std::memory_order_relaxed))));
        }
        // Flush all bar updates to the terminal in one pass (cursor-up + reprint).
        std::cout << "\033[?2026h";
        dyn->print_progress();
        std::cout << "\033[?2026l" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
      }
    });
  }

  // Worker-thread API: write atomics only, never touch bars directly.
  void startSeed(unsigned slot, int seedValue) {
    slotSeedValue_[slot].store(seedValue, std::memory_order_relaxed);
    slotProgress_[slot].store(0, std::memory_order_relaxed);
  }
  void seedProgress(unsigned slot, int nObs) {
    slotProgress_[slot].store(std::min(nObs, maxEvals),
                              std::memory_order_relaxed);
  }
  void seedDone(unsigned slot) {
    seedsDone_.fetch_add(1, std::memory_order_relaxed);
    slotSeedValue_[slot].store(-1, std::memory_order_relaxed);
    slotProgress_[slot].store(0, std::memory_order_relaxed);
  }

  void finish() {
    if (!active || finished.exchange(true))
      return;
    stop.store(true, std::memory_order_relaxed);
    if (printer.joinable())
      printer.join();
    dyn->print_progress();
    std::cout << std::endl;
  }
  ~MultiSeedProgress() { finish(); }
};
