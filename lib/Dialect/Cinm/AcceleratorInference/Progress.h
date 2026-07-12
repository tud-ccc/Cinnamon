#pragma once

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

/// True when progress bars can render to stdout without conflicting with IR
/// output — i.e. the result IR is diverted to a file and stdout is a terminal.
bool canRenderProgress();

/// A single progress bar driven by a background printer thread.
/// Workers increment the atomic counter; the printer calls set_progress() every
/// ~150 ms. Active only when canRenderProgress() is true.
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
        bar->set_progress(done_.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
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
    bar->set_progress(done_.load(std::memory_order_relaxed));
    bar->mark_as_completed();
  }
  ~SimpleProgressBar() { finish(); }
};

/// Live multi-bar progress for concurrent seeds: one overall bar (seeds
/// completed) plus one bar per worker slot (current seed's evaluations). A
/// dedicated printer thread renders every ~150 ms so worker threads only touch
/// the bars' internal (mutex-guarded, non-printing in multi-progress mode)
/// setters. `indicators::DynamicProgress` renders to std::cout; that also
/// carries the result IR, so bars are only enabled when canRenderProgress().
struct MultiSeedProgress {
  using Bar = indicators::ProgressBar;
  bool active;
  int maxEvals;
  std::vector<std::unique_ptr<Bar>> bars; // [0]=overall, [1..cap]=slots
  std::unique_ptr<indicators::DynamicProgress<Bar>> dyn;
  std::thread printer;
  std::atomic<bool> stop{false};
  std::atomic<bool> finished{false};

  MultiSeedProgress(int nSeeds, unsigned cap, int maxEvals)
      : active(canRenderProgress()), maxEvals(maxEvals) {
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
        dyn->print_progress();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
      }
    });
  }

  void startSeed(unsigned slot, int seedValue) {
    if (!active)
      return;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "  seed %6d", seedValue);
    bars[slot + 1]->set_option(indicators::option::PrefixText{buf});
    bars[slot + 1]->set_progress(0);
  }
  void seedProgress(unsigned slot, int nObs) {
    if (active)
      bars[slot + 1]->set_progress(
          static_cast<size_t>(std::min(nObs, maxEvals)));
  }
  void seedDone() {
    if (active)
      bars[0]->tick();
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
