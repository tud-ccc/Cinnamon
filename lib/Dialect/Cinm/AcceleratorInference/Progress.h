#pragma once

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/// True when progress bars can render to stdout without conflicting with IR
/// output — i.e. the result IR is diverted to a file.
bool canRenderProgress();

/// True when stdout is a terminal. When it is not (stdout redirected to a log
/// file), progress is emitted as plain text lines without escape codes, at the
/// much slower `logProgressInterval` cadence.
bool progressIsTerminal();

/// Render period when stdout is a log file: one line per render, so keep it
/// sparse enough that a long search does not bloat the log.
inline constexpr std::chrono::milliseconds logProgressInterval{20000};
/// Render period for the multi-bar live display on a terminal.
inline constexpr std::chrono::milliseconds multiBarTtyInterval{150};
/// Render period for the single live bar on a terminal.
inline constexpr std::chrono::milliseconds singleBarTtyInterval{2000};

/// Formats a duration as `MM:SS`, or `HH:MM:SS` past one hour.
std::string formatElapsed(std::chrono::steady_clock::duration d);

/// A single progress bar driven by a background printer thread.
/// Workers call tick(); the printer renders periodically.
/// Active only when canRenderProgress() is true.
struct SimpleProgressBar {
  using Bar = indicators::ProgressBar;
  bool active;
  bool tty;
  size_t maxProgress_;
  std::string prefix_;
  std::chrono::steady_clock::time_point start_;
  std::unique_ptr<Bar> bar;
  std::atomic<size_t> done_{0};
  bool stop_ = false; // guarded by mu_
  std::mutex mu_;
  std::condition_variable cv_;
  std::thread printer_;

  /// `enabled` = false silences the bar entirely (all methods become no-ops):
  /// how a caller running many searches at once keeps them from interleaving
  /// their renders.
  SimpleProgressBar(size_t maxProgress, std::string prefix, bool enabled = true)
      : active(enabled && canRenderProgress()), tty(progressIsTerminal()),
        maxProgress_(maxProgress), prefix_(std::move(prefix)),
        start_(std::chrono::steady_clock::now()) {
    if (!active)
      return;
    if (tty)
      bar = std::make_unique<Bar>(indicators::option::BarWidth{30},
                                  indicators::option::MaxProgress{maxProgress_},
                                  indicators::option::PrefixText{prefix_},
                                  indicators::option::ShowPercentage{true},
                                  indicators::option::ShowElapsedTime{true},
                                  indicators::option::ShowRemainingTime{true});
    printer_ = std::thread([this] {
      do {
        render();
      } while (!waitForStop());
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
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (printer_.joinable())
      printer_.join();
    if (tty)
      // Printer thread has stopped; one final render to mark completion.
      // Do NOT call set_progress() here — that would render a second line.
      bar->mark_as_completed();
    else
      render();
  }
  ~SimpleProgressBar() { finish(); }

private:
  void render() {
    size_t done = done_.load(std::memory_order_relaxed);
    if (tty) {
      // Wrap the update in a synchronized-output sequence so the terminal
      // shows it in one pass.
      std::cout << "\033[?2026h";
      bar->set_progress(done);
      std::cout << "\033[?2026l" << std::flush;
      return;
    }
    std::cout << "[progress "
              << formatElapsed(std::chrono::steady_clock::now() - start_)
              << "] " << prefix_ << ' ' << done << '/' << maxProgress_ << '\n'
              << std::flush;
  }

  /// Sleeps until the next render is due. True once finish() has been called.
  bool waitForStop() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, tty ? singleBarTtyInterval : logProgressInterval,
                 [this] { return stop_; });
    return stop_;
  }
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
  bool tty;
  int nSeeds;
  int maxEvals;
  unsigned cap_;
  std::chrono::steady_clock::time_point start_;
  std::vector<std::unique_ptr<Bar>> bars; // [0]=overall, [1..cap_]=slots
  std::unique_ptr<indicators::DynamicProgress<Bar>> dyn;
  std::thread printer;
  bool stop = false; // guarded by mu_
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> finished{false};

  // Written by worker threads, read exclusively by the printer thread.
  std::atomic<int> seedsDone_{0};
  std::unique_ptr<std::atomic<int>[]> slotProgress_;  // [0..cap_)
  std::unique_ptr<std::atomic<int>[]> slotSeedValue_; // [0..cap_), -1 = idle

  /// `enabled` = false silences all bars; see SimpleProgressBar.
  MultiSeedProgress(int nSeeds, unsigned cap, int maxEvals, bool enabled = true)
      : active(enabled && canRenderProgress()), tty(progressIsTerminal()),
        nSeeds(nSeeds), maxEvals(maxEvals), cap_(cap),
        start_(std::chrono::steady_clock::now()),
        slotProgress_(std::make_unique<std::atomic<int>[]>(cap)),
        slotSeedValue_(std::make_unique<std::atomic<int>[]>(cap)) {
    for (unsigned i = 0; i < cap; ++i)
      slotSeedValue_[i].store(-1, std::memory_order_relaxed);

    if (!active)
      return;

    if (tty) {
      auto makeBar = [](size_t maxProgress, const std::string &prefix) {
        return std::make_unique<Bar>(
            indicators::option::BarWidth{30},
            indicators::option::MaxProgress{maxProgress},
            indicators::option::PrefixText{prefix},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowElapsedTime{true});
      };
      bars.push_back(
          makeBar(static_cast<size_t>(std::max(1, nSeeds)), "seeds        "));
      for (unsigned t = 0; t < cap; ++t)
        bars.push_back(makeBar(static_cast<size_t>(std::max(1, maxEvals)),
                               "  slot idle  "));
      dyn = std::make_unique<indicators::DynamicProgress<Bar>>();
      for (auto &b : bars)
        dyn->push_back(*b);
    }

    printer = std::thread([this] {
      do {
        render();
      } while (!waitForStop());
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
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop = true;
    }
    cv_.notify_all();
    if (printer.joinable())
      printer.join();
    if (tty) {
      dyn->print_progress();
      std::cout << std::endl;
    } else {
      render();
    }
  }
  ~MultiSeedProgress() { finish(); }

private:
  void render() {
    int seedsDone = seedsDone_.load(std::memory_order_relaxed);
    if (tty) {
      // Only the printer thread calls set_option/set_progress on bars.
      // DynamicProgress re-renders (cursor-up + reprint) on each set_progress
      // call, so keeping all mutations here prevents racing renders that would
      // break the cursor position.
      bars[0]->set_progress(static_cast<size_t>(seedsDone));
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
      // Flush all bar updates to the terminal in one pass (cursor-up +
      // reprint).
      std::cout << "\033[?2026h";
      dyn->print_progress();
      std::cout << "\033[?2026l" << std::flush;
      return;
    }
    // One plain line per render, all slots inlined, so a log tail still shows
    // what every worker is doing.
    std::cout << "[progress "
              << formatElapsed(std::chrono::steady_clock::now() - start_)
              << "] seeds " << seedsDone << '/' << nSeeds;
    for (unsigned i = 0; i < cap_; ++i) {
      int sv = slotSeedValue_[i].load(std::memory_order_relaxed);
      if (sv < 0) {
        std::cout << " | idle";
        continue;
      }
      std::cout << " | seed " << sv << ": "
                << std::max(0, slotProgress_[i].load(std::memory_order_relaxed))
                << '/' << maxEvals;
    }
    std::cout << '\n' << std::flush;
  }

  /// Sleeps until the next render is due. True once finish() has been called.
  bool waitForStop() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, tty ? multiBarTtyInterval : logProgressInterval,
                 [this] { return stop; });
    return stop;
  }
};
