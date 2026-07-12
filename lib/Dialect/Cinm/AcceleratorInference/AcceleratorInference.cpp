#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>

#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unistd.h>
#include <variant>
#include <vector>

#define DEBUG_TYPE "cinm-inference"

using mlir::cinm::utils::Maybe;

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// SearchParam
// ===----------------------------------------------------------------------===//

double SearchParam::dlo() const {
  return std::visit(
      [](auto &&d) -> double {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return static_cast<double>(d.lo);
        else
          return 0.0;
      },
      domain);
}

double SearchParam::dhi() const {
  return std::visit(
      [](auto &&d) -> double {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return static_cast<double>(d.hi);
        else
          return static_cast<double>(d.values.size() - 1);
      },
      domain);
}

int64_t SearchParam::cardinality() const {
  return std::visit(
      [](auto &&d) -> int64_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return (d.hi - d.lo) / d.step + 1;
        else
          return static_cast<int64_t>(d.values.size());
      },
      domain);
}

int64_t SearchParam::discretize(double v) const {
  return std::visit(
      [v](auto &&d) -> int64_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>) {
          int64_t rounded =
              static_cast<int64_t>(std::round(v / d.step)) * d.step;
          return std::clamp(rounded, d.lo, d.hi);
        } else {
          size_t idx = static_cast<size_t>(
              std::clamp(static_cast<int64_t>(std::round(v)), int64_t(0),
                         static_cast<int64_t>(d.values.size() - 1)));
          return d.values[idx];
        }
      },
      domain);
}

double SearchParam::featurize(int64_t v) const {
  return std::visit(
      [v](auto &&d) -> double {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>) {
          return v;
        } else {
          // return v;
          return log2(v);
          // auto idx = std::find(d.values.begin(), d.values.end(), v);
          // assert(idx != d.values.end());
          // return std::distance(d.values.begin(), idx);
        }
      },
      domain);
}

SearchParam &SearchParam::keepDivisorsOf(int64_t n) {
  if (auto *range = std::get_if<IntRange>(&domain)) {
    std::vector<int64_t> kept;
    for (int64_t v = range->lo; v <= range->hi; v += range->step)
      if (v > 0 && n % v == 0)
        kept.push_back(v);
    domain = ValueList{std::move(kept)};
  } else {
    auto &vals = std::get<ValueList>(domain).values;
    vals.erase(std::remove_if(vals.begin(), vals.end(),
                              [n](int64_t v) { return v <= 0 || n % v != 0; }),
               vals.end());
  }
  return *this;
}

int64_t SearchParam::valueAt(size_t subIdx) const {
  return std::visit(
      [subIdx](auto &&d) -> int64_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return d.lo + static_cast<int64_t>(subIdx) * d.step;
        else
          return d.values[subIdx];
      },
      domain);
}

size_t SearchParam::subIndexOf(int64_t value) const {
  return std::visit(
      [value](auto &&d) -> size_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return static_cast<size_t>((value - d.lo) / d.step);
        else {
          auto it = std::find(d.values.begin(), d.values.end(), value);
          return static_cast<size_t>(std::distance(d.values.begin(), it));
        }
      },
      domain);
}

// ===----------------------------------------------------------------------===//
// SearchParam factories
// ===----------------------------------------------------------------------===//

SearchParam makeRange(llvm::StringRef name, int64_t lo, int64_t hi,
                      int64_t step) {
  return SearchParam(name, IntRange{lo, hi, step});
}

SearchParam makePow2Range(llvm::StringRef name, int64_t loExp, int64_t hiExp) {
  std::vector<int64_t> vals;
  for (int64_t e = loExp; e <= hiExp; ++e)
    vals.push_back(int64_t(1) << e);
  return SearchParam(name, ValueList{std::move(vals)});
}

SearchParam makeValues(llvm::StringRef name, std::vector<int64_t> values) {
  return SearchParam(name, ValueList{std::move(values)});
}

// ===----------------------------------------------------------------------===//
// ConfigSpace
// ===----------------------------------------------------------------------===//
int ConfigSpace::findIndex(llvm::StringRef name) const {
  for (int i = 0; i < static_cast<int>(params.size()); ++i)
    if (params[i].name == name)
      return i;
  return -1;
}

int64_t ConfigSpace::get(const Configuration &config,
                         llvm::StringRef name) const {
  int idx = findIndex(name);
  if (idx < 0 || idx >= static_cast<int>(config.size()))
    return 0;
  return config[idx];
}

bool ConfigSpace::isValid(const Configuration &config) const {
  auto wrapper = ConfWrapper(*this, config);
  for (auto &c : constraints)
    if (!c(wrapper))
      return false;
  return true;
}

void ConfigSpace::ensureEncoding() const {
  if (encodingValid_)
    return;

  slots_.clear();

  std::vector<bool> isChild(params.size(), false);
  for (const auto &g : groups)
    isChild[g.childIdx] = true;

  std::vector<size_t> parentToGroup(params.size(), SIZE_MAX);
  for (size_t gi = 0; gi < groups.size(); ++gi)
    parentToGroup[groups[gi].parentIdx] = gi;

  for (size_t i = 0; i < params.size(); ++i) {
    if (isChild[i])
      continue;
    size_t gi = parentToGroup[i];
    size_t slotSize = (gi == SIZE_MAX)
                          ? static_cast<size_t>(params[i].cardinality())
                          : groups[gi].totalCount();
    slots_.push_back({i, gi, slotSize});
  }

  const size_t S = slots_.size();
  suffixProd_.resize(S + 1);
  suffixProd_[S] = 1;
  for (size_t i = S; i-- > 0;)
    suffixProd_[i] = suffixProd_[i + 1] * slots_[i].slotSize;

  encodingValid_ = true;
}

void ConfigSpace::addMultiplesConstraint(StringRef parentName,
                                         StringRef childName) {
  int parentIdx = findIndex(parentName);
  int childIdx = findIndex(childName);
  assert(parentIdx >= 0);
  assert(childIdx >= 0);
  const SearchParam &parent = params[parentIdx];
  const SearchParam &child = params[childIdx];
  int64_t parentCard = parent.cardinality();

  std::vector<std::vector<int64_t>> childValues(
      static_cast<size_t>(parentCard));
  for (int64_t pi = 0; pi < parentCard; ++pi) {
    int64_t parentVal = parent.valueAt(static_cast<size_t>(pi));
    for (int64_t ci = 0, cc = child.cardinality(); ci < cc; ++ci) {
      int64_t childVal = child.valueAt(static_cast<size_t>(ci));
      if (childVal % parentVal == 0)
        childValues[static_cast<size_t>(pi)].push_back(childVal);
    }
  }

  std::vector<size_t> cumCount(static_cast<size_t>(parentCard) + 1);
  cumCount[0] = 0;
  for (int64_t pi = 0; pi < parentCard; ++pi)
    cumCount[static_cast<size_t>(pi) + 1] =
        cumCount[static_cast<size_t>(pi)] +
        childValues[static_cast<size_t>(pi)].size();

  groups.push_back({static_cast<size_t>(parentIdx),
                    static_cast<size_t>(childIdx), std::move(childValues),
                    std::move(cumCount)});
  encodingValid_ = false;
}

size_t ConfigSpace::totalSize() const {
  ensureEncoding();
  return slots_.empty() ? 1 : suffixProd_[0];
}

void ConfigSpace::at(size_t idx, Configuration &conf) const {
  ensureEncoding();
  conf.resize(params.size());
  for (size_t si = slots_.size(); si-- > 0;) {
    const auto &slot = slots_[si];
    size_t subIdx = idx % slot.slotSize;
    idx /= slot.slotSize;
    if (slot.groupIdx == SIZE_MAX) {
      conf[slot.dimIdx] = params[slot.dimIdx].valueAt(subIdx);
    } else {
      const auto &g = groups[slot.groupIdx];
      auto it = std::upper_bound(g.cumCount.begin(), g.cumCount.end(), subIdx);
      --it;
      size_t parentSubIdx = static_cast<size_t>(it - g.cumCount.begin());
      size_t childLocalIdx = subIdx - g.cumCount[parentSubIdx];
      conf[g.parentIdx] = params[g.parentIdx].valueAt(parentSubIdx);
      conf[g.childIdx] = g.childValues[parentSubIdx][childLocalIdx];
    }
  }
}

void ConfigSpace::forEach(
    std::function<bool(const Configuration &, size_t)> fn) const {
  ensureEncoding();
  const size_t S = slots_.size();
  // Use suffixProd_[0] directly — avoids a redundant ensureEncoding() call
  // inside totalSize() after we already ensured encoding above.
  const size_t total = S == 0 ? 1 : suffixProd_[0];
  if (total == 0)
    return;

  // Per-slot combined sub-index in [0, slot.slotSize).
  std::vector<size_t> subIdx(S, 0);
  Configuration conf(params.size());

  // Decode sub-index k for slot si and write the corresponding parameter
  // values into conf.  For grouped slots this uses the same upper_bound logic
  // as ConfigSpace::at(), which correctly handles parents that have no valid
  // children (their cumCount entries are equal and are never selected).
  auto applySubIdx = [&](size_t si, size_t k) {
    const auto &slot = slots_[si];
    if (slot.groupIdx == SIZE_MAX) {
      conf[slot.dimIdx] = params[slot.dimIdx].valueAt(k);
    } else {
      const auto &g = groups[slot.groupIdx];
      auto it = std::upper_bound(g.cumCount.begin(), g.cumCount.end(), k);
      --it; // it now points to the last cumCount entry ≤ k
      size_t psi = static_cast<size_t>(it - g.cumCount.begin());
      size_t cli = k - g.cumCount[psi];
      conf[g.parentIdx] = params[g.parentIdx].valueAt(psi);
      conf[g.childIdx] = g.childValues[psi][cli];
    }
  };

  // Initialise conf at sub-index 0 for every slot.
  for (size_t si = 0; si < S; ++si)
    applySubIdx(si, 0);

  for (size_t flat = 0; flat < total; ++flat) {
    if (!fn(conf, flat))
      return;

    if (flat + 1 == total)
      break;

    // Mixed-radix increment from the least-significant slot.
    for (size_t si = S; si-- > 0;) {
      ++subIdx[si];
      bool carry = (subIdx[si] >= slots_[si].slotSize);
      if (carry)
        subIdx[si] = 0;
      applySubIdx(si, subIdx[si]);
      if (!carry)
        break;
    }
  }
}

size_t ConfigSpace::indexOf(const Configuration &conf) const {
  ensureEncoding();
  size_t idx = 0;
  for (size_t si = 0; si < slots_.size(); ++si) {
    const auto &slot = slots_[si];
    size_t subIdx;
    if (slot.groupIdx == SIZE_MAX) {
      subIdx = params[slot.dimIdx].subIndexOf(conf[slot.dimIdx]);
    } else {
      const auto &g = groups[slot.groupIdx];
      size_t parentSubIdx = params[g.parentIdx].subIndexOf(conf[g.parentIdx]);
      const auto &cv = g.childValues[parentSubIdx];
      auto it = std::find(cv.begin(), cv.end(), conf[g.childIdx]);
      subIdx = g.cumCount[parentSubIdx] + static_cast<size_t>(it - cv.begin());
    }
    idx = idx * slot.slotSize + subIdx;
  }
  return idx;
}

void ConfigSpace::neighborIndices(size_t idx,
                                  llvm::SmallVectorImpl<size_t> &result) const {
  ensureEncoding();
  for (size_t si = 0; si < slots_.size(); ++si) {
    size_t stride = suffixProd_[si + 1];
    size_t subIdx = (idx / stride) % slots_[si].slotSize;
    if (subIdx > 0)
      result.push_back(idx - stride);
    if (subIdx + 1 < slots_[si].slotSize)
      result.push_back(idx + stride);
  }
}

void ConfigSpace::dump(llvm::raw_ostream &out,
                       const Configuration &config) const {
  out << " {";
  for (auto [i, dim, value] : llvm::enumerate(params, config)) {
    out << dim.name << "=" << value << (i + 1 < size() ? ", " : "");
  }
  out << "}";
}

// ===----------------------------------------------------------------------===//
// Core framework
// ===----------------------------------------------------------------------===//

// Build a minimal trial module: module { func @host(arg0, arg1, ...) -> (r0,
// r1, ...) {
//   %r = cinm.compute_block(arg0, arg1, ...) { <clone of computeOp body> }
//   return %r
// } }
// Returns the module and the cloned compute block (the refClone).
static std::pair<mlir::OwningOpRef<mlir::ModuleOp>, cinm::ComputeBlockOp>
buildRefModule(cinm::ComputeBlockOp computeOp) {
  mlir::MLIRContext *ctx = computeOp->getContext();
  mlir::Location loc = computeOp->getLoc();
  mlir::OpBuilder b(ctx);

  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  auto hostFunc = mlir::func::FuncOp::create(
      loc, "host",
      mlir::FunctionType::get(
          ctx, llvm::SmallVector<mlir::Type>(computeOp->getOperandTypes()),
          llvm::SmallVector<mlir::Type>(computeOp->getResultTypes())));
  module->getBody()->push_back(hostFunc);
  mlir::Block *entry = hostFunc.addEntryBlock();
  b.setInsertionPointToStart(entry);

  mlir::IRMapping mapping;
  for (auto [operand, arg] :
       llvm::zip(computeOp->getOperands(), entry->getArguments()))
    mapping.map(operand, arg);

  auto *cloned = b.clone(*computeOp, mapping);
  mlir::func::ReturnOp::create(b, loc, cloned->getResults());

  return {std::move(module), llvm::cast<cinm::ComputeBlockOp>(cloned)};
}

void buildConfigSpace(cinm::ComputeBlockOp refClone, InferencePlugin &plugin,
                      ConfigSpace &space) {
  plugin.initializeSpace(refClone, space);
  LLVM_DEBUG({
    int64_t totalPoints = 1;
    for (auto &p : space.params)
      totalPoints *= p.cardinality();
    llvm::dbgs() << "[cinm-inference] Config space (" << space.size()
                 << " params, " << totalPoints << " total points):\n";
    for (auto &p : space.params)
      llvm::dbgs() << "  " << p.name << " in [" << p.dlo() << ", " << p.dhi()
                   << "] (" << p.cardinality() << " points)\n";
  });
}

/// A fixed set of per-thread evaluation workers. Each worker owns a warmed-up
/// InferencePlugin clone and its own reference-module clone, so evaluate() can
/// run concurrently from accept() callbacks. Workers are leased for the
/// duration of a call and returned automatically, so with one worker per pool
/// thread a worker is always available.
struct EvaluatorPool {
  EvaluatorPool(InferencePlugin &plugin, ModuleOp refModule,
                mlir::MLIRContext *ctx, unsigned nWorkers)
      : workers(nWorkers), freeWorkers(nWorkers) {
    for (unsigned i = 0; i < nWorkers; ++i) {
      workers[i].plugin = plugin.clone();
      workers[i].plugin->warmUp(ctx);
      workers[i].ref =
          OwningOpRef<ModuleOp>(llvm::cast<ModuleOp>(refModule->clone()));
      freeWorkers[i] = i;
    }
  }

  unsigned size() const { return static_cast<unsigned>(workers.size()); }

  /// Lease a worker, invoke `fn(plugin, refModule)`, then return the worker.
  /// The worker is released even if `fn` throws.
  template <class Fn>
  auto withWorker(Fn &&fn)
      -> decltype(fn(std::declval<InferencePlugin &>(),
                     std::declval<ModuleOp>())) {
    unsigned w = lease();
    struct Guard {
      EvaluatorPool *pool;
      unsigned w;
      ~Guard() { pool->release(w); }
    } guard{this, w};
    return fn(*workers[w].plugin, *workers[w].ref);
  }

private:
  struct Worker {
    std::unique_ptr<InferencePlugin> plugin;
    OwningOpRef<ModuleOp> ref;
  };

  unsigned lease() {
    std::lock_guard<std::mutex> g(mutex);
    unsigned w = freeWorkers.back();
    freeWorkers.pop_back();
    return w;
  }
  void release(unsigned w) {
    std::lock_guard<std::mutex> g(mutex);
    freeWorkers.push_back(w);
  }

  std::vector<Worker> workers;
  std::vector<unsigned> freeWorkers; // indices of idle workers
  std::mutex mutex;                  // guards freeWorkers
};

/// True when cinm-opt was invoked with `-o <file>` (result IR is written to
/// that file rather than stdout). In that case stdout carries no IR and is free
/// for live progress rendering.
static bool resultGoesToFile() {
  auto &opts = llvm::cl::getRegisteredOptions();
  auto it = opts.find("o");
  if (it == opts.end() || !it->second)
    return false;
  // The `-o` option registered by MlirOptMain is a cl::opt<std::string>
  // defaulting to "-" (stdout).
  auto *opt = static_cast<llvm::cl::opt<std::string> *>(it->second);
  const std::string &v = opt->getValue();
  return !v.empty() && v != "-";
}

/// Live multi-bar progress for concurrent seeds: one overall bar (seeds
/// completed) plus one bar per worker slot (current seed's evaluations). A
/// dedicated printer thread renders every ~150 ms so worker threads only touch
/// the bars' internal (mutex-guarded, non-printing in multi-progress mode)
/// setters. `indicators::DynamicProgress` renders to std::cout; that also
/// carries the result IR, so bars are only enabled when the IR is diverted to a
/// file via `-o` (see resultGoesToFile) and stdout is an interactive terminal.
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
      : active(resultGoesToFile() && ::isatty(fileno(stdout))),
        maxEvals(maxEvals) {
    if (!active)
      return;
    auto makeBar = [](size_t maxProgress, const std::string &prefix) {
      return std::make_unique<Bar>(
          indicators::option::BarWidth{30},
          indicators::option::MaxProgress{maxProgress},
          indicators::option::PrefixText{prefix},
          indicators::option::ShowPercentage{true},
          indicators::option::ShowElapsedTime{true});
    };
    bars.push_back(makeBar(static_cast<size_t>(std::max(1, nSeeds)),
                           "seeds        "));
    for (unsigned t = 0; t < cap; ++t)
      bars.push_back(makeBar(static_cast<size_t>(std::max(1, maxEvals)),
                             "  slot idle  "));
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
    bars[slot + 1]->set_option(
        indicators::option::PrefixText{"  seed " + std::to_string(seedValue)});
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

  /// Stop the printer and leave the cursor below the bars.
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

struct InferenceTask; // forward declaration for InferenceState::tryEval

struct InferenceState {
  bool anySuccess = false;
  TrialInfo bestTrial;
  double bestCost = std::numeric_limits<double>::max();
  DiagnosedSilenceableFailure err;
  int trialCount = 1;
  int budget;

  InferenceState(int maxEvals, mlir::Location loc)
      : err(mlir::emitSilenceableFailure(loc, "No candidates were evaluated")),
        budget(maxEvals) {}

  bool hasBudget() const { return budget > 0; }

  /// Evaluate the config at `poolIdx` and commit the result into shared state.
  /// The (expensive) evaluate() call runs outside any lock. When `lock` is
  /// non-null the shared-state commit is serialised through it, and `plugin`/
  /// `refOverride` supply a per-thread plugin clone and reference module so the
  /// evaluation is safe to run concurrently.
  bool tryEval(size_t poolIdx, InferenceTask &task, CandidatePool &pool,
               double &cost, size_t iter = 0,
               InferencePlugin *pluginOverride = nullptr,
               mlir::ModuleOp refOverride = nullptr, std::mutex *lock = nullptr);
};

struct InferenceTask {
  const InferenceOptions &options;
  InferencePlugin &plugin;
  ComputeBlockOp original;
  ConfigSpace space;

  OwningOpRef<ModuleOp> refModule;
  ComputeBlockOp refClone;

  std::mt19937 rng;

  InferenceTask(const InferenceOptions &options, InferencePlugin &plugin,
                cinm::ComputeBlockOp original)
      : options(options), plugin(plugin), original(original),
        rng(options.rngSeed) {

    auto [refModule, refClone] = buildRefModule(original);
    this->refClone = refClone;
    this->refModule = std::move(refModule);
    buildConfigSpace(refClone, plugin, space);
  }

  // Clone refModule to produce a fresh isolated trial module per evaluation.
  TrialInfo makeTrialInfo(Configuration config, ModuleOp refModule = nullptr) {
    if (!refModule)
      refModule = *this->refModule;
    mlir::OwningOpRef<mlir::ModuleOp> trialModule(
        llvm::cast<mlir::ModuleOp>(refModule->clone()));
    cinm::ComputeBlockOp candidate;
    trialModule->walk([&](cinm::ComputeBlockOp op) { candidate = op; });
    return TrialInfo{std::move(trialModule), candidate, std::move(config),
                     &space};
  }

  ConfWrapper wrap(const Configuration &conf) {
    return ConfWrapper(space, conf);
  }

  /// Lease an evaluator (plugin clone + reference module) for one evaluation
  /// and invoke `fn` on it. Single-seed mode leases from a parallel
  /// EvaluatorPool; a multi-seed worker binds its one dedicated per-seed clone.
  using EvalLease = std::function<bool(
      const std::function<bool(InferencePlugin &, ModuleOp)> &)>;

  /// Core Bayesian-optimisation loop for one seed over a pre-built pool and
  /// (pre-evaluated) validation set. `withEval` supplies the evaluator;
  /// `stateMx` serialises bookkeeping when the LHS phase evaluates concurrently
  /// (null when the caller is single-threaded). `onProgress` receives the
  /// running observation count after each successful evaluation. When
  /// `outBestCost` is non-null it receives the seed's best cost.
  Maybe<TrialInfo>
  runSeedBO(std::mt19937 &rng, CandidatePool &pool, ValidationSet validSet,
            const EvalLease &withEval, std::mutex *stateMx,
            unsigned sampleWorkers, const std::string &dumpDir,
            const std::function<void(int)> &onProgress = {},
            double *outBestCost = nullptr) {
    // Record the training set in its own "validation set" to output the same
    // kind of data for plotting.
    ValidationSet trainingSet(space);
    InferenceState state(options.maxEvals, refClone.getLoc());

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    std::vector<std::pair<int, double>> timings; // (nObs, elapsed_ms)
    auto recordTiming = [&]() {
      double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      timings.emplace_back(static_cast<int>(pool.nObs), ms);
    };

    // Evaluate config `idx`. `recordTraining` preserves the original behaviour
    // where LHS / surrogate picks feed the training set but cold-start
    // (nObs < 2) picks do not. tryEval runs evaluate() outside `stateMx`; the
    // bookkeeping below is guarded only when a mutex is supplied (the concurrent
    // LHS phase of single-seed mode).
    auto evalConf = [&](size_t idx, bool recordTraining) -> bool {
      double cost;
      bool evaluated = withEval([&](InferencePlugin &plug, ModuleOp ref) {
        return state.tryEval(idx, *this, pool, cost,
                             static_cast<size_t>(pool.nObs), &plug, ref,
                             stateMx);
      });
      bool success = evaluated || !options.sampleOnlyValid;
      if (success) {
        std::unique_lock<std::mutex> guard;
        if (stateMx)
          guard = std::unique_lock<std::mutex>(*stateMx);
        if (recordTraining)
          trainingSet.record(idx, cost);
        recordTiming();
        if (onProgress)
          onProgress(static_cast<int>(pool.nObs));
      }
      return success;
    };
    auto evalTrain = [&](size_t idx) { return evalConf(idx, true); };

    // Phase 1: LHS initialisation.
    int nInit = std::min(options.nInit, static_cast<int>(pool.size()));
    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Phase 1 (Generate initial population): "
               << nInit << " configs\n");
    pool.sampleInitialSet(nInit, rng, evalTrain, sampleWorkers);

    // Phase 2: surrogate-guided.
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Phase 2 (surrogate): budget="
                            << state.budget << "\n");
    while (state.hasBudget()) {
      if (pool.numVisited() >= pool.size())
        break;

      if (pool.nObs < 2) {
        // Not enough observations to fit a surrogate — pick first unvisited.
        if (size_t idx = pool.firstUnvisited(); idx >= 0)
          evalConf(idx, /*recordTraining=*/false);
        continue;
      }

      auto succeeded = pool.nextCandidateIndices(
          options, rng, evalTrain, validSet, trainingSet, pool.nObs);
      if (!succeeded)
        break;
    }

    if (!dumpDir.empty()) {
      auto dumpPath = std::filesystem::path(dumpDir);
      std::filesystem::create_directories(dumpPath);
      pool.dumpToCSV(space, options, dumpPath / "pool.csv");
      pool.dumpMetadataJSON(space, dumpPath / "space.json");
      validSet.dumpToCSV(dumpPath / "validation.csv");
      trainingSet.dumpToCSV(dumpPath / "training.csv");
      if (!timings.empty()) {
        std::ofstream timOut(dumpPath / "timings.csv");
        if (timOut) {
          timOut << "iter,elapsed_ms\n";
          for (auto [iter, ms] : timings)
            timOut << iter << "," << ms << "\n";
        }
      }
    }

    if (outBestCost)
      *outBestCost = state.bestCost;
    if (!state.anySuccess)
      return std::move(state.err);

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Best config (cost=" << state.bestCost << ")"
               << state.bestTrial.conf() << "\n");
    return std::move(state.bestTrial);
  }

  /// Worker-count knob shared by BO and exhaustive search: honour numWorkers,
  /// else hardware_concurrency, clamped to 1 when the plugin is single-threaded.
  unsigned resolveWorkers() const {
    return plugin.supportsMultithreading()
               ? (options.numWorkers > 0
                      ? options.numWorkers
                      : std::max(1u, std::thread::hardware_concurrency()))
               : 1u;
  }

  /// Run Bayesian optimization over the config space (single seed).
  /// Returns the TrialInfo from the winning evaluation — its module is
  /// fully lowered and ready for commitBestCandidate.
  Maybe<TrialInfo> runInference() {
    // Trivial: zero-dimensional space → evaluate the only possible config.
    if (space.size() == 0) {
      TrialInfo trial = makeTrialInfo({});
      auto cost = plugin.evaluate(trial);
      if (std::holds_alternative<DiagnosedSilenceableFailure>(cost))
        return std::move(std::get<DiagnosedSilenceableFailure>(cost));
      return std::move(trial);
    }

    CandidatePool pool(space, static_cast<size_t>(options.maxEvals));
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Pool: " << pool.size()
                            << " valid configs (" << pool.N << " total)\n");
    if (pool.empty())
      return emitSilenceableFailure(
          refClone.getLoc(), "No valid configurations found in search space");

    // Parallel evaluation workers. sampleInitialSet() / nextCandidateIndices()
    // dispatch accept() calls to a thread pool, so evaluation runs on a
    // per-thread plugin+module clone leased from `evalPool` while shared-state
    // commits are serialised by `stateMx`.
    unsigned nWorkers = resolveWorkers();
    EvaluatorPool evalPool(plugin, *refModule, refClone->getContext(), nWorkers);
    std::mutex stateMx; // guards state / validSet / trainingSet / timings
    EvalLease withEval =
        [&](const std::function<bool(InferencePlugin &, ModuleOp)> &fn) {
          return evalPool.withWorker(fn);
        };

    // Validation set: pre-evaluate a set of points for surrogate quality
    // tracking. NOT marked visited — BO may still select these points later.
    ValidationSet validSet(space);
    if (options.nValidation > 0) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Sampling "
                              << options.nValidation << " validation points\n");
      pool.sampleInitialSet(
          static_cast<size_t>(options.nValidation), rng,
          [&](size_t idx) {
            auto result =
                evalPool.withWorker([&](InferencePlugin &plug, ModuleOp ref) {
                  TrialInfo trial = makeTrialInfo(pool[idx], ref);
                  return plug.evaluate(trial);
                });
            if (auto *cost = std::get_if<double>(&result)) {
              std::lock_guard<std::mutex> g(stateMx);
              validSet.record(idx, *cost);
            }
            return true;
          },
          nWorkers);
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Validation set: "
                              << validSet.size() << " points\n");
    }

    return runSeedBO(rng, pool, std::move(validSet), withEval, &stateMx,
                     nWorkers, options.dumpDir);
  }

  /// Run `options.nSeeds` independent BO seeds concurrently, sharing the
  /// ConfigSpace, the valid-config scan, and the validation set. Each seed runs
  /// single-threaded on its own warmed plugin+module clone; up to
  /// `resolveWorkers()` seeds run at once. `baseDumpDir` (when non-empty)
  /// receives one `seed_<value>/` subdirectory per seed. Returns the globally
  /// best trial across all seeds for commitBestCandidate.
  Maybe<TrialInfo> runMultiSeed(const std::string &baseDumpDir) {
    // Trivial: zero-dimensional space → a single config; seeds are redundant.
    if (space.size() == 0)
      return runInference();

    // Shared valid-config scan (the expensive part), computed once.
    llvm::BitVector validMask = CandidatePool::computeValidMask(space);
    if (validMask.none())
      return emitSilenceableFailure(
          refClone.getLoc(), "No valid configurations found in search space");
    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Multi-seed: " << options.nSeeds << " seeds, "
               << validMask.count() << " valid configs\n");

    MLIRContext *ctx = refClone->getContext();
    unsigned cap =
        std::min<unsigned>(resolveWorkers(),
                           static_cast<unsigned>(std::max(1, options.nSeeds)));

    // One warmed (plugin, reference module) clone per concurrent worker slot.
    struct Evaluator {
      std::unique_ptr<InferencePlugin> plugin;
      OwningOpRef<ModuleOp> ref;
    };
    std::vector<Evaluator> evaluators(cap);
    for (unsigned t = 0; t < cap; ++t) {
      evaluators[t].plugin = plugin.clone();
      evaluators[t].plugin->warmUp(ctx);
      evaluators[t].ref =
          OwningOpRef<ModuleOp>(llvm::cast<ModuleOp>(refModule->clone()));
    }

    // Shared validation set: evaluated once, serially, on slot 0's clone.
    // Validation points are NOT marked visited, so a throwaway pool suffices.
    ValidationSet validTemplate(space);
    if (options.nValidation > 0) {
      CandidatePool samplePool(space, static_cast<size_t>(options.nValidation),
                               validMask);
      std::mt19937 vrng(options.rngSeed);
      samplePool.sampleInitialSet(
          static_cast<size_t>(options.nValidation), vrng,
          [&](size_t idx) {
            TrialInfo trial =
                makeTrialInfo(samplePool[idx], *evaluators[0].ref);
            auto result = evaluators[0].plugin->evaluate(trial);
            if (auto *cost = std::get_if<double>(&result))
              validTemplate.record(idx, *cost);
            return true;
          },
          /*workers=*/1);
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Shared validation set: "
                              << validTemplate.size() << " points\n");
    }

    // Seed values, comparable to run.py's `i*31 + offset` scheme (offset =
    // rngSeed): the k-th seed (k in [0, nSeeds)) uses (k+1)*31 + rngSeed.
    auto seedValue = [&](int k) { return (k + 1) * 31 + options.rngSeed; };

    MultiSeedProgress progress(options.nSeeds, cap, options.maxEvals);

    std::atomic<int> nextSeed{0};
    std::mutex bestMx;
    TrialInfo globalBest;
    double globalBestCost = std::numeric_limits<double>::max();
    bool anySuccess = false;
    DiagnosedSilenceableFailure firstErr =
        DiagnosedSilenceableFailure::success();

    auto worker = [&](unsigned slot) {
      Evaluator &ev = evaluators[slot];
      EvalLease withEval =
          [&](const std::function<bool(InferencePlugin &, ModuleOp)> &fn) {
            return fn(*ev.plugin, *ev.ref);
          };
      while (true) {
        int k = nextSeed.fetch_add(1, std::memory_order_relaxed);
        if (k >= options.nSeeds)
          break;
        int sv = seedValue(k);
        progress.startSeed(slot, sv);

        std::mt19937 rng(static_cast<unsigned>(sv));
        CandidatePool pool(space, static_cast<size_t>(options.maxEvals),
                           validMask);
        ValidationSet vs = validTemplate; // copy of shared contents
        std::string dir =
            baseDumpDir.empty()
                ? std::string()
                : (std::filesystem::path(baseDumpDir) /
                   ("seed_" + std::to_string(sv)))
                      .string();
        auto onProgress = [&, slot](int nObs) {
          progress.seedProgress(slot, nObs);
        };

        double bestCost = std::numeric_limits<double>::max();
        auto result = runSeedBO(rng, pool, std::move(vs), withEval,
                                /*stateMx=*/nullptr, /*sampleWorkers=*/1, dir,
                                onProgress, &bestCost);
        progress.seedDone();

        std::lock_guard<std::mutex> g(bestMx);
        if (auto *trial = std::get_if<TrialInfo>(&result)) {
          anySuccess = true;
          if (bestCost < globalBestCost) {
            globalBestCost = bestCost;
            globalBest = std::move(*trial);
          }
        } else if (firstErr.succeeded()) {
          firstErr = std::move(std::get<DiagnosedSilenceableFailure>(result));
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(cap - 1);
    for (unsigned t = 1; t < cap; ++t)
      threads.emplace_back(worker, t);
    worker(0);
    for (auto &t : threads)
      t.join();
    progress.finish();

    if (!anySuccess)
      return std::move(firstErr);
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Multi-seed best cost="
                            << globalBestCost << "\n");
    return std::move(globalBest);
  }

  /// Evaluate every valid configuration in the search space in parallel.
  /// Results are collected per-thread and merged into a pool for CSV dump.
  /// Does not commit a best candidate — returns success to skip
  /// commitBestCandidate.
  Maybe<TrialInfo> runExhaustive() {
    const size_t N = space.totalSize();
    unsigned nThreads = plugin.supportsMultithreading()
                            ? (options.numWorkers > 0
                                   ? options.numWorkers
                                   : std::max(1u, std::thread::hardware_concurrency()))
                            : 1u;
    MLIRContext *ctx = refClone->getContext();

    // Build and warm up one plugin clone per thread on the main thread.
    std::vector<std::unique_ptr<InferencePlugin>> pluginClones;
    pluginClones.reserve(nThreads);
    for (unsigned t = 0; t < nThreads; ++t) {
      pluginClones.push_back(plugin.clone());
      pluginClones.back()->warmUp(ctx);
    }

    // Build pool now: constructor pre-marks invalid configs as visited,
    // giving us the valid count before spawning threads.
    CandidatePool pool(space, N, true);
    size_t nValid = pool.size();

    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Exhaustive search: " << nValid
                            << " valid / " << N << " total configs, "
                            << nThreads << " threads\n");

    indicators::ProgressBar bar{
        indicators::option::BarWidth{40},
        indicators::option::MaxProgress{N},
        indicators::option::PrefixText{"Exhaustive search "},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::Stream{std::cerr},
    };
    // Workers only touch this relaxed counter — zero synchronisation cost.
    // A dedicated printer thread wakes every 100 ms and calls set_progress(),
    // keeping all getenv/termcolor/mutex overhead off the worker threads.
    std::atomic<size_t> barDone{0};
    std::atomic<bool> barStop{false};
    std::thread printerThread([&] {
      while (!barStop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        bar.set_progress(barDone.load(std::memory_order_relaxed));
      }
    });

    std::atomic<size_t> nextIdx{0};

    struct Obs {
      size_t idx;
      std::optional<double> cost;
      std::chrono::milliseconds eval_time;
    };
    std::vector<std::vector<Obs>> perThreadObs(nThreads);

    auto worker = [&](unsigned tid) {
      auto &myPlugin = *pluginClones[tid];
      OwningOpRef<ModuleOp> threadRef(llvm::cast<ModuleOp>(refModule->clone()));
      Configuration conf;
      while (true) {
        size_t i = nextIdx.fetch_add(1, std::memory_order_relaxed);
        if (i >= N)
          break;
        space.at(i, conf);
        barDone.fetch_add(1, std::memory_order_relaxed);
        if (!space.isValid(conf))
          continue;

        auto trial = makeTrialInfo(conf, *threadRef);
        auto t0 = std::chrono::steady_clock::now();
        auto result = myPlugin.evaluate(trial);
        auto evalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        double *cost = std::get_if<double>(&result);
        std::optional<double> opt_cost =
            cost ? std::make_optional(*cost) : std::nullopt;
        perThreadObs[tid].push_back(
            {.idx = i, .cost = opt_cost, .eval_time = evalTime});
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(nThreads - 1);
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned t = 1; t < nThreads; ++t)
      threads.emplace_back(worker, t);
    worker(0);
    for (auto &t : threads)
      t.join();
    barStop.store(true, std::memory_order_relaxed);
    printerThread.join();
    bar.mark_as_completed();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);

    // Merge per-thread observations into the pool for the CSV dump.
    size_t total = 0;
    size_t total_successful = 0;
    for (auto &obs : perThreadObs) {
      total += obs.size();
      for (auto &[idx, cost, eval_time] : obs) {
        pool.markVisited(idx);
        if (cost) {
          pool.recordObservation(idx, *cost, 0, eval_time);
          total_successful++;
        }
        // otherwise failed.
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Exhaustive: " << total_successful
               << " successful / " << total << " valid / " << N
               << " points, across " << nThreads << " threads in "
               << elapsed.count() << " ms\n");
    plugin.printStats();

    if (!options.dumpDir.empty()) {
      auto path = options.dumpDir + "/pool.csv";
      pool.dumpToCSV(space, options, path);
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-inference] Pool dumped to " << path << "\n");
    }

    return DiagnosedSilenceableFailure::success();
  }
};

bool InferenceState::tryEval(size_t poolIdx, InferenceTask &task,
                             CandidatePool &pool, double &costVal, size_t iter,
                             InferencePlugin *pluginOverride,
                             mlir::ModuleOp refOverride, std::mutex *lock) {
  InferencePlugin &plugin = pluginOverride ? *pluginOverride : task.plugin;

  // The evaluation itself runs without holding any lock so the simulator can
  // execute concurrently across worker threads.
  TrialInfo trial = task.makeTrialInfo(pool[poolIdx], refOverride);
  auto t0 = std::chrono::steady_clock::now();
  auto cost = plugin.evaluate(trial);
  auto evalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);

  // Everything below mutates shared state and must be serialised.
  std::unique_lock<std::mutex> guard;
  if (lock)
    guard = std::unique_lock<std::mutex>(*lock);

  pool.markVisited(poolIdx);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Trial #" << trialCount++ << " "
                          << task.wrap(pool[poolIdx]) << "\n");

  if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
    err = std::move(std::get<DiagnosedSilenceableFailure>(cost));
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> failed\n");
    pool.recordFailedEvaluation(poolIdx, iter);
    return false;
  }
  // only decrement budget if evaluation succeeded
  --budget;

  costVal = std::get<double>(cost);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> cost = " << costVal
                          << "\n");
  pool.recordObservation(poolIdx, costVal, iter, evalTime);
  anySuccess = true;
  if (costVal < bestCost) {
    bestCost = costVal;
    bestTrial = std::move(trial);
  }
  return true;
}

// ===----------------------------------------------------------------------===//
// inferAcceleratorConfig
// ===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts) {
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Starting inference for "
                          << computeOp.getLoc()
                          << " (maxEvals=" << opts.maxEvals
                          << ", nInit=" << opts.nInit << ")\n");

  InferenceTask task(opts, plugin, computeOp);

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Reference clone:\n";
             task.refClone->print(llvm::dbgs()); llvm::dbgs() << "\n");

  TrialInfo bestResult;
  if (opts.evalSingleSolution) {
    bestResult = task.makeTrialInfo(*opts.evalSingleSolution);
    plugin.warmUp(computeOp->getContext());
    TRY_GET(plugin.evaluate(bestResult)); //may return early
  } else if (opts.exhaustiveSearch) {
    bestResult = TRY_GET(task.runExhaustive());
  } else if (opts.nSeeds > 1) {
    // Multi-seed: shares the space / valid scan / validation set across seeds,
    // dumping each into a `seed_<value>/` subdir of opts.dumpDir. Returns the
    // globally best trial for committing.
    bestResult = TRY_GET(task.runMultiSeed(opts.dumpDir));
  } else {
    bestResult = TRY_GET(task.runInference());
  }

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Committing best config"
                          << bestResult.conf() << "\n");

  return plugin.commitBestCandidate(computeOp, std::move(bestResult));
}

static Operation *createCast(OpBuilder &builder, Location loc, Type toType,
                             Value operand) {
  if (isa<TensorType>(operand.getType()) && isa<MemRefType>(toType)) {
    return bufferization::ToBufferOp::create(builder, loc, toType, operand);
  } else if (isa<MemRefType>(operand.getType()) && isa<TensorType>(toType)) {
    return bufferization::ToTensorOp::create(builder, loc, toType, operand);
  }
  return mlir::UnrealizedConversionCastOp::create(builder, loc, toType,
                                                  operand);
}

DiagnosedSilenceableFailure
InferencePlugin::commitBestCandidate(cinm::ComputeBlockOp original,
                                     TrialInfo bestTrial) {
  // Capture the host func before we disturb the trial module. The pipeline
  // may have introduced auxiliary top-level ops (globals, kernel functions)
  // that are referenced from inside the compute block body; those need to be
  // moved into the original module alongside the body.
  auto hostFunc = bestTrial.computeBlock->getParentOfType<func::FuncOp>();

  // Move all top-level ops that the pipeline introduced into the trial module
  // (e.g. kernel functions, globals) into the original module. We skip the
  // host func wrapper — its compute block body was already taken above.
  ModuleOp originalModule = original->getParentOfType<ModuleOp>();
  auto *destBlock = originalModule.getBody();

  // We take care of renaming symbols if needed.
  // This needs to happen before we move the body of
  // the compute block to its destination.

  SymbolTable dest(originalModule);
  SymbolTable src(bestTrial.module.get());

  SmallVector<Operation *> extraOps;
  for (Operation &op : *bestTrial.module.get().getBody()) {
    if (&op == hostFunc)
      continue;
    if (op.hasTrait<SymbolOpInterface::Trait>()) {
      if (failed(src.renameToUnique(&op, {&dest})))
        LLVM_DEBUG(llvm::dbgs()
                   << "Could not rename " << op << " to unique name");
    }
    extraOps.push_back(&op);
  }
  for (auto *op : extraOps) {
    op->moveBefore(destBlock, destBlock->end());
  }

  original.getBody().takeBody(bestTrial.computeBlock.getBody());
  original.setPlatformAttr({}); // remove platform attr
  original->setAttrs(bestTrial.computeBlock->getAttrs());

  // Fix up any type mismatches introduced by bufferization.
  OpBuilder builder(original->getContext());
  for (auto [arg, opnd] : original.zipArgsWithOperands()) {
    if (arg.getType() != opnd.getType()) {
      auto innerTy = arg.getType();
      arg.setType(opnd.getType());
      builder.setInsertionPointAfterValue(arg);
      auto cast = createCast(builder, arg.getLoc(), innerTy, arg);
      arg.replaceAllUsesExcept(cast->getResult(0), cast);
    }
  }
  for (auto [res, yieldOpnd] : original.zipResultsWithYieldOperands()) {
    if (res.getType() != yieldOpnd.get().getType()) {
      builder.setInsertionPointAfterValue(yieldOpnd.get());
      auto cast =
          createCast(builder, res.getLoc(), res.getType(), yieldOpnd.get());
      yieldOpnd.set(cast->getResult(0));
    }
  }

  return DiagnosedSilenceableFailure::success();
}
} // namespace mlir::cinm
