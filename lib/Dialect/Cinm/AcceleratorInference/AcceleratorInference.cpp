#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "BananasSearch.h"
#include "Progress.h"
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
  auto withWorker(Fn &&fn) -> decltype(fn(std::declval<InferencePlugin &>(),
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

struct InferenceTask; // forward declaration for InferenceState::tryEval

struct InferenceState {
  bool anySuccess = false;
  TrialInfo bestTrial;
  double bestCost = std::numeric_limits<double>::max();
  DiagnosedSilenceableFailure err;
  int trialCount = 1;
  int budget;
  llvm::raw_ostream *log = nullptr; // per-seed log stream; null = silent

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
               mlir::ModuleOp refOverride = nullptr,
               std::mutex *lock = nullptr);
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

  /// Shared state produced by prepareBO() and consumed by both runInference and
  /// runMultiSeed.
  struct BOSetup {
    std::shared_ptr<llvm::BitVector> validMask;
    std::shared_ptr<std::vector<size_t>> validIndices;
    ValidationSet validSet;
  };

  /// Compute the valid-config mask and pre-evaluate the validation set using a
  /// temporary `nWorkers`-wide pool. The pool is destroyed before returning so
  /// callers can create a correctly-sized BO pool without doubling memory.
  /// `nWorkers` is always resolveWorkers() — independent of nSeeds so
  /// validation is never artificially throttled.
  Maybe<BOSetup> prepareBO(unsigned nWorkers) {

    auto validMask = std::make_shared<llvm::BitVector>();
    auto validIndices = std::make_shared<std::vector<size_t>>();

    CandidatePool::computeValidMask(space, *validMask, *validIndices);
    if (validIndices->empty())
      return emitSilenceableFailure(
          refClone.getLoc(), "No valid configurations found in search space");
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] " << validIndices->size()
                            << " valid configs\n");
    ValidationSet validSet = [&] {
      // Scoped pool: freed before the caller creates its BO-sized pool,
      // so nWorkers clones never overlap with per-seed CandidatePool allocs.
      EvaluatorPool validationPool(plugin, *refModule, refClone->getContext(),
                                   nWorkers);
      return buildValidationSet(validMask, validIndices, validationPool);
    }();
    return BOSetup{std::move(validMask), std::move(validIndices),
                   std::move(validSet)};
  }

  /// Sample `options.nValidation` configs via LHS and evaluate them in
  /// parallel across `evalPool`, returning the resulting ValidationSet.
  ValidationSet
  buildValidationSet(std::shared_ptr<llvm::BitVector> validMask,
                     std::shared_ptr<std::vector<size_t>> validIndices,
                     EvaluatorPool &evalPool) {
    ValidationSet result(space);
    if (options.nValidation <= 0)
      return result;

    unsigned nWorkers = evalPool.size();
    CandidatePool samplePool(space, options.nValidation, std::move(validMask),
                             std::move(validIndices));
    std::mt19937 vrng(options.rngSeed);
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Sampling validation set: "
                            << options.nValidation << " points\n");
    {
      SimpleProgressBar validBar(static_cast<size_t>(options.nValidation),
                                 "  validation ");
      // Phase 1: LHS candidate selection (serial, cheap — O(n×M) NN search).
      std::vector<size_t> validIdxs;
      validIdxs.reserve(options.nValidation);
      samplePool.sampleInitialSet(
          static_cast<size_t>(options.nValidation), vrng,
          [&](size_t idx) {
            validIdxs.push_back(idx);
            return true;
          },
          /*workers=*/1);

      // Phase 2: evaluate in parallel via the shared pool.
      std::mutex recordMx;
      std::atomic<size_t> nextValIdx{0};
      auto evalOne = [&]() {
        for (;;) {
          size_t pos = nextValIdx.fetch_add(1, std::memory_order_relaxed);
          if (pos >= validIdxs.size())
            break;
          size_t idx = validIdxs[pos];
          auto r =
              evalPool.withWorker([&](InferencePlugin &plug, ModuleOp ref) {
                auto trial = makeTrialInfo(samplePool[idx], ref);
                return plug.evaluate(trial);
              });
          if (auto *c = std::get_if<double>(&r)) {
            std::lock_guard<std::mutex> g(recordMx);
            result.record(idx, *c);
          }
          validBar.tick();
        }
      };
      std::vector<std::thread> valThreads;
      valThreads.reserve(nWorkers - 1);
      for (unsigned t = 1; t < nWorkers; ++t)
        valThreads.emplace_back(evalOne);
      evalOne();
      for (auto &t : valThreads)
        t.join();
    }
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Validation set: "
                            << result.size() << " points\n");
    return result;
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
  Maybe<TrialInfo> runSeedBO(std::mt19937 &rng, CandidatePool &pool,
                             ValidationSet validSet, const EvalLease &withEval,
                             std::mutex *stateMx, unsigned sampleWorkers,
                             const std::string &dumpDir,
                             const std::function<void(int)> &onProgress = {},
                             double *outBestCost = nullptr,
                             llvm::raw_ostream *log = nullptr) {
    // Record the training set in its own "validation set" to output the same
    // kind of data for plotting.
    ValidationSet trainingSet(space);
    InferenceState state(options.maxEvals, refClone.getLoc());
    state.log = log;

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
    // bookkeeping below is guarded only when a mutex is supplied (the
    // concurrent LHS phase of single-seed mode).
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
    if (log)
      *log << "[cinm-inference] Phase 1 (Generate initial population): "
           << nInit << " configs\n";
    pool.sampleInitialSet(nInit, rng, evalTrain, sampleWorkers);

    // Phase 2: surrogate-guided.
    if (log)
      *log << "[cinm-inference] Phase 2 (surrogate): budget=" << state.budget
           << "\n";
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

    if (log) {
      *log << "Finished inference\n"
           << "- " << pool.nObs << " / " << pool.visited.size()
           << " successful trials\n";
      *log << "[cinm-inference] Best config (cost=" << state.bestCost << ")"
           << state.bestTrial.conf() << "\n";
    }
    return std::move(state.bestTrial);
  }

  /// Worker-count knob shared by BO and exhaustive search: honour numWorkers,
  /// else hardware_concurrency, clamped to 1 when the plugin is
  /// single-threaded.
  unsigned resolveWorkers() const {
    return plugin.supportsMultithreading()
               ? (options.numWorkers > 0
                      ? options.numWorkers
                      : std::max(1u, std::thread::hardware_concurrency()))
               : 1u;
  }

  /// Run Bayesian optimisation (single seed). Delegates to runMultiSeed with
  /// nSeeds=1, which uses all available workers for the BO LHS phase and
  /// seeds the RNG directly with options.rngSeed (seedValue(0) = 0*31+rngSeed).
  Maybe<TrialInfo> runInference() { return runMultiSeed(options.dumpDir); }

  /// Unified BO entry point for one or more seeds.
  ///
  /// Validation and evaluator warm-up always use resolveWorkers() — never
  /// capped by nSeeds. When nSeeds <= 1, all workers go to the single seed's
  /// LHS phase (identical to the old runInference). When nSeeds > 1, up to
  /// min(nWorkers, nSeeds) seeds run concurrently, each single-threaded.
  ///
  /// Seed RNG: seedValue(k) = k * 31 + rngSeed, so k=0 → rngSeed exactly.
  Maybe<TrialInfo> runMultiSeed(const std::string &baseDumpDir) {
    // Trivial: zero-dimensional space → evaluate the single possible config.
    if (space.size() == 0) {
      TrialInfo trial = makeTrialInfo({});
      auto cost = plugin.evaluate(trial);
      if (std::holds_alternative<DiagnosedSilenceableFailure>(cost))
        return std::move(std::get<DiagnosedSilenceableFailure>(cost));
      return std::move(trial);
    }

    unsigned nWorkers = resolveWorkers();
    // Validation uses a temporary nWorkers-wide pool (freed before BO starts).
    auto [validMask, validIndices, validSet] = TRY_GET(prepareBO(nWorkers));

    // Single-seed fast path: behaves exactly like the old runInference.
    // seedValue(0) = 0 * 31 + rngSeed = rngSeed. Uses full nWorkers for LHS.
    if (options.nSeeds <= 1) {
      auto evalPool = std::make_unique<EvaluatorPool>(
          plugin, *refModule, refClone->getContext(), nWorkers);
      EvalLease withEval = [&](auto &fn) { return evalPool->withWorker(fn); };
      CandidatePool pool(space, static_cast<size_t>(options.maxEvals),
                         std::move(validMask), std::move(validIndices));
      std::mutex stateMx;
      llvm::raw_ostream *log = nullptr;
      LLVM_DEBUG(log = &llvm::dbgs());
      std::mt19937 seedRng(static_cast<unsigned>(options.rngSeed));
      return runSeedBO(seedRng, pool, std::move(validSet), withEval, &stateMx,
                       nWorkers, baseDumpDir, {}, nullptr, log);
    }

    // Multi-seed: run nSeeds BO seeds concurrently, capped by nWorkers.
    // Pool is sized to cap (not nWorkers) — one slot per concurrent seed.
    unsigned cap = std::min<unsigned>(
        nWorkers, static_cast<unsigned>(std::max(1, options.nSeeds)));
    auto evalPool = std::make_unique<EvaluatorPool>(
        plugin, *refModule, refClone->getContext(), cap);
    EvalLease withEval = [&](auto &fn) { return evalPool->withWorker(fn); };
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Multi-seed: " << options.nSeeds
                            << " seeds, cap=" << cap << "\n");

    // k * 31 + rngSeed: k=0 gives rngSeed (matches single-seed path above).
    auto seedValue = [&](int k) { return k * 31 + options.rngSeed; };

    MultiSeedProgress progress(options.nSeeds, cap, options.maxEvals);
    if (!progress.active)
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] (no progress bar)\n");

    std::atomic<int> nextSeed{0};
    std::mutex bestMx;
    TrialInfo globalBest;
    double globalBestCost = std::numeric_limits<double>::max();
    bool anySuccess = false;
    DiagnosedSilenceableFailure firstErr =
        DiagnosedSilenceableFailure::success();

    auto worker = [&](unsigned slot) {
      while (true) {
        int k = nextSeed.fetch_add(1, std::memory_order_relaxed);
        if (k >= options.nSeeds)
          break;
        int sv = seedValue(k);
        progress.startSeed(slot, sv);

        std::mt19937 seedRng(static_cast<unsigned>(sv));
        CandidatePool pool(space, static_cast<size_t>(options.maxEvals),
                           validMask, validIndices);
        ValidationSet vs = validSet; // copy of shared contents
        std::string dir = baseDumpDir.empty()
                              ? std::string()
                              : (std::filesystem::path(baseDumpDir) /
                                 ("seed_" + std::to_string(sv)))
                                    .string();
        auto onProgress = [&, slot](int nObs) {
          progress.seedProgress(slot, nObs);
        };

        // Per-seed log file; fall back to llvm::dbgs() when no dump dir.
        std::unique_ptr<llvm::raw_fd_ostream> logFile;
        llvm::raw_ostream *log = nullptr;
        if (!dir.empty()) {
          std::filesystem::create_directories(dir);
          std::error_code ec;
          logFile =
              std::make_unique<llvm::raw_fd_ostream>(dir + "/seed.log", ec);
          if (!ec)
            log = logFile.get();
        }
        LLVM_DEBUG(if (!log) log = &llvm::dbgs());

        double bestCost = std::numeric_limits<double>::max();
        // Each seed is single-threaded; outer cap-parallelism covers all cores.
        auto result = runSeedBO(seedRng, pool, std::move(vs), withEval,
                                /*stateMx=*/nullptr, /*sampleWorkers=*/1, dir,
                                onProgress, &bestCost, log);
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
    unsigned nThreads =
        plugin.supportsMultithreading()
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
    auto pool = CandidatePool::build(space, N, true);
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
  if (log)
    *log << "[cinm-inference] Trial #" << trialCount << " "
         << task.wrap(pool[poolIdx]) << "\n";
  ++trialCount;

  if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
    err = std::move(std::get<DiagnosedSilenceableFailure>(cost));
    if (log)
      *log << "[cinm-inference]   -> failed\n";
    pool.recordFailedEvaluation(poolIdx, iter);
    return false;
  }
  // only decrement budget if evaluation succeeded
  --budget;

  costVal = std::get<double>(cost);
  if (log)
    *log << "[cinm-inference]   -> cost = " << costVal << "\n";
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
    TRY_GET(plugin.evaluate(bestResult)); // may return early
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
