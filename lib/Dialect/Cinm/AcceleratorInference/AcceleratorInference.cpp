#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "BananasSearch.h"
#include "Progress.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Utils/Permutation.h"

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
#include <llvm/Support/Format.h>
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
#include <ctime>
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

static double getThreadCpuTimeMs() {
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double getProcessCpuTimeMs() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

namespace mlir::cinm {

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

LogicalResult buildConfigSpace(cinm::ComputeBlockOp refClone,
                               InferencePlugin &plugin, ConfigSpace &space,
                               const InferenceOptions &opts) {
  SpaceBuilder builder;
  plugin.initializeSpace(refClone, builder);
  // Pins come after the plugin's declarations and constrain them; a name the
  // plugin never declared is a hard error, since running unpinned would
  // silently measure something other than what the caller asked for.
  for (const auto &entry : opts.pinnedParams)
    if (!builder.pin(entry.first(), entry.second))
      return emitError(refClone.getLoc())
             << "cannot pin '" << entry.first()
             << "': the search space declares no such integer parameter";
  builder.buildInto(space, opts.nSolveWorkers);
  LLVM_DEBUG({
    // Two different sizes, and the interesting thing about a space is the
    // ratio between them: the Cartesian product of the declared domains,
    // against what the encoding can actually address once the structural
    // constraints are folded in. The feasible count is a third number again,
    // and is only known once the predicates have been screened.
    double cartesian = 1;
    for (const SearchParam &p : space.params)
      cartesian *= p.numValues();
    const size_t feasible = space.totalSize();
    llvm::dbgs() << "[cinm-inference] Config space (" << space.numParams()
                 << " params, " << feasible << " addressable of " << cartesian
                 << " Cartesian, density" << (feasible / cartesian) << "):\n";
    for (auto &p : space.params)
      llvm::dbgs() << "  " << p.name << " in [" << p.dlo() << ", " << p.dhi()
                   << "] (" << p.numValues() << " values)\n";
  });
  return success();
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

  /// False when buildConfigSpace failed (an error has been emitted); every
  /// run method then refuses to search.
  bool spaceValid = true;

  InferenceTask(const InferenceOptions &options, InferencePlugin &plugin,
                cinm::ComputeBlockOp original)
      : options(options), plugin(plugin), original(original),
        rng(options.rngSeed) {

    auto [refModule, refClone] = buildRefModule(original);
    this->refClone = refClone;
    this->refModule = std::move(refModule);
    spaceValid = succeeded(buildConfigSpace(refClone, plugin, space, options));
    // initializeSpace is allowed to rewrite the reference in place (the UPMEM
    // plugin lowers it to linalg, so that every trial starts from the form the
    // space was read off), and a rewrite can replace the compute block op
    // itself -- canonicalization rebuilds it to drop an unused block argument.
    // Find it again rather than keeping a handle that may have been erased.
    this->refClone = nullptr;
    this->refModule->walk([&](ComputeBlockOp op) { this->refClone = op; });
    assert(this->refClone && "initializeSpace erased the reference block");
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

  /// Resolve the named values of `named` into a positional Configuration of
  /// this task's space, checking that it is complete, mentions nothing the
  /// space does not have, and names a configuration the space contains.
  ///
  /// The names are the space's *dimensions*, not its parameters, so a
  /// parameter spanning several is given one value per dimension under the
  /// names ConfigSpace::dimName reports. Every one must be given: a missing
  /// one has no defensible default, and silently picking one would produce a
  /// configuration the caller did not ask for.
  Maybe<Configuration>
  resolveNamedConfig(const llvm::StringMap<ParmValue> &named, Location loc) {
    SmallVector<std::string> dimNames;
    for (size_t d = 0; d < space.numDims(); ++d)
      dimNames.push_back(space.dimName(d));

    Configuration conf;
    conf.reserve(dimNames.size());
    SmallVector<std::string> missing;
    for (const std::string &dimName : dimNames) {
      auto it = named.find(dimName);
      if (it == named.end())
        missing.push_back(dimName);
      else
        conf.push_back(it->second);
    }
    if (!missing.empty())
      return emitDefiniteFailure(loc, "eval-solution is missing a value for: ")
             << llvm::join(missing, ", ");

    SmallVector<std::string> unknown;
    for (const auto &entry : named)
      if (!llvm::is_contained(dimNames, entry.first()))
        unknown.push_back(entry.first().str());
    if (!unknown.empty()) {
      llvm::sort(unknown);
      return emitDefiniteFailure(loc, "eval-solution names parameters this "
                                      "space does not have: ")
             << llvm::join(unknown, ", ") << "; the space declares "
             << llvm::join(dimNames, ", ");
    }

    // Membership is the whole check: the space holds exactly the feasible
    // configurations, so anything it does not contain violates a constraint.
    // Letting such a point through would have the pipeline reject it much
    // further down for a reason that reads like a lowering bug.
    if (!space.isEncodable(conf)) {
      std::string details;
      llvm::raw_string_ostream detailsOs(details);
      space.debugIsEncodable(conf, detailsOs);
      return emitDefiniteFailure(loc, "Configuration is not one this space "
                                      "contains: ")
             << wrap(conf) << "\n"
             << details;
    }
    return conf;
  }

  /// Pre-evaluate the validation set using a temporary `nWorkers`-wide pool.
  /// The pool is destroyed before returning so callers can create a
  /// correctly-sized BO pool without doubling memory. `nWorkers` is always
  /// resolveWorkers() — independent of nSeeds so validation is never
  /// artificially throttled.
  Maybe<ValidationSet> prepareBO(unsigned nWorkers) {
    if (space.totalSize() == 0)
      return emitSilenceableFailure(
          refClone.getLoc(), "No valid configurations found in search space");
    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] " << space.totalSize() << " configs\n");
    // Scoped pool: freed before the caller creates its BO-sized pool, so
    // nWorkers clones never overlap with per-seed CandidatePool allocs.
    EvaluatorPool validationPool(plugin, *refModule, refClone->getContext(),
                                 nWorkers);
    return buildValidationSet(validationPool);
  }

  /// Sample `options.nValidation` configs via LHS and evaluate them in
  /// parallel across `evalPool`, returning the resulting ValidationSet.
  ValidationSet buildValidationSet(EvaluatorPool &evalPool) {
    ValidationSet result(space);
    if (options.nValidation <= 0)
      return result;

    unsigned nWorkers = evalPool.size();
    CandidatePool samplePool(space, options.nValidation);
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
          if (auto *c = std::get_if<utils::SimCost>(&r)) {
            std::lock_guard<std::mutex> g(recordMx);
            result.record(idx, c->total());
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
    double cpuT0 = getProcessCpuTimeMs();
    std::vector<std::tuple<int, double, double>>
        timings; // (nObs, wall_ms, cpu_ms)
    auto recordTiming = [&]() {
      double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      double cpuMs = getProcessCpuTimeMs() - cpuT0;
      timings.emplace_back(static_cast<int>(pool.nObs), ms, cpuMs);
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
          timOut << "iter,elapsed_ms,cpu_ms\n";
          for (auto [iter, ms, cpu_ms] : timings)
            timOut << iter << "," << ms << "," << cpu_ms << "\n";
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

  /// Run whichever search mode `options` selects (single solution, random
  /// sample, exhaustive, or (multi-seed) BO) and return the best trial with
  /// its cost, without committing anything. Defined after the pass-mode
  /// methods below; inferAcceleratorConfig and profileComputeBlock are the
  /// two callers.
  Maybe<TrialInfo> runDispatch();

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
    if (space.numParams() == 0) {
      TrialInfo trial = makeTrialInfo({});
      auto cost = plugin.evaluate(trial);
      if (std::holds_alternative<DiagnosedSilenceableFailure>(cost))
        return std::move(std::get<DiagnosedSilenceableFailure>(cost));
      return std::move(trial);
    }

    unsigned nWorkers = resolveWorkers();
    // Validation uses a temporary nWorkers-wide pool (freed before BO starts).
    ValidationSet validSet = TRY_GET(prepareBO(nWorkers));

    // Single-seed fast path: behaves exactly like the old runInference.
    // seedValue(0) = 0 * 31 + rngSeed = rngSeed. Uses full nWorkers for LHS.
    if (options.nSeeds <= 1) {
      auto evalPool = std::make_unique<EvaluatorPool>(
          plugin, *refModule, refClone->getContext(), nWorkers);
      EvalLease withEval = [&](auto &fn) { return evalPool->withWorker(fn); };
      CandidatePool pool(space, static_cast<size_t>(options.maxEvals));
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
        CandidatePool pool(space, static_cast<size_t>(options.maxEvals));
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
        progress.seedDone(slot);

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

    CandidatePool pool(space, N, /*exhaustive=*/true);

    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Exhaustive search: " << N
                            << " configs, " << nThreads << " threads\n");

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
      uint64_t cpu_eval_time_ms;
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

        auto trial = makeTrialInfo(conf, *threadRef);
        auto t0 = std::chrono::steady_clock::now();
        double cpuT0 = getThreadCpuTimeMs();
        auto result = myPlugin.evaluate(trial);
        auto evalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        auto cpuMs = static_cast<uint64_t>(getThreadCpuTimeMs() - cpuT0);
        utils::SimCost *cost = std::get_if<utils::SimCost>(&result);
        std::optional<double> opt_cost =
            cost ? std::make_optional(cost->total()) : std::nullopt;
        perThreadObs[tid].push_back({.idx = i,
                                     .cost = opt_cost,
                                     .eval_time = evalTime,
                                     .cpu_eval_time_ms = cpuMs});
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
      for (auto &[idx, cost, eval_time, cpu_eval_time_ms] : obs) {
        pool.markVisited(idx);
        if (cost) {
          pool.recordObservation(idx, *cost, 0, eval_time, cpu_eval_time_ms);
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

    return bestObservedTrial(pool);
  }

  /// The best observation in `pool` as a TrialInfo carrying its cost. The
  /// trial's module is a fresh *unlowered* clone -- the evaluations' modules
  /// were discarded -- so this is an argmin report (what profiling consumes),
  /// not something commitBestCandidate can splice.
  Maybe<TrialInfo> bestObservedTrial(const CandidatePool &pool) {
    const std::pair<const size_t, double> *best = nullptr;
    for (const auto &entry : pool.costByIdx)
      if (!best || entry.second < best->second)
        best = &entry;
    if (!best)
      return emitSilenceableFailure(refClone.getLoc(),
                                    "No configuration evaluated successfully");
    TrialInfo trial = makeTrialInfo(pool[best->first]);
    trial.cost = best->second;
    return trial;
  }

  /// Like runExhaustive, but only evaluates a random sample of `sampleN`
  /// valid configurations instead of every valid config in the space, and
  /// rejects (without counting towards sampleN) any candidate predicted to
  /// cost more than options.sampleMaxCostMs -- a uniform-random draw over
  /// the valid space routinely turns up configs whose actual on-hardware
  /// cost is minutes instead of milliseconds, which is wasteful once every
  /// sampled config gets compiled and benchmarked downstream. Uses
  /// CandidatePool::sampleInitialSet (Latin Hypercube Sampling, already
  /// parallelized internally -- see its own doc comment) rather than
  /// fillRandom precisely so the accept/reject decision can happen inside
  /// the sampling loop itself: a rejected candidate is immediately replaced
  /// by another LHS draw instead of being sampled once ahead of time.
  ///
  /// Exhaustive search's cost is entirely its simulator calls, one per
  /// configuration in the space -- so evaluating a bounded random subset
  /// instead turns an O(N) sweep (hours, for spaces with hundreds of thousands
  /// of configurations) into roughly an O(sampleN) one (seconds to low
  /// minutes, depending how much sampleMaxCostMs ends up rejecting).
  Maybe<TrialInfo> runRandomSample(size_t sampleN) {
    unsigned nThreads =
        plugin.supportsMultithreading()
            ? (options.numWorkers > 0
                   ? options.numWorkers
                   : std::max(1u, std::thread::hardware_concurrency()))
            : 1u;
    MLIRContext *ctx = refClone->getContext();

    CandidatePool pool(space, sampleN);

    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Random sample: requesting "
                            << sampleN << " / " << pool.size()
                            << " configs (max cost " << options.sampleMaxCostMs
                            << " ms), " << nThreads << " threads\n");

    indicators::ProgressBar bar{
        indicators::option::BarWidth{40},
        indicators::option::MaxProgress{sampleN},
        indicators::option::PrefixText{"Random sample search "},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::Stream{std::cerr},
    };
    std::atomic<bool> barStop{false};
    std::thread printerThread([&] {
      while (!barStop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        bar.set_progress(pool.numVisited());
      }
    });

    std::mutex poolMutex;
    std::atomic<size_t> nAttempted{0};
    std::atomic<size_t> nRejected{0};
    double maxCostMs = options.sampleMaxCostMs;

    // Invoked concurrently on sampleInitialSet's own thread pool -- a fixed
    // set of persistent worker threads, so thread_local here gives every
    // worker its own InferencePlugin clone + module clone, created once on
    // first use and reused for every later candidate that thread evaluates
    // (mirrors runExhaustive's explicit per-thread clone vector, just
    // without needing a stable numeric thread index to index into it).
    auto accept = [&](size_t idx) -> bool {
      thread_local std::unique_ptr<InferencePlugin> tlsPlugin = [&] {
        auto p = plugin.clone();
        p->warmUp(ctx);
        return p;
      }();
      thread_local OwningOpRef<ModuleOp> tlsRef(
          llvm::cast<ModuleOp>(refModule->clone()));

      nAttempted.fetch_add(1, std::memory_order_relaxed);
      Configuration conf;
      space.at(idx, conf);
      auto trial = makeTrialInfo(conf, *tlsRef);
      auto t0 = std::chrono::steady_clock::now();
      double cpuT0 = getThreadCpuTimeMs();
      auto result = tlsPlugin->evaluate(trial);
      auto evalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0);
      auto cpuMs = static_cast<uint64_t>(getThreadCpuTimeMs() - cpuT0);

      utils::SimCost *cost = std::get_if<utils::SimCost>(&result);
      if (!cost || (maxCostMs && cost->total() > maxCostMs)) {
        // Rejected (or failed to evaluate): not recorded, so it never shows
        // up in the dump (dumpFullPool is off by default) and doesn't count
        // towards sampleN -- sampleInitialSet's own `used` bookkeeping
        // already ensures this exact candidate is never retried.
        nRejected.fetch_add(1, std::memory_order_relaxed);
        return false;
      }

      std::lock_guard<std::mutex> guard(poolMutex);
      pool.markVisited(idx);
      pool.recordObservation(idx, cost->total(), 0, evalTime, cpuMs);
      return true;
    };

    auto t0 = std::chrono::steady_clock::now();
    pool.sampleInitialSet(sampleN, rng, accept, nThreads);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);

    barStop.store(true, std::memory_order_relaxed);
    printerThread.join();
    bar.mark_as_completed();

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Random sample: " << pool.numVisited()
               << " / " << sampleN << " accepted (<= " << maxCostMs << " ms), "
               << nRejected.load() << " rejected / " << nAttempted.load()
               << " attempted, across " << nThreads << " threads in "
               << elapsed.count() << " ms\n");
    plugin.printStats();

    if (!options.dumpDir.empty()) {
      pool.dumpToCSV(space, options, options.dumpDir + "/pool.csv");
      pool.dumpMetadataJSON(space, options.dumpDir + "/space.json");
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Pool and metadata dumped to "
                              << options.dumpDir << "\n");
    }

    return bestObservedTrial(pool);
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
  double cpuT0 = getThreadCpuTimeMs();
  auto cost = plugin.evaluate(trial);
  auto evalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);
  auto cpuEvalMs = static_cast<uint64_t>(getThreadCpuTimeMs() - cpuT0);

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

  costVal = std::get<utils::SimCost>(cost).total();
  if (log)
    *log << "[cinm-inference]   -> cost = " << costVal << "\n";
  pool.recordObservation(poolIdx, costVal, iter, evalTime, cpuEvalMs);
  anySuccess = true;
  if (costVal < bestCost) {
    bestCost = costVal;
    trial.cost = costVal;
    bestTrial = std::move(trial);
  }
  return true;
}

// ===----------------------------------------------------------------------===//
// inferAcceleratorConfig
// ===----------------------------------------------------------------------===//

Maybe<TrialInfo> InferenceTask::runDispatch() {
  if (!spaceValid)
    return emitDefiniteFailure(original.getLoc(),
                               "the search space could not be built");

  if (options.evalSingleSolution) {
    // Resolve the named parameters against the space that was just built.
    Configuration conf = TRY_GET(
        resolveNamedConfig(*options.evalSingleSolution, original->getLoc()));
    TrialInfo bestResult = makeTrialInfo(std::move(conf));
    plugin.warmUp(original->getContext());
    auto estimate = TRY_GET(plugin.evaluate(bestResult)); // may return early
    bestResult.cost = estimate.total();
    llvm::errs() << "Estimated cost: " << llvm::format("%.3f", estimate.total())
                 << " ms\n";
    estimate.forEachEntry(
        [&](utils::CostCategory category, StringRef label, double value) {
          llvm::errs() << "  " << utils::costCategoryName(category);
          if (!label.empty())
            llvm::errs() << "." << label;
          llvm::errs() << ": " << llvm::format("%.3f", value) << " ms\n";
        });
    return bestResult;
  }
  if (options.sampleN > 0)
    return runRandomSample(options.sampleN);
  if (options.exhaustiveSearch)
    return runExhaustive();
  // Multi-seed: shares the space / valid scan / validation set across seeds,
  // dumping each into a `seed_<value>/` subdir of options.dumpDir. Returns
  // the globally best trial for committing.
  return runMultiSeed(options.dumpDir);
}

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

  TrialInfo bestResult = TRY_GET(task.runDispatch());

  // Exhaustive and random-sample runs are data collection: their result
  // reports the argmin but was never lowered, so there is nothing to commit.
  if (!opts.evalSingleSolution && (opts.sampleN > 0 || opts.exhaustiveSearch))
    return DiagnosedSilenceableFailure::success();

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Committing best config"
                          << bestResult.conf() << "\n");

  return plugin.commitBestCandidate(computeOp, std::move(bestResult));
}

// ===----------------------------------------------------------------------===//
// profileComputeBlock (Stage A)
// ===----------------------------------------------------------------------===//

Maybe<SmallVector<ProfilePoint>>
profileComputeBlock(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                    const InferenceOptions &opts) {
  StringRef param = plugin.sharedResourceParam();
  SmallVector<int64_t> menu = plugin.sharedResourceMenu();
  if (param.empty() || menu.empty())
    return emitDefiniteFailure(
        computeOp.getLoc(),
        "this target declares no shared resource to profile over");

  SmallVector<ProfilePoint> points;
  for (int64_t resource : menu) {
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Profiling " << param << "="
                            << resource << "\n");
    InferenceOptions pointOpts = opts;
    pointOpts.pinnedParams[param] = static_cast<ParmValue>(resource);
    if (!opts.dumpDir.empty())
      pointOpts.dumpDir =
          opts.dumpDir + "/" + param.str() + "_" + std::to_string(resource);

    InferenceTask task(pointOpts, plugin, computeOp);
    Maybe<TrialInfo> result = task.runDispatch();
    if (auto *fail = std::get_if<DiagnosedSilenceableFailure>(&result)) {
      if (fail->isDefiniteFailure())
        return std::move(*fail);
      // An infeasible menu value (divisibility, capacity) is data, not an
      // error: the profile simply has no point there.
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-inference]   no point at " << param << "="
                 << resource << ": " << fail->getMessage() << "\n");
      (void)fail->silence();
      continue;
    }

    TrialInfo &best = std::get<TrialInfo>(result);
    ProfilePoint point{resource, best.cost, {}};
    for (size_t dim = 0; dim < task.space.numDims(); ++dim)
      point.config[task.space.dimName(dim)] = best.config[dim];
    points.push_back(std::move(point));
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   L(" << resource
                            << ") = " << best.cost << " ms\n");
  }

  if (points.empty())
    return emitSilenceableFailure(computeOp.getLoc())
           << "no value of '" << param
           << "' in the allocation menu is feasible for this block";
  return points;
}

static Operation *createCast(OpBuilder &builder, Location loc, Type toType,
                             Value operand) {
  if (isa<TensorType>(operand.getType()) && isa<MemRefType>(toType)) {
    return bufferization::ToBufferOp::create(builder, loc, toType, operand);
  } else if (isa<MemRefType>(operand.getType()) && isa<TensorType>(toType)) {
    return bufferization::ToTensorOp::create(builder, loc, toType, operand,
                                             true);
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
