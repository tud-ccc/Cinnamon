#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "BananasSearch.h"
#include "Progress.h"
#include "SearchStrategy.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Utils/Permutation.h"

#include <llvm/ADT/ScopeExit.h>

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
#include <condition_variable>
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
  for (auto [i, operand] : llvm::enumerate(computeOp->getOperands())) {
    mapping.map(operand, entry->getArgument(i));
    // Staticness is a property of the *original* operands (weights reachable
    // from annotated parameters, constants); the trial modules only see the
    // host func's arguments, so forward it onto them. isStaticValue then
    // resolves inside any trial exactly as it would in the original module.
    if (isStaticValue(operand))
      hostFunc.setArgAttr(i, CinmDialect::STATIC_ATTR_NAME, b.getUnitAttr());
  }

  auto *cloned = b.clone(*computeOp, mapping);
  mlir::func::ReturnOp::create(b, loc, cloned->getResults());

  return {std::move(module), llvm::cast<cinm::ComputeBlockOp>(cloned)};
}

LogicalResult buildConfigSpace(cinm::ComputeBlockOp refClone,
                               InferencePlugin &plugin, ConfigSpace &space,
                               const InferenceOptions &opts) {
  auto buildStart = std::chrono::steady_clock::now();
  auto recordBuildTime = llvm::scope_exit([&] {
    space.buildWallSeconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - buildStart)
                                 .count();
  });
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
  /// Blocks until a worker is free, so callers may outnumber the pool: the
  /// pool is the concurrency limit, not the caller count. The worker is
  /// released even if `fn` throws.
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
    std::unique_lock<std::mutex> g(mutex);
    available.wait(g, [&] { return !freeWorkers.empty(); });
    unsigned w = freeWorkers.back();
    freeWorkers.pop_back();
    return w;
  }
  void release(unsigned w) {
    {
      std::lock_guard<std::mutex> g(mutex);
      freeWorkers.push_back(w);
    }
    available.notify_one();
  }

  std::vector<Worker> workers;
  std::vector<unsigned> freeWorkers; // indices of idle workers
  std::mutex mutex;                  // guards freeWorkers
  std::condition_variable available; // signalled on release
};

struct InferenceTask; // forward declaration for InferenceState::tryEval

/// Consecutive failed evaluations, before any has succeeded, after which a
/// search gives the point up as infeasible.
///
/// A failed evaluation does not spend budget -- it is a candidate the
/// constraint system admitted and the lowering then rejected, which is not
/// an observation. That is right for a point where most configurations
/// work, and unbounded for one where none does: phase 1 draws until nInit
/// candidates pass, phase 2 until the budget is spent, and neither
/// condition can be reached, so the search walks the whole space. A 512MB
/// gemm on 64 DPUs is such a point -- the space offers 14715
/// configurations there and every one fails to lower -- and it stalled the
/// whole-program arm indefinitely.
///
/// Four is deliberately small. It only ever fires before the first success,
/// where the question is not "is this point good" but "does anything here
/// lower at all", and four independent draws failing answers that about as
/// well as four hundred. A point that recovers after three failures keeps
/// its whole budget.
static constexpr int kInfeasibleFailStreak = 4;

/// Failures a point may accumulate, as a multiple of its evaluation budget,
/// before it is given up whatever it has managed to evaluate.
///
/// kInfeasibleFailStreak only fires before the first success, so it says
/// nothing about a point that is feasible but low-yield -- and that is the
/// expensive case, because neither phase counts failures against anything.
/// Phase 1 draws until nInit candidates *pass* and phase 2 until the budget
/// is *spent*, so the work both do scales with the reciprocal of the success
/// rate. Profiling a 512MB gemm at 64 DPUs is 2.4% yield: collecting 64 +
/// 256 successes there cost 10499 failed evaluations, around two thirds of
/// everything that config's search did, for a device size the allocator
/// cannot choose for a class that size.
///
/// Four budgets' worth is deliberately loose. A point at 33% yield -- still
/// perfectly usable, and the allocator does pick those -- spends about two
/// failures per success and stays well inside it, so the cap only bites
/// where the yield is bad enough that the estimate was going to be thin
/// regardless. A point stopped this way keeps the observations it has: it is
/// profiled from fewer samples, not dropped.
static constexpr int kFailBudgetFactor = 4;

struct InferenceState {
  bool anySuccess = false;
  /// Failures since the last success, for kInfeasibleFailStreak. Atomic
  /// because abandoned() is read by the search loops while evaluation
  /// threads are committing results.
  std::atomic<int> failStreak{0};
  /// Failures over the whole point, against failBudget. Both phases harvest
  /// successes, so without this nothing bounds the work a low-yield point
  /// does -- see kFailBudgetFactor.
  std::atomic<int> failCount{0};
  int failBudget;
  TrialInfo bestTrial;
  double bestCost = std::numeric_limits<double>::max();
  DiagnosedSilenceableFailure err;
  int trialCount = 1;
  int budget;
  llvm::raw_ostream *log = nullptr; // per-seed log stream; null = silent

  InferenceState(int maxEvals, mlir::Location loc)
      : failBudget(std::max(1, maxEvals) * kFailBudgetFactor),
        err(mlir::emitSilenceableFailure(loc, "No candidates were evaluated")),
        budget(maxEvals) {}

  /// Whether this point has been given up: either nothing has evaluated and
  /// the failures have run on long enough to call it infeasible, or the
  /// failures have outrun the budget by enough that harvesting the rest of
  /// the successes is not worth what it costs (see kFailBudgetFactor).
  bool abandoned() const {
    if (failCount.load(std::memory_order_relaxed) >= failBudget)
      return true;
    return !anySuccess &&
           failStreak.load(std::memory_order_relaxed) >= kInfeasibleFailStreak;
  }

  bool hasBudget() const { return budget > 0 && !abandoned(); }

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

    if (options.stampConfigs) {
      // The space is built on the original itself: the parameter names land
      // on the original's ops, and the reference (and every trial cloned from
      // it) inherits them, so the winning configuration can be resolved back
      // onto the original at commit time without any op correspondence
      // maintained on the side. The module is already in the plugin's
      // converted form -- the pass ran the conversion once, up front -- so
      // initializeSpace rewrites nothing and `original` stays valid.
      spaceValid =
          succeeded(buildConfigSpace(original, plugin, space, options));
      auto [refModule, refClone] = buildRefModule(original);
      this->refClone = refClone;
      this->refModule = std::move(refModule);
      return;
    }

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
    // further down for a reason that reads like a lowering bug -- except
    // when that is the point: under evalSolutionForce the lowering's own
    // verdict on an infeasible configuration is the measurement.
    if (!space.isEncodable(conf)) {
      if (options.evalSolutionForce) {
        llvm::errs() << "eval-solution-force: configuration is outside the "
                        "feasible set, attempting the lowering anyway\n";
        return conf;
      }
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
    CandidatePool samplePool(space, options.nValidation, options);
    std::mt19937 vrng(options.rngSeed);
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Sampling validation set: "
                            << options.nValidation << " points\n");
    {
      SimpleProgressBar validBar(static_cast<size_t>(options.nValidation),
                                 "  validation ", options.showProgress);
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
  /// `evalWorkers` is how many evaluations may run at once, which both the LHS
  /// phase and a surrogate round's batch use, and `stateMx` serialises the
  /// bookkeeping they share (null when the caller is single-threaded, in which
  /// case `evalWorkers` must be 1). `onProgress` receives the running
  /// observation count after each successful evaluation. When `outBestCost` is
  /// non-null it receives the seed's best cost.
  Maybe<TrialInfo> runSeedBO(std::mt19937 &rng, CandidatePool &pool,
                             ValidationSet validSet, const EvalLease &withEval,
                             std::mutex *stateMx, unsigned evalWorkers,
                             const std::string &dumpDir,
                             const std::function<void(int)> &onProgress = {},
                             double *outBestCost = nullptr,
                             llvm::raw_ostream *log = nullptr) {
    // Record the training set in its own "validation set" to output the same
    // kind of data for plotting.
    ValidationSet trainingSet(space);
    // The search policy for Phase 2. Everything around it -- init sampling,
    // budget accounting, evaluation bookkeeping, dumps -- is shared by every
    // strategy, so runs differing only in options.searchStrategy are directly
    // comparable.
    std::unique_ptr<SearchStrategy> strategy =
        makeSearchStrategy(pool, validSet, trainingSet);
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
    // LHS explicitly, which is what this call has always used: it is the
    // parameter before `abort`, not options.samplingMode (that one belongs
    // to the census draw, which is a different sample for a different
    // purpose).
    pool.sampleInitialSet(nInit, rng, evalTrain, evalWorkers,
                          InferenceOptions::SamplingMode::LHS,
                          [&state] { return state.abandoned(); });

    // Phase 2: surrogate-guided.
    if (log)
      *log << "[cinm-inference] Phase 2 (surrogate): budget=" << state.budget
           << " batch=" << options.boBatchSize << " workers=" << evalWorkers
           << "\n";
    // Counts surrogate-guided rounds, which is not the observation count: a
    // round costs one fit and may spend more than one evaluation.
    int round = 0;
    while (state.hasBudget()) {
      if (pool.numVisited() >= pool.size())
        break;

      if (pool.nObs < 2) {
        // Not enough observations to fit a surrogate — pick first unvisited.
        if (size_t idx = pool.firstUnvisited(); idx != pool.N)
          evalConf(idx, /*recordTraining=*/false);
        continue;
      }

      // Never overshoot the budget: the batch is what the round will spend.
      size_t batch = std::min<size_t>(std::max<size_t>(options.boBatchSize, 1),
                                      static_cast<size_t>(state.budget));
      auto accepted = strategy->step(rng, evalTrain, round++, pool.nObs, batch,
                                     evalWorkers);
      if (accepted == 0)
        break;
    }

    if (!dumpDir.empty()) {
      auto dumpPath = std::filesystem::path(dumpDir);
      std::filesystem::create_directories(dumpPath);
      pool.dumpToCSV(space, options, dumpPath / "pool.csv", strategy.get());
      pool.dumpMetadataJSON(space, dumpPath / "space.json");
      validSet.dumpToCSV(dumpPath / "validation.csv");
      trainingSet.dumpToCSV(dumpPath / "training.csv");
      strategy->dumpDiagnostics(dumpPath);
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
  /// min(nWorkers, nSeeds) seeds run concurrently and the workers are split
  /// between them, so that fewer seeds than workers still occupies the
  /// machine: a seed is parallel inside itself as well as against its peers.
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
      CandidatePool pool(space, static_cast<size_t>(options.maxEvals), options);
      std::mutex stateMx;
      llvm::raw_ostream *log = nullptr;
      LLVM_DEBUG(log = &llvm::dbgs());
      std::mt19937 seedRng(static_cast<unsigned>(options.rngSeed));
      return runSeedBO(seedRng, pool, std::move(validSet), withEval, &stateMx,
                       nWorkers, baseDumpDir, {}, nullptr, log);
    }

    // Multi-seed: run nSeeds BO seeds concurrently, capped by nWorkers.
    unsigned cap = std::min<unsigned>(
        nWorkers, static_cast<unsigned>(std::max(1, options.nSeeds)));

    // One lease pool is shared by every seed, and each seed may have up to
    // boBatchSize evaluations in flight (a round never dispatches more).
    // withWorker blocks past the pool's capacity, so the pool -- not a
    // per-seed split -- is the concurrency limit, and it is work-conserving:
    // a seed stalled on one slow simulation or busy fitting its surrogate
    // donates its idle leases to whichever seed has work, instead of holding
    // its share of the machine hostage to its own round barrier. The pool is
    // never larger than the seeds can submit, so no plugin clone is built
    // only to sit idle; a thread blocked on a lease costs nothing.
    const unsigned perSeedCeiling =
        std::max<unsigned>(1u, static_cast<unsigned>(options.boBatchSize));
    const unsigned poolSize = std::min(nWorkers, cap * perSeedCeiling);

    auto evalPool = std::make_unique<EvaluatorPool>(
        plugin, *refModule, refClone->getContext(), poolSize);
    EvalLease withEval = [&](auto &fn) { return evalPool->withWorker(fn); };
    LLVM_DEBUG({
      llvm::dbgs() << "[cinm-inference] Multi-seed: " << options.nSeeds
                   << " seeds, cap=" << cap << ", " << poolSize
                   << " shared evaluators, up to " << perSeedCeiling
                   << " in flight per seed\n";
    });

    // k * 31 + rngSeed: k=0 gives rngSeed (matches single-seed path above).
    auto seedValue = [&](int k) { return k * 31 + options.rngSeed; };

    MultiSeedProgress progress(options.nSeeds, cap, options.maxEvals,
                               options.showProgress);
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
                           options);
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
        // This seed's own bookkeeping lock. Every structure runSeedBO touches
        // -- the pool, the state, the training set -- belongs to this seed, so
        // seeds never contend with each other, only with their own evaluators.
        std::mutex seedMx;
        const unsigned seedWorkers = perSeedCeiling;
        auto result = runSeedBO(seedRng, pool, std::move(vs), withEval,
                                seedWorkers > 1 ? &seedMx : nullptr,
                                seedWorkers, dir, onProgress, &bestCost, log);
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

    CandidatePool pool(space, N, options);

    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Exhaustive search: " << N
                            << " configs, " << nThreads << " threads\n");

    SimpleProgressBar bar(N, "Exhaustive search ", options.showProgress);

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
        bar.tick();

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
    bar.finish();
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
  /// CandidatePool::sampleInitialSet (already parallelized internally -- see
  /// its own doc comment) rather than fillRandom precisely so the
  /// accept/reject decision can happen inside the sampling loop itself: a
  /// rejected candidate is immediately replaced by another draw instead of
  /// being sampled once ahead of time.
  ///
  /// options.samplingMode picks how that draw is made. LHS spreads the sample
  /// over the space, which is what a model wants to be fit on; Uniform is the
  /// only mode whose sample is an unbiased picture of the space, so it is the
  /// one to use when the sample feeds a statistic about the space rather than
  /// a model. Note that sampleMaxCostMs conditions the sample either way, so
  /// an unbiased draw wants it at 0.
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

    CandidatePool pool(space, sampleN, options);

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Random sample: requesting " << sampleN
               << " / " << pool.size() << " configs ("
               << (options.samplingMode == InferenceOptions::SamplingMode::LHS
                       ? "lhs"
                       : "uniform")
               << ", max cost " << options.sampleMaxCostMs << " ms), "
               << nThreads << " threads\n");

    SimpleProgressBar bar(sampleN, "Random sample search ",
                          options.showProgress);

    std::mutex poolMutex;
    std::atomic<size_t> nAttempted{0};
    // Split by reason, because the reasons do different things to the sample.
    // An over-budget candidate is replaced by another draw, so it conditions
    // the sample on the accepted region; a failed evaluation does the same but
    // on a region nobody chose. Both have to be reportable for the sample to
    // be describable as a draw from anything. A timed-out candidate is neither:
    // it is kept, with a non-finite cost standing in for the value the
    // simulator did not reach.
    std::atomic<size_t> nFailed{0};
    std::atomic<size_t> nOverBudget{0};
    std::atomic<size_t> nTimedOut{0};
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
        (cost ? nOverBudget : nFailed).fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      // A simulation that ran out of its eval-timeout-ms budget yields a
      // non-finite cost rather than no cost, so the candidate keeps its slot
      // in the sample: the draw stays a draw over the whole space, and the
      // row still reaches the downstream compile+bench that supplies its
      // measured runtime. Only the prediction is missing, and the census says
      // for how many rows.
      if (!std::isfinite(cost->total()))
        nTimedOut.fetch_add(1, std::memory_order_relaxed);

      std::lock_guard<std::mutex> guard(poolMutex);
      pool.markVisited(idx);
      pool.recordObservation(idx, cost->total(), 0, evalTime, cpuMs);
      bar.tick();
      return true;
    };

    auto t0 = std::chrono::steady_clock::now();
    pool.sampleInitialSet(sampleN, rng, accept, nThreads, options.samplingMode);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);

    bar.finish();

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Random sample: " << pool.numVisited()
               << " / " << sampleN << " accepted (<= " << maxCostMs << " ms), "
               << nTimedOut.load() << " timed out, " << nOverBudget.load()
               << " over budget, " << nFailed.load() << " failed / "
               << nAttempted.load() << " attempted, across " << nThreads
               << " threads in " << elapsed.count() << " ms\n");
    plugin.printStats();

    if (!options.dumpDir.empty()) {
      pool.dumpToCSV(space, options, options.dumpDir + "/pool.csv");
      pool.dumpMetadataJSON(space, options.dumpDir + "/space.json");
      // The census of what the draw did, as a sibling of the pool it
      // describes. Everything here except `timed_out` is unrecoverable from
      // pool.csv -- a rejected candidate leaves no row -- and a sample whose
      // rejections are unknown cannot be described as a draw from the space,
      // so this file is what lets the downstream reporting state the bound it
      // states.
      std::string statsPath = options.dumpDir + "/sample_stats.json";
      std::ofstream stats(statsPath);
      if (stats) {
        stats << "{\n"
              << "  \"requested\": " << sampleN << ",\n"
              << "  \"accepted\": " << pool.numVisited() << ",\n"
              << "  \"attempted\": " << nAttempted.load() << ",\n"
              << "  \"timed_out\": " << nTimedOut.load() << ",\n"
              << "  \"over_budget\": " << nOverBudget.load() << ",\n"
              << "  \"failed\": " << nFailed.load() << ",\n"
              << "  \"space_size\": " << space.totalSize() << ",\n"
              << "  \"sampling_mode\": \""
              << (options.samplingMode == InferenceOptions::SamplingMode::LHS
                      ? "lhs"
                      : "uniform")
              << "\",\n"
              << "  \"max_cost_ms\": " << maxCostMs << "\n"
              << "}\n";
      }
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
      *log << "[cinm-inference]   -> failed: " << err.getMessage() << "\n";
    pool.recordFailedEvaluation(poolIdx, iter);
    const int streak = failStreak.fetch_add(1, std::memory_order_relaxed) + 1;
    const int failures = failCount.fetch_add(1, std::memory_order_relaxed) + 1;
    if (log && streak == kInfeasibleFailStreak && !anySuccess)
      *log << "[cinm-inference] " << kInfeasibleFailStreak
           << " consecutive failures with nothing evaluated: giving this "
              "point up as infeasible\n";
    else if (log && failures == failBudget)
      // The evaluation budget, not what is left of it: `budget` is spent
      // down as successes land.
      *log << "[cinm-inference] " << failures
           << " failed evaluations against an evaluation budget of "
           << failBudget / kFailBudgetFactor
           << ": giving this point up, profiled from what it evaluated\n";
    return false;
  }
  // only decrement budget if evaluation succeeded
  --budget;
  failStreak.store(0, std::memory_order_relaxed);

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

    // Stamping writes the configuration onto the op as attributes and leaves
    // the lowering to --upmem-lower-stamped, so nothing here ever reads the
    // lowered module: evaluating it would price a compilation that is thrown
    // away. That is the whole commit cost of a graph solve -- one lowering
    // per member, where a transformer has hundreds of members sharing a
    // dozen configurations already lowered and priced during profiling.
    //
    // What is given up is the check that this configuration lowers at all.
    // In a graph solve it was checked: the configuration is a profile point
    // of this block's own class, and class members are the same program by
    // construction. A hand-written eval-solution finds out in the lowering
    // pass instead of here.
    if (options.stampConfigs)
      return bestResult;

    plugin.warmUp(original->getContext());
    auto estimate = TRY_GET(plugin.evaluate(bestResult)); // may return early
    bestResult.cost = estimate.total();
    llvm::errs() << "Estimated cost: " << llvm::format("%.3f", estimate.total())
                 << " ms\n";
    estimate.forEachEntry([&](utils::CostCategory category, StringRef label,
                              double value, bool excluded) {
      llvm::errs() << "  " << utils::costCategoryName(category);
      if (!label.empty())
        llvm::errs() << "." << label;
      llvm::errs() << ": " << llvm::format("%.3f", value) << " ms";
      // Reported but not part of the estimate above: amortized over the
      // serving lifetime rather than paid per inference.
      if (excluded)
        llvm::errs() << " (excluded)";
      llvm::errs() << "\n";
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

  // Space-dump mode: the space was built by the constructor above; write it
  // out and stop. No evaluation, no commit -- the IR stays untouched, which
  // is the point (see InferenceOptions::dumpSpaceOnly).
  if (opts.dumpSpaceOnly) {
    if (!task.spaceValid)
      return emitDefiniteFailure(computeOp.getLoc(),
                                 "the search space could not be built");
    if (opts.dumpDir.empty())
      return emitDefiniteFailure(
          computeOp.getLoc(),
          "dump-space-only without dump-dir would build the space and "
          "write it nowhere; pass dump-dir");
    dumpSpaceJSON(task.space, task.space.totalSize(),
                  std::filesystem::path(opts.dumpDir) / "space.json");
    return DiagnosedSilenceableFailure::success();
  }

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Reference clone:\n";
             task.refClone->print(llvm::dbgs()); llvm::dbgs() << "\n");

  TrialInfo bestResult = TRY_GET(task.runDispatch());

  // Exhaustive and random-sample runs are data collection: their result
  // reports the argmin but was never lowered, so there is nothing to commit.
  if (!opts.evalSingleSolution && (opts.sampleN > 0 || opts.exhaustiveSearch))
    return DiagnosedSilenceableFailure::success();

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Committing best config"
                          << bestResult.conf() << "\n");

  if (opts.stampConfigs)
    return plugin.stampBestCandidate(computeOp, bestResult);
  return plugin.commitBestCandidate(computeOp, std::move(bestResult));
}

// ===----------------------------------------------------------------------===//
// profileComputeBlock
// ===----------------------------------------------------------------------===//

Maybe<SmallVector<ProfilePoint>>
profileComputeBlock(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                    const InferenceOptions &opts,
                    SmallVectorImpl<ProfileSample> *samples,
                    ProfileGate *gate) {
  StringRef param = plugin.sharedResourceParam();
  if (param.empty())
    return emitDefiniteFailure(
        computeOp.getLoc(),
        "this target declares no shared resource to profile over");
  SmallVector<int64_t> menu = plugin.sharedResourceMenu(computeOp);
  if (menu.empty())
    return emitSilenceableFailure(computeOp.getLoc())
           << "no profiling menu could be derived for this block";

  // The menu points are independent searches, so they run concurrently. Each
  // point gets its own plugin clone (initializeSpace mutates the plugin) and
  // builds its own modules; the original block is only ever read. Gated on
  // the same conditions as every other parallel evaluation here: the plugin
  // must tolerate concurrent evaluation, and the context must have its
  // thread-safe uniquing on.
  // Repeats are independent searches of the same pinned space, so they join
  // the menu points as ordinary units of work: the sweep is over
  // (menu point, seed) pairs, and the outer parallelism covers both.
  const unsigned nSeeds = std::max(1, opts.profileSeeds);
  const size_t nJobs = menu.size() * nSeeds;

  MLIRContext *ctx = computeOp->getContext();
  const unsigned baseWorkers =
      opts.numWorkers ? opts.numWorkers
                      : std::max(1u, std::thread::hardware_concurrency());
  const bool threaded =
      plugin.supportsMultithreading() && ctx->isMultithreadingEnabled();

  // Two different numbers, and conflating them is what leaves cores idle.
  //
  // The per-point worker budget is derived from the MENU alone. It decides
  // how parallel one search is, and a search's trajectory depends on it, so
  // it must not move when repeats are asked for: otherwise seed 0 of a
  // 3-seed sweep would be a different search from a 1-seed sweep's only
  // search, and the repeats would stop being a measurement of an unchanged
  // experiment.
  const unsigned menuThreads =
      threaded ? std::min<unsigned>(menu.size(), baseWorkers) : 1;
  InferenceOptions pointBase = opts;
  // A gated sweep shares the machine with its siblings, so its searches are
  // single-threaded and the gate decides how many run: parallelism comes from
  // the number of points in flight, which is the level that still has work
  // when one sweep is nearly done.
  pointBase.numWorkers = gate ? 1 : std::max(1u, baseWorkers / menuThreads);
  // Progress bars from concurrent searches would interleave.
  pointBase.showProgress = false;

  // How many jobs run at once is a separate question, because a search does
  // not sustain its whole budget: a surrogate round dispatches boBatchSize
  // evaluations and waits for them, so past that the workers only hold
  // clones while idling. Sizing the sweep by the budget therefore parks
  // (budget - boBatchSize) cores per point for most of the run. Fill them by
  // overlapping more jobs instead -- one more menu point, or one of the
  // repeats, which would otherwise wait for a whole search to finish.
  //
  // When the budget is the binding constraint rather than the batch this
  // works out to menuThreads exactly, i.e. to what the sweep did before.
  const unsigned perJob =
      std::min(pointBase.numWorkers,
               std::max(1u, static_cast<unsigned>(opts.boBatchSize)));
  unsigned outerWorkers = 1;
  if (threaded)
    // Gated: offer everything, since a thread waiting on a permit costs
    // nothing and is what lets this sweep take the machine once its siblings
    // finish. Ungated: fill the machine from this sweep alone.
    outerWorkers =
        gate ? std::min<size_t>(nJobs, gate->capacity())
             : std::min<size_t>(nJobs, std::max(1u, baseWorkers / perJob));

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] sweep: " << nJobs
                          << " jobs over " << outerWorkers << " thread(s), "
                          << pointBase.numWorkers << " worker(s) per search"
                          << (gate ? ", gated" : "") << "\n");

  // One slot per (menu value, seed), so the profile comes out in menu order
  // whatever the finish order. Slot i*nSeeds is the seed the profile keeps.
  struct Slot {
    std::optional<ProfilePoint> point;
    std::optional<DiagnosedSilenceableFailure> fail;
  };
  std::vector<Slot> slots(nJobs);

  auto runPoint = [&](size_t job) {
    const size_t i = job / nSeeds, seed = job % nSeeds;
    const int64_t resource = menu[i];
    // Everything a search allocates -- the plugin clone, the trial modules --
    // is held under the permit, so a waiting job costs a stack and nothing
    // else.
    struct Permit {
      ProfileGate *gate;
      explicit Permit(ProfileGate *g) : gate(g) {
        if (gate)
          gate->acquire();
      }
      ~Permit() {
        if (gate)
          gate->release();
      }
    } permit(gate);
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Profiling " << param << "="
                            << resource << " seed " << seed << "\n");
    std::unique_ptr<InferencePlugin> pointPlugin = plugin.clone();
    InferenceOptions pointOpts = pointBase;
    pointOpts.pinnedParams[param] = static_cast<ParmValue>(resource);
    // Same derivation as the multi-seed BO engine's seedValue(), so repeat 0
    // is rngSeed itself and reproduces the single-seed run exactly.
    pointOpts.rngSeed = static_cast<int>(seed) * 31 + opts.rngSeed;
    if (!opts.dumpDir.empty()) {
      pointOpts.dumpDir =
          opts.dumpDir + "/" + param.str() + "_" + std::to_string(resource);
      if (nSeeds > 1)
        pointOpts.dumpDir += "/seed_" + std::to_string(pointOpts.rngSeed);
    }

    InferenceTask task(pointOpts, *pointPlugin, computeOp);
    Maybe<TrialInfo> result = task.runDispatch();
    if (auto *fail = std::get_if<DiagnosedSilenceableFailure>(&result)) {
      slots[job].fail = std::move(*fail);
      return;
    }

    TrialInfo &best = std::get<TrialInfo>(result);
    ProfilePoint point{resource, best.cost, {}, {}};
    for (size_t dim = 0; dim < task.space.numDims(); ++dim)
      point.config[task.space.dimName(dim)] = best.config[dim];
    // Residency is measured on a fresh unlowered clone: depending on the
    // search mode, `best`'s own module may already be lowered past the form
    // the plugin can read tile parameters from. Only the kept seed needs it;
    // the repeats exist to be compared on cost.
    if (seed == 0) {
      TrialInfo probe = task.makeTrialInfo(best.config);
      point.residency = pointPlugin->measureResidency(probe);
      // The search only kept the incumbent's total; the transfer-bound gate
      // and profiles.csv want its breakdown, so price it once more (the
      // simulator is deterministic, so this reproduces costMs). A fresh
      // trial module: evaluate() lowers what it is given.
      TrialInfo breakdownProbe = task.makeTrialInfo(best.config);
      auto priced = pointPlugin->evaluate(breakdownProbe);
      if (auto *cost = std::get_if<utils::SimCost>(&priced)) {
        const double total = cost->total();
        if (std::isfinite(total) && total > 0)
          point.transferShare =
              (cost->categoryTotal(utils::CostCategory::Transfer) +
               cost->categoryTotal(utils::CostCategory::TransferBack)) /
              total;
      }
    }
    slots[job].point = std::move(point);
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   L(" << resource
                            << ") = " << best.cost << " ms\n");
  };

  if (outerWorkers <= 1) {
    for (size_t job = 0; job < nJobs; ++job)
      runPoint(job);
  } else {
    std::atomic<size_t> next{0};
    auto worker = [&] {
      for (size_t job = next.fetch_add(1, std::memory_order_relaxed);
           job < nJobs; job = next.fetch_add(1, std::memory_order_relaxed))
        runPoint(job);
    };
    std::vector<std::thread> threads;
    threads.reserve(outerWorkers - 1);
    for (unsigned t = 1; t < outerWorkers; ++t)
      threads.emplace_back(worker);
    worker();
    for (std::thread &t : threads)
      t.join();
  }

  // Harvest in menu order. A definite failure wins over everything (its
  // diagnostic is already emitted); an infeasible menu value (divisibility,
  // capacity) is data, not an error -- the profile simply has no point there.
  std::optional<DiagnosedSilenceableFailure> definite;
  SmallVector<ProfilePoint> points;
  for (auto [job, slot] : llvm::enumerate(slots)) {
    const size_t i = job / nSeeds, seed = job % nSeeds;
    if (slot.fail) {
      if (slot.fail->isDefiniteFailure() && !definite) {
        definite = std::move(*slot.fail);
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-inference]   no point at " << param << "=" << menu[i]
                 << ": " << slot.fail->getMessage() << "\n");
      (void)slot.fail->silence();
      continue;
    }
    if (!slot.point)
      continue;
    if (samples)
      samples->push_back(
          {menu[i], static_cast<unsigned>(seed), slot.point->costMs});
    // Only the first repeat reaches the profile, so what the allocator solves
    // over is exactly what a single-seed run would have handed it.
    if (seed == 0)
      points.push_back(std::move(*slot.point));
  }
  if (definite)
    return std::move(*definite);

  if (points.empty())
    return emitSilenceableFailure(computeOp.getLoc())
           << "no value of '" << param
           << "' in the allocation menu is feasible for this block";

  // Lower-envelope repair (see InferenceOptions::profileRepair). Points are
  // in menu order, i.e. ascending resource; a running argmin over the
  // measured costs replaces any point a stalled seed left above the envelope
  // with the best smaller point's incumbent. The residency travels with the
  // configuration -- it describes what actually runs -- and rawCostMs keeps
  // the measurement.
  for (ProfilePoint &p : points)
    p.rawCostMs = p.costMs;
  if (opts.profileRepair) {
    const ProfilePoint *best = nullptr;
    for (ProfilePoint &p : points) {
      if (best && best->costMs < p.costMs) {
        p.costMs = best->costMs;
        p.config = best->config;
        p.residency = best->residency;
        p.repairedFrom = best->resource;
        LLVM_DEBUG(llvm::dbgs()
                   << "[cinm-inference]   repaired L(" << p.resource
                   << ") = " << p.rawCostMs << " -> " << p.costMs << " (from "
                   << param << "=" << best->resource << ")\n");
      } else {
        best = &p;
      }
    }
  }
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

/// Whether every use of `root`, following view-like aliases (subview,
/// expand_shape, ...), at most READS the buffer according to declared memory
/// effects. Conservative: an aliasing user without MemoryEffectOpInterface,
/// or with a write effect on the value, means "possibly written".
///
/// This is what licenses `read_only` on the bufferization.to_buffer casts
/// the commit inserts below. The spliced body is already-bufferized code
/// inside a tensor-land compute block, so one-shot-bufferize cannot re-run
/// its own conflict analysis on it -- it sees only the to_buffer boundary
/// and, without the attribute, must assume the buffer is mutated and copy
/// the operand defensively (ComputeBufferizableInterface's isValueWritten
/// walks exactly to this cast). The effects, unlike the tensor SSA graph,
/// survive lowering, so they are the honest source of the answer.
static bool onlyReadsBuffer(Value root) {
  SmallVector<Value> worklist{root};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() == use.get()) {
          llvm::append_range(worklist, user->getResults());
          continue;
        }
      }
      auto mem = dyn_cast<MemoryEffectOpInterface>(user);
      if (!mem)
        return false;
      SmallVector<MemoryEffects::EffectInstance> effects;
      mem.getEffectsOnValue(use.get(), effects);
      for (const MemoryEffects::EffectInstance &effect : effects)
        if (isa<MemoryEffects::Write>(effect.getEffect()))
          return false;
    }
  }
  return true;
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
      // An operand the body only reads (a scattered input, per the transfer
      // ops' declared effects) must say so on its cast, or the enclosing
      // function's bufferization pays an alloc+copy for it (see
      // onlyReadsBuffer). Checked after the use replacement above, since it
      // is the cast's uses that carry the answer.
      if (auto toBuffer = dyn_cast<bufferization::ToBufferOp>(cast))
        if (onlyReadsBuffer(toBuffer.getResult()))
          toBuffer.setReadOnly(true);
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
