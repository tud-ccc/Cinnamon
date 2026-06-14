#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <cstddef>
#include <cstdint>

#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
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
#include <limits>
#include <optional>
#include <random>
#include <thread>
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

size_t ConfigSpace::totalSize() const {
  size_t n = 1;
  for (const auto &p : params)
    n *= static_cast<size_t>(p.cardinality());
  return n;
}

void ConfigSpace::at(size_t idx, Configuration &conf) const {
  const size_t D = params.size();
  conf.resize(D);
  for (size_t i = D; i-- > 0;) {
    size_t card = static_cast<size_t>(params[i].cardinality());
    conf[i] = params[i].valueAt(idx % card);
    idx /= card;
  }
}

size_t ConfigSpace::indexOf(const Configuration &conf) const {
  size_t idx = 0;
  for (size_t i = 0; i < params.size(); ++i)
    idx = idx * static_cast<size_t>(params[i].cardinality()) +
          params[i].subIndexOf(conf[i]);
  return idx;
}

void ConfigSpace::neighborIndices(size_t idx,
                                  llvm::SmallVectorImpl<size_t> &result) const {
  const size_t D = params.size();
  // Compute per-dimension strides (stride[d] = product of cardinalities of
  // d+1..D-1).
  llvm::SmallVector<size_t, 8> stride(D);
  stride[D - 1] = 1;
  for (size_t i = D - 1; i-- > 0;)
    stride[i] =
        stride[i + 1] * static_cast<size_t>(params[i + 1].cardinality());

  for (size_t d = 0; d < D; ++d) {
    size_t card = static_cast<size_t>(params[d].cardinality());
    size_t subIdx = (idx / stride[d]) % card;
    if (subIdx > 0)
      result.push_back(idx - stride[d]);
    if (subIdx < card - 1)
      result.push_back(idx + stride[d]);
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

  bool tryEval(size_t poolIdx, InferenceTask &task, CandidatePool &pool,
               double &cost, size_t iter = 0);
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

  /// Run Bayesian optimization over the config space.
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

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Pool: " << (pool.size() - pool.numVisited())
               << " valid configs (" << pool.size() << " total)\n");

    if (pool.empty())
      return emitSilenceableFailure(
          refClone.getLoc(), "No valid configurations found in search space");

    // Validation set: pre-evaluate a set of points for surrogate quality
    // tracking. NOT marked visited — BO may still select these points later.
    ValidationSet validSet(space);
    if (options.nValidation > 0) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Sampling "
                              << options.nValidation << " validation points\n");
      pool.sampleInitialSet(static_cast<size_t>(options.nValidation), rng,
                            [&](size_t idx) {
                              TrialInfo trial = makeTrialInfo(pool[idx]);
                              auto result = plugin.evaluate(trial);
                              if (auto *cost = std::get_if<double>(&result))
                                validSet.record(idx, *cost);
                              return true;
                            });
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Validation set: "
                              << validSet.size() << " points\n");
    }
    // Record the training set in its own "validation set" to output
    // the same kind of data for plotting
    ValidationSet trainingSet(space);

    InferenceState state(options.maxEvals, refClone.getLoc());

    auto evalConf = [&](size_t idx) -> bool {
      double cost;
      bool success =
          state.tryEval(idx, *this, pool, cost, static_cast<int>(pool.nObs)) ||
          !options.sampleOnlyValid;
      if (success) {
        trainingSet.record(idx, cost);
      }
      return success;
    };

    // Phase 1: LHS initialisation.
    int nInit = std::min(options.nInit, static_cast<int>(pool.size()));
    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Phase 1 (Generate initial population): "
               << nInit << " configs\n");

    pool.sampleInitialSet(nInit, rng, evalConf);

    // Phase 2: surrogate-guided.
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Phase 2 (surrogate): budget="
                            << state.budget << "\n");
    while (state.hasBudget()) {
      if (pool.numVisited() >= pool.size())
        break;

      if (pool.nObs < 2) {
        // Not enough observations to fit a surrogate — pick first unvisited.
        if (size_t idx = pool.firstUnvisited(); idx >= 0) {
          double cost;
          state.tryEval(idx, *this, pool, cost, pool.nObs);
        }
        continue;
      }

      auto succeeded = pool.nextCandidateIndices(
          options, rng, evalConf, validSet, trainingSet, pool.nObs);

      if (!succeeded)
        break;
    }

    if (!options.dumpDir.empty()) {
      auto dumpPath = std::filesystem::path(options.dumpDir);
      pool.dumpToCSV(space, options, dumpPath / "pool.csv");
      validSet.dumpToCSV(dumpPath / "validation.csv");
      trainingSet.dumpToCSV(dumpPath / "training.csv");
    }

    if (!state.anySuccess)
      return std::move(state.err);

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] Best config (cost=" << state.bestCost << ")"
               << state.bestTrial.conf() << "\n");
    return std::move(state.bestTrial);
  }

  /// Evaluate every valid configuration in the search space in parallel.
  /// Results are collected per-thread and merged into a pool for CSV dump.
  /// Does not commit a best candidate — returns success to skip
  /// commitBestCandidate.
  Maybe<TrialInfo> runExhaustive() {
    const size_t N = space.totalSize();
    unsigned nThreads = std::max(1u, std::thread::hardware_concurrency()); 
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
    CandidatePool pool(space, N);
    size_t nValid = N - pool.numVisited();

    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Exhaustive search: " << nValid
                            << " valid / " << N << " total configs, "
                            << nThreads << " threads\n");

    std::atomic<size_t> nextIdx{0};

    struct Obs {
      size_t idx;
      std::optional<double> cost;
    };
    std::vector<std::vector<Obs>> perThreadObs(nThreads);

    auto worker = [&](unsigned tid) {
      auto &myPlugin = *pluginClones[tid];
      // Each thread owns its own ref module copy so per-iteration clones
      // need no synchronization.
      OwningOpRef<ModuleOp> threadRef(llvm::cast<ModuleOp>(refModule->clone()));
      Configuration conf;
      while (true) {
        size_t i = nextIdx.fetch_add(1, std::memory_order_relaxed);
        if (i >= N)
          break;
        space.at(i, conf);
        if (!space.isValid(conf))
          continue;

        auto trial = makeTrialInfo(conf, *threadRef);

        auto result = myPlugin.evaluate(trial);
        double *cost = std::get_if<double>(&result);
        std::optional<double> opt_cost =
            cost ? std::make_optional(*cost) : std::nullopt;
        perThreadObs[tid].push_back({i, opt_cost});
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
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);

    // Merge per-thread observations into the pool for the CSV dump.
    size_t total = 0;
    size_t total_successful = 0;
    for (auto &obs : perThreadObs) {
      total += obs.size();
      for (auto &[idx, cost] : obs) {
        pool.markVisited(idx);
        if (cost) {
          pool.recordObservation(idx, *cost);
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
                             CandidatePool &pool, double &costVal,
                             size_t iter) {
  --budget;
  pool.markVisited(poolIdx);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Trial #" << trialCount++ << " "
                          << task.wrap(pool[poolIdx]) << "\n");

  TrialInfo trial = task.makeTrialInfo(pool[poolIdx]);
  auto cost = task.plugin.evaluate(trial);

  if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
    err = std::move(std::get<DiagnosedSilenceableFailure>(cost));
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> failed\n");
    pool.recordFailedEvaluation(poolIdx, iter);
    return false;
  }

  costVal = std::get<double>(cost);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> cost = " << costVal
                          << "\n");
  pool.recordObservation(poolIdx, costVal, iter);
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

  auto bestResult = TRY_GET(opts.exhaustiveSearch ? task.runExhaustive()
                                                  : task.runInference());

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
  original.setAcceleratorAttr(bestTrial.computeBlock.getAcceleratorAttr());

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
