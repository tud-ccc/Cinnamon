#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <cstdint>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>

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

SearchParam &SearchParam::keepDivisorsOf(int64_t n) {
  if (auto *range = std::get_if<IntRange>(&domain)) {
    llvm::SmallVector<int64_t> kept;
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

// ===----------------------------------------------------------------------===//
// SearchParam factories
// ===----------------------------------------------------------------------===//

SearchParam makeRange(llvm::StringRef name, int64_t lo, int64_t hi,
                      int64_t step) {
  return {name.str(), IntRange{lo, hi, step}};
}

SearchParam makePow2Range(llvm::StringRef name, int64_t loExp, int64_t hiExp) {
  llvm::SmallVector<int64_t> vals;
  for (int64_t e = loExp; e <= hiExp; ++e)
    vals.push_back(int64_t(1) << e);
  return {name.str(), ValueList{std::move(vals)}};
}

SearchParam makeValues(llvm::StringRef name,
                       llvm::SmallVector<int64_t> values) {
  return {name.str(), ValueList{std::move(values)}};
}

// ===----------------------------------------------------------------------===//
// ConfigSpace
// ===----------------------------------------------------------------------===//

int64_t ConfigSpace::addRange(std::string name, int64_t lo, int64_t hi,
                              int64_t step) {
  return addDim(makeRange(std::move(name), lo, hi, step));
}

int64_t ConfigSpace::addPow2Range(std::string name, int64_t loExp,
                                  int64_t hiExp) {
  return addDim(makePow2Range(std::move(name), loExp, hiExp));
}

int64_t ConfigSpace::addValues(std::string name,
                               llvm::SmallVector<int64_t> values) {
  return addDim(makeValues(std::move(name), std::move(values)));
}

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

void ConfigSpace::addConstraint(Constraint constraint) {
  constraints.push_back(std::move(constraint));
}

bool ConfigSpace::isValid(const Configuration &config) const {
  auto wrapper = ConfWrapper(*this, config);
  for (auto &c : constraints)
    if (!c(wrapper))
      return false;
  return true;
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

ConfigSpace buildConfigSpace(cinm::ComputeBlockOp refClone,
                             InferencePlugin &plugin) {
  ConfigSpace space;
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
  return space;
}

// ===----------------------------------------------------------------------===//
// Candidate pool generation
// ===----------------------------------------------------------------------===//

struct ConfigHash {
  size_t operator()(const Configuration &c) const {
    size_t h = c.size();
    for (int64_t v : c)
      h ^= static_cast<size_t>(v) + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
  }
};

/// Rejection-sample up to *maxPool* distinct valid configurations from *space*.
static llvm::SmallVector<Configuration>
sampleCandidatePool(const ConfigSpace &space, size_t maxPool,
                    std::mt19937 &rng) {
  llvm::SmallVector<Configuration> pool;
  std::unordered_set<Configuration, ConfigHash> seen;
  const size_t n = space.size();

  for (size_t tries = 0, limit = maxPool * 50;
       tries < limit && pool.size() < maxPool; ++tries) {
    Configuration config(n);
    for (size_t i = 0; i < n; ++i) {
      std::uniform_real_distribution<double> dist(space[i].dlo(),
                                                  space[i].dhi());
      config[i] = space[i].discretize(dist(rng));
    }
    if (space.isValid(config) && seen.insert(config).second)
      pool.push_back(config);
  }
  return pool;
}

// ===----------------------------------------------------------------------===//
// runInference
// ===----------------------------------------------------------------------===//

Maybe<TrialInfo> runInference(mlir::ModuleOp refModule,
                              cinm::ComputeBlockOp refClone,
                              InferencePlugin &plugin, const ConfigSpace &space,
                              const InferenceOptions &opts) {
  // Clone refModule to produce a fresh isolated trial module per evaluation.
  auto makeTrialInfo = [&](Configuration config) -> TrialInfo {
    mlir::IRMapping mapping;
    mlir::OwningOpRef<mlir::ModuleOp> trialModule(
        llvm::cast<mlir::ModuleOp>(refModule->clone(mapping)));
    cinm::ComputeBlockOp candidate;
    trialModule->walk([&](cinm::ComputeBlockOp op) { candidate = op; });
    return TrialInfo{std::move(trialModule), candidate, std::move(config),
                     &space};
  };

  // Trivial: zero-dimensional space → evaluate the only possible config.
  if (space.size() == 0) {
    TrialInfo trial = makeTrialInfo({});
    auto cost = plugin.evaluate(trial);
    if (std::holds_alternative<DiagnosedSilenceableFailure>(cost))
      return std::move(std::get<DiagnosedSilenceableFailure>(cost));
    return std::move(trial);
  }

  const size_t nDims = space.size();

  // Build pool of valid candidates.
  std::mt19937 rng(42);
  const size_t maxPool =
      std::max<size_t>(500, static_cast<size_t>(opts.maxEvals) * 10);
  auto pool = sampleCandidatePool(space, maxPool, rng);

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Pool: " << pool.size()
                          << " valid configs (max " << maxPool << ")\n");

  if (pool.empty())
    return emitSilenceableFailure(
        refClone.getLoc(), "No valid configurations found in search space");

  // Encode pool into a row-major float matrix (pool.size() × nDims).
  std::vector<float> encodedPool(pool.size() * nDims);
  for (size_t i = 0; i < pool.size(); ++i) {
    auto enc = encodeConfig(space, pool[i]);
    for (size_t d = 0; d < nDims; ++d)
      encodedPool[i * nDims + d] = enc[d];
  }

  // Evaluation state.
  std::vector<bool> evaluated(pool.size(), false);
  std::vector<size_t> obsIdx;  // pool indices that were successfully evaluated
  std::vector<float> obsCosts; // cost parallel to obsIdx

  bool anySuccess = false;
  TrialInfo bestTrial;
  double bestCost = std::numeric_limits<double>::max();
  DiagnosedSilenceableFailure err = mlir::emitSilenceableFailure(
      refClone.getLoc(), "No candidates were evaluated");

  auto tryEval = [&](size_t poolIdx) {
    evaluated[poolIdx] = true;
    LLVM_DEBUG({
      const auto &config = pool[poolIdx];
      llvm::dbgs() << "[cinm-inference] Trial {";
      for (size_t i = 0; i < nDims; ++i)
        llvm::dbgs() << space[i].name << "=" << config[i]
                     << (i + 1 < nDims ? ", " : "");
      llvm::dbgs() << "}\n";
    });

    TrialInfo trial = makeTrialInfo(pool[poolIdx]);
    auto cost = plugin.evaluate(trial);

    if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
      err = std::move(std::get<DiagnosedSilenceableFailure>(cost));
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> failed\n");
      return;
    }
    float costVal = static_cast<float>(std::get<double>(cost));
    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference]   -> cost = " << costVal << "\n");
    obsIdx.push_back(poolIdx);
    obsCosts.push_back(costVal);
    anySuccess = true;
    if (costVal < bestCost) {
      bestCost = costVal;
      bestTrial = std::move(trial);
    }
    // trial (now empty after move, or non-best) is destroyed here.
  };

  // Phase 1: LHS initialisation.
  int nInit = std::min(opts.nInit, static_cast<int>(pool.size()));
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Phase 1 (LHS): " << nInit
                          << " configs\n");
  auto initIdx = lhsIndices(encodedPool, pool.size(), nDims, nInit);
  for (size_t idx : initIdx)
    tryEval(idx);

  // Phase 2: surrogate-guided.
  int budget = opts.maxEvals - static_cast<int>(obsIdx.size());
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Phase 2 (surrogate): budget="
                          << budget << "\n");
  while (budget > 0) {
    // Gather unvisited pool entries.
    std::vector<size_t> unvisited;
    std::vector<float> X_pool;
    for (size_t i = 0; i < pool.size(); ++i) {
      if (!evaluated[i]) {
        unvisited.push_back(i);
        X_pool.insert(X_pool.end(), encodedPool.begin() + i * nDims,
                      encodedPool.begin() + (i + 1) * nDims);
      }
    }
    if (unvisited.empty())
      break;

    if (obsIdx.size() < 2) {
      // Not enough observations to fit a surrogate — pick randomly.
      tryEval(unvisited[0]);
      --budget;
      continue;
    }

    // Build observation arrays.
    std::vector<float> X_obs;
    X_obs.reserve(obsIdx.size() * nDims);
    for (size_t oi : obsIdx)
      X_obs.insert(X_obs.end(), encodedPool.begin() + oi * nDims,
                   encodedPool.begin() + (oi + 1) * nDims);

    auto nextIdx = nextCandidateIndices(X_obs, obsIdx.size(), obsCosts, X_pool,
                                        unvisited.size(), nDims);
    if (nextIdx.empty())
      break;

    tryEval(unvisited[nextIdx[0]]);
    --budget;
  }

  if (!anySuccess)
    return err;

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Best config (cost=" << bestCost
                          << ")\n");
  return std::move(bestTrial);
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

  auto [refModule, refClone] = buildRefModule(computeOp);
  if (!refClone)
    return emitDefiniteFailure(computeOp->getLoc(),
                               "Could not build reference module");

  ConfigSpace space = buildConfigSpace(refClone, plugin);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Reference clone:\n";
             refClone->print(llvm::dbgs()); llvm::dbgs() << "\n");

  auto bestResult =
      runInference(refModule.get(), refClone, plugin, space, opts);
  if (std::holds_alternative<DiagnosedSilenceableFailure>(bestResult))
    return std::move(std::get<DiagnosedSilenceableFailure>(bestResult));

  auto bestTrial = std::move(std::get<TrialInfo>(bestResult));
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Committing best config:\n";
             bestTrial.computeBlock->print(llvm::dbgs());
             llvm::dbgs() << "======================\n";);

  return plugin.commitBestCandidate(computeOp, std::move(bestTrial));
}

} // namespace mlir::cinm
