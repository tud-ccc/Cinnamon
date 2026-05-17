#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <dlib/global_optimization.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cmath>
#include <limits>

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

// ===----------------------------------------------------------------------===//
// ConfigSpace
// ===----------------------------------------------------------------------===//

void ConfigSpace::addRange(std::string name, int64_t lo, int64_t hi,
                           int64_t step) {
  params.push_back({std::move(name), IntRange{lo, hi, step}});
}

void ConfigSpace::addPow2Range(std::string name, int64_t loExp, int64_t hiExp) {
  llvm::SmallVector<int64_t> vals;
  for (int64_t e = loExp; e <= hiExp; ++e)
    vals.push_back(int64_t(1) << e);
  params.push_back({std::move(name), ValueList{std::move(vals)}});
}

void ConfigSpace::addValues(std::string name,
                            llvm::SmallVector<int64_t> values) {
  params.push_back({std::move(name), ValueList{std::move(values)}});
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

// Clone the full parent ModuleOp into a sandbox. Returns the sandbox (kept
// alive by the caller) and the corresponding cloned compute op inside it.
static std::pair<mlir::OwningOpRef<mlir::ModuleOp>, cinm::ComputeBlockOp>
cloneModuleToSandbox(cinm::ComputeBlockOp computeOp) {
  auto parentModule = computeOp->getParentOfType<mlir::ModuleOp>();
  if (!parentModule)
    return {nullptr, nullptr};

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> sandbox(
      llvm::cast<mlir::ModuleOp>(parentModule->clone(mapping)));

  // Locate the cloned compute op by parallel walk — IRMapping tracks values
  // and blocks but not operations, so walk order is our index.
  llvm::SmallVector<cinm::ComputeBlockOp> origOps, clonedOps;
  parentModule->walk([&](cinm::ComputeBlockOp op) { origOps.push_back(op); });
  sandbox->walk([&](cinm::ComputeBlockOp op) { clonedOps.push_back(op); });

  cinm::ComputeBlockOp refClone;
  for (size_t i = 0; i < origOps.size() && i < clonedOps.size(); ++i) {
    if (origOps[i] == computeOp) {
      refClone = clonedOps[i];
      break;
    }
  }
  return {std::move(sandbox), refClone};
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

Maybe<cinm::ComputeBlockOp> runInference(cinm::ComputeBlockOp refClone,
                                         InferencePlugin &plugin,
                                         const ConfigSpace &space,
                                         const InferenceOptions &opts) {
  // Helper: clone refClone and insert the candidate right after it.
  auto makeCandidate = [&]() {
    mlir::OpBuilder builder(refClone->getContext());
    builder.setInsertionPointAfter(refClone.getOperation());
    return llvm::cast<cinm::ComputeBlockOp>(
        builder.clone(*refClone.getOperation()));
  };

  if (space.size() == 0) {
    auto candidate = makeCandidate();
    auto cost = plugin.evaluate(candidate, space, {});
    if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
      plugin.disposeCandidate(candidate);
      return std::move(std::get<DiagnosedSilenceableFailure>(cost));
    }
    return candidate;
  }

  const size_t n = space.size();
  dlib::matrix<double, 0, 1> lo, hi;
  lo.set_size(n);
  hi.set_size(n);
  for (size_t i = 0; i < n; ++i) {
    lo(i) = space[i].dlo();
    hi(i) = space[i].dhi();
  }

  bool anySuccess = false;
  cinm::ComputeBlockOp bestCandidate;
  double bestCost = std::numeric_limits<double>::max();

  DiagnosedSilenceableFailure err = mlir::emitSilenceableFailure(
      refClone.getLoc(), "No candidates were evaluated");

  dlib::global_function_search search(dlib::function_spec(lo, hi));

  // Drive the optimizer manually so constraint-rejected configs don't consume
  // the evaluation budget.  Each call to get_next_x() must be answered with
  // req.set() before the next call; unanswered requests are harmlessly dropped
  // by dlib's destructor but we always answer them to keep the surrogate model
  // informed.
  unsigned trialIdx = 0;
  for (int validCount = 0, totalCount = 0;
       validCount < opts.maxEvals && totalCount < opts.maxEvals * 10;
       ++totalCount) {
    auto req = search.get_next_x();

    Configuration config(n);
    for (size_t i = 0; i < n; ++i)
      config[i] = space[i].discretize(req.x()(i));

    LLVM_DEBUG({
      llvm::dbgs() << "[cinm-inference] Trial #" << trialIdx++ << ": {";
      for (size_t i = 0; i < n; ++i)
        llvm::dbgs() << space[i].name << "=" << config[i]
                     << (i + 1 < n ? ", " : "");
      llvm::dbgs() << "}\n";
    });

    if (!space.isValid(config)) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> skipped (constraint violated)\n");
      req.set(std::numeric_limits<double>::max());
      continue;
    }
    ++validCount;

    auto candidate = makeCandidate();
    auto cost = plugin.evaluate(candidate, space, config);
    if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
      err = std::move(std::get<DiagnosedSilenceableFailure>(cost));
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> failed\n");
      plugin.disposeCandidate(candidate);
      req.set(std::numeric_limits<double>::max());
      continue;
    }
    double costVal = std::get<0>(cost);
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> cost = " << costVal << "\n");
    anySuccess = true;
    req.set(costVal);
    if (costVal < bestCost) {
      if (bestCandidate)
        plugin.disposeCandidate(bestCandidate);
      bestCandidate = candidate;
      bestCost = costVal;
    } else {
      plugin.disposeCandidate(candidate);
    }
  }

  if (!anySuccess)
    return err;

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Best candidate (cost="
                           << bestCost << ")\n");
  return bestCandidate;
}

DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts) {
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Starting inference for "
                           << computeOp.getLoc() << " (maxEvals="
                           << opts.maxEvals << ")\n");

  // Clone the full parent module into a sandbox. The plugin may annotate
  // refClone during buildConfigSpace; those annotations propagate to every
  // per-trial candidate cloned from it.
  auto [sandbox, refClone] = cloneModuleToSandbox(computeOp);
  if (!refClone)
    return emitDefiniteFailure(computeOp->getLoc(),
                               "Could not clone compute op");

  ConfigSpace space = buildConfigSpace(refClone, plugin);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Reference clone:\n";
             refClone->print(llvm::dbgs()); llvm::dbgs() << "\n");

  auto bestCandidate = TRY_GET(runInference(refClone, plugin, space, opts));

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Committing best candidate\n");
  return plugin.commitBestCandidate(computeOp, bestCandidate);
  // sandbox goes out of scope here, destroying remaining sandbox contents.
}

} // namespace mlir::cinm
