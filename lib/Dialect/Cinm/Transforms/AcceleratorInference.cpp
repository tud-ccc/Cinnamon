#include "cinm-mlir/Dialect/Cinm/Transforms/AcceleratorInference.h"
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

// ===----------------------------------------------------------------------===//
// Core framework
// ===----------------------------------------------------------------------===//

// Clone the compute op into a fresh module by cloning its nearest top-level
// ancestor (the direct child of the enclosing ModuleOp). Returns the new
// module (kept alive by the caller) and the corresponding cloned compute op.
static std::pair<mlir::OwningOpRef<mlir::ModuleOp>, cinm::ComputeOp>
cloneComputeOpToFreshModule(cinm::ComputeOp computeOp) {
  // Walk up to find the direct child of the enclosing ModuleOp.
  mlir::Operation *topLevel = computeOp.getOperation();
  while (topLevel->getParentOp() &&
         !llvm::isa<mlir::ModuleOp>(topLevel->getParentOp()))
    topLevel = topLevel->getParentOp();

  mlir::OwningOpRef<mlir::ModuleOp> newModule =
      mlir::ModuleOp::create(computeOp->getLoc());
  mlir::OpBuilder b(newModule->getBody(), newModule->getBody()->end());
  mlir::IRMapping mapping;
  auto *clonedTop = b.clone(*topLevel, mapping);

  // Locate the cloned compute op by parallel walk — IRMapping tracks values
  // and blocks but not operations, so walk order is our index.
  llvm::SmallVector<cinm::ComputeOp> origOps, clonedOps;
  topLevel->walk([&](cinm::ComputeOp op) { origOps.push_back(op); });
  clonedTop->walk([&](cinm::ComputeOp op) { clonedOps.push_back(op); });

  cinm::ComputeOp clonedComputeOp;
  for (size_t i = 0; i < origOps.size() && i < clonedOps.size(); ++i) {
    if (origOps[i] == computeOp) {
      clonedComputeOp = clonedOps[i];
      break;
    }
  }
  return {std::move(newModule), clonedComputeOp};
}

ConfigSpace buildConfigSpace(cinm::ComputeOp refClone,
                             InferencePlugin &plugin) {
  ConfigSpace space;
  plugin.initializeSpace(refClone, space);
  LLVM_DEBUG({
    llvm::dbgs() << "[cinm-inference] Config space (" << space.size()
                 << " params):\n";
    for (auto &p : space.params)
      llvm::dbgs() << "  " << p.name << " in [" << p.dlo() << ", " << p.dhi()
                   << "]\n";
  });
  return space;
}

Maybe<Configuration> runInference(cinm::ComputeOp computeOp,
                                  InferencePlugin &plugin,
                                  const ConfigSpace &space,
                                  const InferenceOptions &opts) {
  if (space.size() == 0)
    return Configuration{};

  const size_t n = space.size();
  dlib::matrix<double, 0, 1> lo, hi;
  lo.set_size(n);
  hi.set_size(n);
  for (size_t i = 0; i < n; ++i) {
    lo(i) = space[i].dlo();
    hi(i) = space[i].dhi();
  }

  bool anySuccess = false;

  DiagnosedSilenceableFailure err = mlir::emitSilenceableFailure(
      computeOp.getLoc(), "No candidates were evaluated");

  unsigned trialIdx = 0;
  auto result = dlib::find_min_global(
      [&](const dlib::matrix<double, 0, 1> &x) -> double {
        Configuration config(n);
        for (size_t i = 0; i < n; ++i)
          config[i] = space[i].discretize(x(i));

        LLVM_DEBUG({
          llvm::dbgs() << "[cinm-inference] Trial #" << trialIdx++ << ": {";
          for (size_t i = 0; i < n; ++i)
            llvm::dbgs() << space[i].name << "=" << config[i]
                         << (i + 1 < n ? ", " : "");
          llvm::dbgs() << "}\n";
        });

        auto [freshModule, clonedOp] = cloneComputeOpToFreshModule(computeOp);
        if (!clonedOp)
          return std::numeric_limits<double>::max();
        auto cost = plugin.evaluate(clonedOp, space, config);
        if (std::holds_alternative<DiagnosedSilenceableFailure>(cost)) {
          err = std::move(std::get<DiagnosedSilenceableFailure>(cost));
          LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> failed\n");
          return std::numeric_limits<double>::max();
        }
        double costVal = std::get<0>(cost);
        LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   -> cost = " << costVal
                                 << "\n");
        anySuccess = true;
        return costVal;
      },
      lo, hi, dlib::max_function_calls(opts.maxEvals));

  if (!anySuccess)
    return err;

  Configuration best(n);
  for (size_t i = 0; i < n; ++i)
    best[i] = space[i].discretize(result.x(i));
  LLVM_DEBUG({
    llvm::dbgs() << "[cinm-inference] Best config (cost=" << result.y << "): {";
    for (size_t i = 0; i < n; ++i)
      llvm::dbgs() << space[i].name << "=" << best[i]
                   << (i + 1 < n ? ", " : "");
    llvm::dbgs() << "}\n";
  });
  return best;
}

DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts) {
  // Make one reference clone. The plugin may annotate it during
  // buildConfigSpace; those annotations will be inherited by every
  // per-evaluation clone created inside runInference.
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Starting inference for "
                           << computeOp.getLoc() << " (maxEvals="
                           << opts.maxEvals << ")\n");
  auto [refModule, refClone] = cloneComputeOpToFreshModule(computeOp);
  if (!refClone)
    return emitDefiniteFailure(computeOp->getLoc(),
                               "Could not clone compute op");

  ConfigSpace space = buildConfigSpace(refClone, plugin);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Reference clone:\n";
             refClone->print(llvm::dbgs()); llvm::dbgs() << "\n");
  auto config = TRY_GET(runInference(refClone, plugin, space, opts));

  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Applying best config\n");
  return plugin.applyBestConfig(computeOp, space, config);
}

} // namespace mlir::cinm
