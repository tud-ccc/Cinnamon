#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <cstdint>
#include <functional>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <optional>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMINFERACCELERATORPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {
using mlir::cinm::utils::Maybe;

// ===----------------------------------------------------------------------===//
// UpmemInferencePlugin
// ===----------------------------------------------------------------------===//

struct UpmemInferencePlugin : cinm::InferencePlugin {
  upmem::UpmemPlatformAttr platform;
  std::unique_ptr<UpmemSimulator> simulator;
  unsigned trialCount = 0;

  static constexpr llvm::StringLiteral kTileParamNamesAttr =
      "upmem.tile_param_names";
  static constexpr llvm::StringLiteral kKernelModuleAttr =
      "upmem.kernel_module";

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), simulator(std::move(sim)) {}

  // --- InferencePlugin interface ---

  cinm::Constraint configurationValid(
      std::function<bool(UpmemAcceleratorAttr, const cinm::ConfWrapper &)>
          constraint) const;

  std::optional<cinm::Constraint>
  tilingConstraint(cinm::CinmTilingInterface op,
                   llvm::SmallVectorImpl<StringRef> &tilingFactorNames) const;

  void initializeSpace(cinm::ComputeBlockOp refClone,
                       cinm::ConfigSpace &space) override {
    space.addRange("ranks", 1, platform.getMaxNumRanks());
    space.addRange("dpus", 1, platform.getMaxNumDpusPerRank());
    space.addRange("tasklets", 1, platform.getMaxNumTasklets());

    auto nameInventor = cinm::utils::NameInventor::getNameInventor(
        refClone.getOperation(), "tile_");
    MLIRContext *ctx = refClone->getContext();

    refClone.getBody().walk([&](mlir::Operation *op) {
      auto tileable = llvm::dyn_cast<cinm::CinmTilingInterface>(op);
      if (!tileable)
        return;

      llvm::SmallVector<int64_t> dimSizes;
      tileable.getTilableDimSizes(dimSizes);

      llvm::SmallVector<StringRef> paramNames;
      for (unsigned d = 0; d < dimSizes.size(); ++d) {
        StringRef paramName = nameInventor.getUniqueName();

        int64_t maxFactor =
            dimSizes[d] == mlir::ShapedType::kDynamic ? 1024 : dimSizes[d];

        int64_t hiExp = 0;
        while ((int64_t(1) << (hiExp + 1)) <= maxFactor)
          ++hiExp;

        space.addPow2Range(paramName.str(), 0, hiExp);
        paramNames.push_back(paramName);
      }
      space.addConstraint(tilingConstraint(tileable, paramNames));

      op->setAttr(kTileParamNamesAttr,
                  OpBuilder(ctx).getStrArrayAttr(paramNames));
    });
  }

  Maybe<double> evaluate(cinm::ComputeBlockOp candidate,
                         const cinm::ConfigSpace &space,
                         const cinm::Configuration &config) override {
    MLIRContext *ctx = candidate->getContext();

    int64_t ranks = space.get(config, "ranks");
    int64_t dpus = space.get(config, "dpus");
    int64_t tasklets = space.get(config, "tasklets");
    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] evaluate: ranks=" << ranks
               << " dpus=" << dpus << " tasklets=" << tasklets << "\n");
    auto accelerator =
        upmem::UpmemAcceleratorAttr::get(platform, ranks, dpus, tasklets);
    candidate.setAcceleratorAttr(accelerator);
    applyTileSizes(candidate, space, config, ctx);

    // Create a fresh trial submodule for DPU kernels in the sandbox module,
    // and tell the cnm-to-upmem pass to use it via annotation.
    std::string trialName = ("trial_" + llvm::Twine(trialCount++)).str();

    // Run the lowering pipeline to UPMEM dialect.
    PassManager pm(ctx);
    pm.addPass(cinm::createCinmTilingPass());
    pm.addPass(cinm::createConvertTiledCinmToCnmPass());
    pm.addPass(cnm::createCnmHoistWorkgroupsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(cnm::createConvertCnmToUPMEMPass(
        {.kernelModuleName = std::move(trialName)}));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createUPMEMDedupKernelsPass());

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference]   running lowering pipeline\n");
    if (mlir::failed(pm.run(candidate))) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   pipeline failed\n");
      return emitSilenceableFailure(candidate->getLoc(), "Pass manager failed");
    }

    auto cost = simulator->simulate(candidate.getBody());
    LLVM_DEBUG({
      if (auto *val = std::get_if<double>(&cost))
        llvm::dbgs() << "[cinm-inference]   simulated cost = " << *val << "\n";
      else
        llvm::dbgs() << "[cinm-inference]   simulation failed\n";
    });
    return cost;
  }

  void disposeCandidate(cinm::ComputeBlockOp candidate) override {
    if (auto attr = candidate->getAttrOfType<StringAttr>(kKernelModuleAttr)) {
      auto sandboxModule = candidate->getParentOfType<ModuleOp>();
      if (sandboxModule) {
        auto *sym = SymbolTable::lookupSymbolIn(sandboxModule, attr.getValue());
        if (sym)
          sym->erase();
      }
    }
    candidate->erase();
  }

  mlir::DiagnosedSilenceableFailure
  commitBestCandidate(cinm::ComputeBlockOp original,
                      cinm::ComputeBlockOp bestCandidate) override {
    auto sandboxModule = bestCandidate->getParentOfType<ModuleOp>();
    auto originalModule = original->getParentOfType<ModuleOp>();

    // Move the trial kernel submodule from the sandbox into the original
    // module.
    if (auto attr =
            bestCandidate->getAttrOfType<StringAttr>(kKernelModuleAttr)) {
      auto *sym = SymbolTable::lookupSymbolIn(sandboxModule, attr.getValue());
      if (auto kernelModule = llvm::dyn_cast_or_null<ModuleOp>(sym)) {
        kernelModule.getOperation()->moveBefore(
            &originalModule.getBodyRegion().front(),
            originalModule.getBodyRegion().front().end());
      }
      original->setAttr(kKernelModuleAttr, attr);
    }

    // Replace the original's body with the lowered body from the best
    // candidate.
    original.getBody().takeBody(bestCandidate.getBody());

    // Copy the accelerator attribute.
    if (auto acc = bestCandidate->getAttr("accelerator"))
      original->setAttr("accelerator", acc);

    bestCandidate->erase();
    return DiagnosedSilenceableFailure::success();
  }

private:
  void applyTileSizes(cinm::ComputeBlockOp computeOp,
                      const cinm::ConfigSpace &space,
                      const cinm::Configuration &config,
                      MLIRContext *ctx) const {
    computeOp.getBody().walk([&](mlir::Operation *op) {
      auto paramNamesAttr = op->getAttrOfType<ArrayAttr>(kTileParamNamesAttr);
      if (!paramNamesAttr)
        return;

      llvm::SmallVector<int64_t> tileSizes;
      for (auto nameAttr : paramNamesAttr)
        tileSizes.push_back(
            space.get(config, llvm::cast<StringAttr>(nameAttr)));

      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-inference]   tiling " << op->getName() << " with "
                 << DenseI64ArrayAttr::get(ctx, tileSizes) << "\n");
      op->setAttr(cinm::CinmDialect::TILING_FACTORS_NAME,
                  DenseI64ArrayAttr::get(ctx, tileSizes));
    });
  }
};

using UpmemTilingConstraint = std::function<bool(
    UpmemAcceleratorAttr, const llvm::SmallVectorImpl<int64_t> &)>;

std::optional<cinm::Constraint> UpmemInferencePlugin::tilingConstraint(
    cinm::CinmTilingInterface op,
    llvm::SmallVectorImpl<StringRef> &tilingFactorNames) const {

  if (auto gemv = llvm::dyn_cast_or_null<cinm::GemvOp>(op.getOperation())) {
    auto m = tilingFactorNames[0], k = tilingFactorNames[1];
    auto eltTy = gemv.getLhs().getType().getElementType();

    return configurationValid([m, k, eltTy](UpmemAcceleratorAttr accelerator,
                                            const cinm::ConfWrapper &conf) {
      auto mv = conf[m], kv = conf[k];
      auto r = accelerator.getNumRanks(), d = accelerator.getNumDpusPerRank(),
           t = accelerator.getNumTaskletsPerDpu();
      if (mv % (r * d * t) != 0 || kv % (r * d) != 0)
        return false;
      auto wm = mv / (r * d * t);
      auto wk = kv / (r * d);
      return wk * (wm + 2) <=
             accelerator.getWramLevel().getSizeInElements(eltTy);
    });
  }
  return std::nullopt;
}

cinm::Constraint UpmemInferencePlugin::configurationValid(
    std::function<bool(UpmemAcceleratorAttr, const cinm::ConfWrapper &)>
        constraint) const {
  auto platform = this->platform;
  return [platform, constraint](const cinm::ConfWrapper &conf) {
    auto acc = upmem::UpmemAcceleratorAttr::get(platform, conf["ranks"],
                                                conf["dpus"], conf["tasklets"]);
    return constraint(acc, conf);
  };
}
// ===----------------------------------------------------------------------===//
// Pass
// ===----------------------------------------------------------------------===//
} // namespace
struct UpmemInferAcceleratorPass
    : impl::UpmemInferAcceleratorPassBase<UpmemInferAcceleratorPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DiagnosedSilenceableFailure failed = DiagnosedSilenceableFailure::success();

    IRRewriter rewriter(module->getContext());
    module.walk([&](cinm::ComputeBlockOp computeOp) -> WalkResult {
      // Look for a UpmemPlatformAttr in cinm.available_platforms on the
      // compute op or its enclosing function.
      upmem::UpmemPlatformAttr platform;
      auto tryExtract = [&](mlir::Operation *op) {
        auto arr = op->getAttrOfType<ArrayAttr>("cinm.available_platforms");
        if (!arr)
          return;
        for (auto attr : arr)
          if (auto p = llvm::dyn_cast<upmem::UpmemPlatformAttr>(attr)) {
            platform = p;
            break;
          }
      };
      tryExtract(computeOp.getOperation());
      if (!platform)
        if (auto func = computeOp->getParentOfType<func::FuncOp>())
          tryExtract(func.getOperation());
      if (!platform)
        return WalkResult::skip(); // not a UPMEM target

      UpmemInferencePlugin plugin(platform, createOpCountSimulator());
      cinm::InferenceOptions opts;
      opts.maxEvals = maxEvals;
      TRY_IN_WALK(failed,
                  cinm::inferAcceleratorConfig(computeOp, plugin, opts));
      return WalkResult::skip();
    });

    if (!failed.succeeded()) {
      (void)failed.checkAndReport();
      signalPassFailure();
    }
  }
};

} // namespace mlir::upmem
