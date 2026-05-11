#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/WalkResult.h>

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

  static constexpr llvm::StringLiteral kTileParamNamesAttr =
      "upmem.tile_param_names";

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), simulator(std::move(sim)) {}

  // --- InferencePlugin interface ---

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

      llvm::SmallVector<Attribute> paramNames;
      for (unsigned d = 0; d < dimSizes.size(); ++d) {
        StringAttr paramName = nameInventor.getUniqueName();

        int64_t maxFactor =
            dimSizes[d] == mlir::ShapedType::kDynamic ? 1024 : dimSizes[d];

        int64_t hiExp = 0;
        while ((int64_t(1) << (hiExp + 1)) <= maxFactor)
          ++hiExp;

        space.addPow2Range(paramName.str(), 0, hiExp);
        paramNames.push_back(paramName);
      }

      op->setAttr(kTileParamNamesAttr, ArrayAttr::get(ctx, paramNames));
    });
  }

  Maybe<double> evaluate(cinm::ComputeBlockOp clonedComputeOp,
                         const cinm::ConfigSpace &space,
                         const cinm::Configuration &config) override {
    MLIRContext *ctx = clonedComputeOp->getContext();

    int64_t ranks = space.get(config, "ranks");
    int64_t dpus = space.get(config, "dpus");
    int64_t tasklets = space.get(config, "tasklets");
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] evaluate: ranks=" << ranks
                             << " dpus=" << dpus << " tasklets=" << tasklets
                             << "\n");
    auto accelerator =
        upmem::UpmemAcceleratorAttr::get(platform, ranks, dpus, tasklets);
    clonedComputeOp->setAttr("accelerator", accelerator);
    applyTileSizes(clonedComputeOp, space, config, ctx);

    // Run the lowering pipeline to UPMEM dialect.
    auto newModule = clonedComputeOp->getParentOfType<ModuleOp>();
    PassManager pm(ctx);
    llvm::StringRef pipeline =
        "builtin.module(func.func(cinm-tiling),"
        "convert-cinm-to-cnm,cnm-hoist-workgroups,canonicalize,cse,"
        "convert-cnm-to-upmem,cse,upmem-dedup-kernels)";
    if (mlir::failed(
            mlir::parsePassPipeline(pipeline, *(mlir::OpPassManager *)&pm)))
      return emitDefiniteFailure(clonedComputeOp->getLoc(),
                                 "Could not parse pass pipeline");
    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   running lowering pipeline\n");
    if (mlir::failed(pm.run(newModule))) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   pipeline failed\n");
      return emitSilenceableFailure(clonedComputeOp->getLoc(),
                                    "Pass manager failed");
    }

    auto cost = simulator->simulate(newModule);
    LLVM_DEBUG({
      if (auto *val = std::get_if<double>(&cost))
        llvm::dbgs() << "[cinm-inference]   simulated cost = " << *val << "\n";
      else
        llvm::dbgs() << "[cinm-inference]   simulation failed\n";
    });
    return cost;
  }

  mlir::DiagnosedSilenceableFailure
  applyBestConfig(cinm::ComputeBlockOp computeOp, const cinm::ConfigSpace &space,
                  const cinm::Configuration &config) override {
    int64_t ranks = space.get(config, "ranks");
    int64_t dpus = space.get(config, "dpus");
    int64_t tasklets = space.get(config, "tasklets");
    auto accelerator =
        upmem::UpmemAcceleratorAttr::get(platform, ranks, dpus, tasklets);
    computeOp->setAttr("accelerator", accelerator);
    applyTileSizes(computeOp, space, config, computeOp->getContext());
    return DiagnosedSilenceableFailure::success();
  }

private:
  void applyTileSizes(cinm::ComputeBlockOp computeOp, const cinm::ConfigSpace &space,
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

      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   tiling " << op->getName()
                               << " with " << DenseI64ArrayAttr::get(ctx, tileSizes)
                               << "\n");
      op->setAttr(cinm::CinmDialect::TILING_FACTORS_NAME,
                  DenseI64ArrayAttr::get(ctx, tileSizes));
    });
  }
};

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
