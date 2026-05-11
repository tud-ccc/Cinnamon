#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/AcceleratorInference.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMINFERACCELERATORPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {

// ===----------------------------------------------------------------------===//
// TilingParamEntry — maps config-space param names back to op positions
// ===----------------------------------------------------------------------===//

struct TilingParamEntry {
  unsigned tileableOpIdx; // index within the pre-order walk of tileable ops
  unsigned dimIdx;
  std::string paramName;
};

// ===----------------------------------------------------------------------===//
// UpmemInferencePlugin
// ===----------------------------------------------------------------------===//

struct UpmemInferencePlugin : cinm::InferencePlugin {
  upmem::UpmemPlatformAttr platform;
  std::unique_ptr<UpmemSimulator> simulator;
  llvm::SmallVector<TilingParamEntry> tilingEntries;
  unsigned tileableOpCounter = 0;

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), simulator(std::move(sim)) {}

  // --- InferencePlugin interface ---

  void initializeSpace(cinm::ComputeOp refClone, cinm::ConfigSpace &space) override {
    space.addRange("ranks", 1, platform.getMaxNumRanks());
    space.addRange("dpus", 1, platform.getMaxNumDpusPerRank());
    space.addRange("tasklets", 1, platform.getMaxNumTasklets());

    refClone.getBody().walk([&](mlir::Operation *op) {
      auto tileable = llvm::dyn_cast<cinm::CinmTilingInterface>(op);
      if (!tileable)
        return;

      
      unsigned opIdx = tileableOpCounter++;
      llvm::SmallVector<int64_t> dimSizes;
      tileable.getTilableDimSizes(dimSizes);

      for (unsigned d = 0; d < dimSizes.size(); ++d) {
        std::string paramName =
            "tile_" + std::to_string(opIdx) + "_dim" + std::to_string(d);

        int64_t maxFactor =
            dimSizes[d] == mlir::ShapedType::kDynamic ? 1024 : dimSizes[d];

        int64_t hiExp = 0;
        while ((int64_t(1) << (hiExp + 1)) <= maxFactor)
          ++hiExp;

        space.addPow2Range(paramName, 0, hiExp);
        tilingEntries.push_back({opIdx, d, std::move(paramName)});
      }
    });
  }

  mlir::FailureOr<double> evaluate(cinm::ComputeOp clonedComputeOp,
                                   const cinm::ConfigSpace &space,
                                   const cinm::Configuration &config) override {
    MLIRContext *ctx = clonedComputeOp->getContext();

    int64_t ranks = space.get(config, "ranks");
    int64_t dpus = space.get(config, "dpus");
    int64_t tasklets = space.get(config, "tasklets");
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
      return mlir::failure();
    if (mlir::failed(pm.run(newModule)))
      return mlir::failure();

    return simulator->simulate(newModule);
  }

  mlir::LogicalResult
  applyBestConfig(cinm::ComputeOp computeOp, const cinm::ConfigSpace &space,
                  const cinm::Configuration &config) override {
    int64_t ranks = space.get(config, "ranks");
    int64_t dpus = space.get(config, "dpus");
    int64_t tasklets = space.get(config, "tasklets");
    auto accelerator =
        upmem::UpmemAcceleratorAttr::get(platform, ranks, dpus, tasklets);
    computeOp->setAttr("accelerator", accelerator);
    applyTileSizes(computeOp, space, config, computeOp->getContext());
    return mlir::success();
  }

private:
  // Set cinm.tile_sizes on every tileable op in the compute body, using the
  // tileableOpIdx counter to match params recorded during populate.
  void applyTileSizes(cinm::ComputeOp computeOp, const cinm::ConfigSpace &space,
                      const cinm::Configuration &config,
                      MLIRContext *ctx) const {
    unsigned tileableOpIdx = 0;
    computeOp.getBody().walk([&](mlir::Operation *op) {
      if (!llvm::isa<cinm::CinmTilingInterface>(op))
        return;
      unsigned myIdx = tileableOpIdx++;

      auto tileable = llvm::cast<cinm::CinmTilingInterface>(op);
      llvm::SmallVector<int64_t> dimSizes;
      tileable.getTilableDimSizes(dimSizes);
      llvm::SmallVector<int64_t> tileSizes(dimSizes.size(), 1);

      for (const auto &entry : tilingEntries) {
        if (entry.tileableOpIdx == myIdx)
          tileSizes[entry.dimIdx] = space.get(config, entry.paramName);
      }

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
    bool failed = false;

    module.walk([&](cinm::ComputeOp computeOp) {
      if (failed)
        return;

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
        return; // not a UPMEM target

      UpmemInferencePlugin plugin(platform, createOpCountSimulator());
      cinm::InferenceOptions opts;
      opts.maxEvals = maxEvals;
      if (mlir::failed(cinm::inferAcceleratorConfig(computeOp, plugin, opts)))
        failed = true;
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace mlir::upmem
