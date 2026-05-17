#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
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
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <utility>

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

  int64_t rankIx = -1;
  int64_t dpuIx = -1;
  int64_t taskletIx = -1;
  std::unique_ptr<PassManager> pipeline;

  static constexpr llvm::StringLiteral kTileParamNamesAttr =
      "upmem.tile_param_names";

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), simulator(std::move(sim)) {}

  // Full lowering pipeline (steps 1-6): cinm → cnm → bufferize → upmem.
  static std::unique_ptr<PassManager> buildPipeline(MLIRContext *ctx) {
    auto pm = std::make_unique<PassManager>(ctx);

    // Step 1: tiling
    pm->addPass(cinm::createCinmTilingPass());
    // pm->addPass(cinm::createCinmIsolateComputePass());
    // Fully unroll single-iteration loops produced by tiling.
    pm->addNestedPass<func::FuncOp>(
        affine::createLoopUnrollPass(1, /*unrollUpToFactor=*/true));
    pm->addPass(createCanonicalizerPass());
    // pm->addPass(cinm::createCinmDeisolateComputeBlocks());

    // Step 2: cinm → cnm
    pm->addPass(cinm::createConvertTiledCinmToCnmPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(cnm::createCnmHoistWorkgroupsPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 3: bufferize
    pm->addPass(bufferization::createEmptyTensorEliminationPass());
    pm->addPass(createCSEPass());
    // {
    //   bufferization::OneShotBufferizePassOptions opts;
    //   opts.unknownTypeConversion =
    //   bufferization::LayoutMapOption::IdentityLayoutMap;
    //   // opts.bufferizeFunctionBoundaries = true;
    //   // opts.functionBoundaryTypeConversion =
    //   //     bufferization::LayoutMapOption::IdentityLayoutMap;
    //   pm->addPass(bufferization::createOneShotBufferizePass(opts));
    // }
    // pm->addPass(createCSEPass());
    // pm->addPass(createCanonicalizerPass());
    // pm->addPass(createConvertLinalgToAffineLoopsPass());
    // pm->addPass(bufferization::createBufferLoopHoistingPass());
    // pm->addPass(bufferization::createBufferHoistingPass());
    // pm->addPass(createCanonicalizerPass());
    // pm->addPass(createCSEPass());
    // {
    //   bufferization::BufferResultsToOutParamsPassOptions outOpts;
    //   outOpts.hoistStaticAllocs = true;
    //   pm->addPass(bufferization::createBufferResultsToOutParamsPass(outOpts));
    // }
    // pm->addPass(createCanonicalizerPass());
    // pm->addPass(createCSEPass());

    // // Step 4: affine opts
    // {
    //   auto &funcs = pm->nest<func::FuncOp>();
    //   funcs.addPass(bufferization::createPromoteBuffersToStackPass());
    //   funcs.addPass(memref::createFoldMemRefAliasOpsPass());
    //   funcs.addPass(createCanonicalizerPass());
    //   funcs.addPass(affine::createLoopFusionPass());
    //   funcs.addPass(createSROA());
    //   funcs.addPass(createCanonicalizerPass());
    //   funcs.addPass(affine::createAffineScalarReplacementPass());
    //   funcs.addPass(createLoopInvariantCodeMotionPass());
    //   funcs.addPass(affine::createAffineLoopInvariantCodeMotionPass());
    //   funcs.addPass(createSROA());
    //   funcs.addPass(affine::createAffineScalarReplacementPass());
    //   funcs.addPass(createCanonicalizerPass());
    //   funcs.addPass(createCSEPass());
    //   funcs.addPass(affine::createLoopUnrollPass(4));
    // }
    // // Step 5: lower affine to SCF
    // pm->addPass(createLowerAffinePass());
    // pm->addPass(bufferization::createBufferLoopHoistingPass());
    // pm->addPass(bufferization::createBufferHoistingPass());
    // pm->addPass(createCanonicalizerPass());
    // pm->addPass(createCSEPass());

    // // Step 6: cnm → upmem
    // pm->addPass(cnm::createConvertCnmToUPMEMPass({}));
    // pm->addPass(createCSEPass());
    // pm->addPass(createUPMEMDedupKernelsPass());
    // pm->addPass(createCSEPass());

    return pm;
  }

  // --- InferencePlugin interface ---

  std::optional<cinm::Constraint>
  tilingConstraint(cinm::CinmTilingInterface op,
                   llvm::SmallVectorImpl<int64_t> &tilingFactorsIx) const;

  void handleOpConstraints(cinm::ConfigSpace &space,
                           cinm::CinmTilingInterface op,
                           llvm::SmallVectorImpl<int64_t> &tilingFactorsIx,
                           cinm::SearchParam &ranks, cinm::SearchParam &dpus,
                           cinm::SearchParam &tasklets) const;

  void initializeSpace(cinm::ComputeBlockOp refClone,
                       cinm::ConfigSpace &space) override {
    auto rankParam = cinm::makeRange("ranks", 1, platform.getMaxNumRanks());
    auto dpuParam = cinm::makeRange("dpus", 1, platform.getMaxNumDpusPerRank());
    auto taskletParam =
        cinm::makeRange("tasklets", 1, platform.getMaxNumTasklets());

    auto nameInventor = cinm::utils::NameInventor::getNameInventor(
        refClone.getOperation(), "tile_");
    MLIRContext *ctx = refClone->getContext();

    refClone.getBody().walk([&](mlir::Operation *op) {
      auto tileable = llvm::dyn_cast<cinm::CinmTilingInterface>(op);
      if (!tileable)
        return;

      llvm::SmallVector<int64_t> dimSizes;
      tileable.getTilableDimSizes(dimSizes);

      llvm::SmallVector<cinm::SearchParam> tilingFactors;
      llvm::SmallVector<StringRef> paramNames;
      for (unsigned d = 0; d < dimSizes.size(); ++d) {
        StringRef paramName = nameInventor.getUniqueName();

        cinm::SearchParam searchParm;
        if (ShapedType::isDynamic(dimSizes[d])) {
          searchParm = cinm::makePow2Range(paramName, 0, 10);
        } else {
          searchParm = cinm::makeRange(paramName, 1, dimSizes[d]);
          searchParm.keepDivisorsOf(dimSizes[d]);
        }
        tilingFactors.emplace_back(std::move(searchParm));
        paramNames.push_back(paramName);
      }

      if (auto gemv = llvm::dyn_cast_or_null<cinm::GemvOp>(op)) {
        if (!ShapedType::isDynamic(dimSizes[0])) {
          rankParam.keepDivisorsOf(dimSizes[0]);
          dpuParam.keepDivisorsOf(dimSizes[0]);
          taskletParam.keepDivisorsOf(dimSizes[0]);
        }
        if (!ShapedType::isDynamic(dimSizes[1])) {
          rankParam.keepDivisorsOf(dimSizes[1]);
          dpuParam.keepDivisorsOf(dimSizes[1]);
          // tasklets are broadcast
          // taskletParam.keepDivisorsOf(dimSizes[0]);
        }
        auto &mname = tilingFactors[0].name, &kname = tilingFactors[1].name;
        auto eltTy = gemv.getLhs().getType().getElementType();
        auto wramLevel = platform.getWramLevel();
        space.addConstraint(
            [this, wramLevel, eltTy, mname, kname](auto &conf) -> bool {
              auto r = conf[rankIx], d = conf[dpuIx], t = conf[taskletIx];
              auto mv = conf[mname], kv = conf[kname];
              if (mv % (r * d * t) != 0 || kv % (r * d) != 0)
                return false;
              auto wm = mv / (r * d * t);
              auto wk = kv / (r * d * t);
              LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   - constraint: WM="
                                      << wm << ", WK=" << wk << "\n");
              return wk * (wm + 2) <= wramLevel.getSizeInElements(eltTy);
            });
      }
      for (auto &parm : tilingFactors) {
        space.addDim(std::move(parm));
      }

      op->setAttr(kTileParamNamesAttr,
                  OpBuilder(ctx).getStrArrayAttr(paramNames));
    });

    this->rankIx = space.addDim(std::move(rankParam));
    this->dpuIx = space.addDim(std::move(dpuParam));
    this->taskletIx = space.addDim(std::move(taskletParam));
  }

  Maybe<double> evaluate(cinm::TrialInfo &trial) override {
    auto conf = trial.conf();
    int64_t ranks = conf["ranks"], dpus = conf["dpus"],
            tasklets = conf["tasklets"];

    LLVM_DEBUG(llvm::dbgs()
               << "[cinm-inference] evaluate: ranks=" << ranks
               << " dpus=" << dpus << " tasklets=" << tasklets << "\n");

    MLIRContext *ctx = trial.computeBlock->getContext();
    mlir::Location loc = trial.computeBlock->getLoc();
    trial.computeBlock.setAcceleratorAttr(
        upmem::UpmemAcceleratorAttr::get(platform, ranks, dpus, tasklets));

    applyTileSizes(trial);

    if (!pipeline)
      pipeline = buildPipeline(ctx);

    LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   running pipeline\n");
    if (mlir::failed(pipeline->run(trial.module.get()))) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   pipeline failed\n";
                 trial.module->print(llvm::dbgs());
                 llvm::dbgs() << "\n========\n";);

      return emitSilenceableFailure(loc, "Pipeline failed");
    }

    // computeBlock is gone after the pipeline; simulate the lowered module.
    return simulator->simulate(trial.computeBlock.getBody());
  }

private:
  void applyTileSizes(cinm::TrialInfo &trial) const {
    trial.computeBlock.getBody().walk([&](mlir::Operation *op) {
      auto paramNamesAttr = op->getAttrOfType<ArrayAttr>(kTileParamNamesAttr);
      if (!paramNamesAttr)
        return;

      llvm::SmallVector<int64_t> tileSizes;
      for (auto nameAttr : paramNamesAttr)
        tileSizes.push_back(
            trial.conf()[llvm::cast<StringAttr>(nameAttr).strref()]);

      auto tileSizesAttr = DenseI64ArrayAttr::get(op->getContext(), tileSizes);
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   tiling " << op->getName()
                              << " with " << tileSizesAttr << "\n");
      op->setAttr(cinm::CinmDialect::TILING_FACTORS_NAME, tileSizesAttr);
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
