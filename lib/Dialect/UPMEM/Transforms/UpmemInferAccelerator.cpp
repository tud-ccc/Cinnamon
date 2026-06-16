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
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

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
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/TilingInterface.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <vector>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMINFERACCELERATORPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

namespace {
using mlir::cinm::utils::Maybe;

// ===----------------------------------------------------------------------===//
// UpmemInferencePlugin
// ===----------------------------------------------------------------------===//

struct UpmemInferencePlugin;

// helper struct to add dynamic/static constraints on variables of the design
// space without manipulating bare arrays.
class ConstraintEditor {
  friend struct UpmemInferencePlugin;

  cinm::ConfigSpace &space;
  UpmemInferencePlugin *plugin;
  cinm::SearchParam &rd;
  cinm::SearchParam &t;
  SmallVector<cinm::SearchParam> tilingFactors;
  ArrayRef<int64_t> tiledDimensions;
  unsigned tfStart;

public:
  ConstraintEditor(cinm::ConfigSpace &space, UpmemInferencePlugin *plugin,
                   cinm::SearchParam &rd, cinm::SearchParam &t,
                   SmallVector<cinm::SearchParam> &&tilingFactors,
                   ArrayRef<int64_t> tiledDims, unsigned tfStart)
      : space(space), plugin(plugin), rd(rd), t(t),
        tilingFactors(std::move(tilingFactors)), tiledDimensions(tiledDims),
        tfStart(tfStart) {}

  void addStaticConstraint(
      function_ref<void(cinm::SearchParam &rd, cinm::SearchParam &t,
                        llvm::SmallVectorImpl<cinm::SearchParam> &tilingFactors,
                        ArrayRef<int64_t> tiledDims)>
          callback);

  void addDynamicConstraint(
      std::function<bool(int64_t rd, int64_t t, ArrayRef<int64_t> tilingFactors,
                         ArrayRef<int64_t> tiledDims)>
          callback);
};

struct UpmemInferencePlugin : cinm::InferencePlugin {
  upmem::UpmemPlatformAttr platform;
  bool mramTiling;
  std::unique_ptr<UpmemSimulator> simulator;

  // int64_t rankIx = -1;
  int64_t dpuIx = -1;
  int64_t taskletIx = -1;
  std::unique_ptr<PassManager> pipeline;

  static constexpr llvm::StringLiteral kTileParamNamesAttr =
      "upmem.tile_param_names";

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform, bool mramTiling,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), mramTiling(mramTiling), simulator(std::move(sim)) {}

  bool supportsMultithreading() const override {
    return simulator && simulator->supportsMultithreading();
  }

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
    pm->addPass(createCanonicalizerPass());
    {
      bufferization::OneShotBufferizePassOptions opts;
      opts.unknownTypeConversion =
          bufferization::LayoutMapOption::IdentityLayoutMap;
      // opts.bufferizeFunctionBoundaries = true;
      // opts.functionBoundaryTypeConversion =
      //     bufferization::LayoutMapOption::IdentityLayoutMap;
      pm->addPass(bufferization::createOneShotBufferizePass(opts));
    }
    pm->addPass(createCSEPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createConvertLinalgToAffineLoopsPass());
    // pm->addPass(bufferization::createBufferLoopHoistingPass());
    // pm->addPass(bufferization::createBufferHoistingPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    {
      bufferization::BufferResultsToOutParamsPassOptions outOpts;
      outOpts.hoistStaticAllocs = true;
      pm->addPass(bufferization::createBufferResultsToOutParamsPass(outOpts));
    }
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 4: affine opts
    {
      auto &funcs = pm->nest<func::FuncOp>();
      funcs.addPass(bufferization::createPromoteBuffersToStackPass());
      funcs.addPass(memref::createFoldMemRefAliasOpsPass());
      funcs.addPass(createCanonicalizerPass());
      funcs.addPass(affine::createLoopFusionPass());
      funcs.addPass(createSROA());
      funcs.addPass(createCanonicalizerPass());
      funcs.addPass(affine::createAffineScalarReplacementPass());
      funcs.addPass(createLoopInvariantCodeMotionPass());
      funcs.addPass(affine::createAffineLoopInvariantCodeMotionPass());
      funcs.addPass(createSROA());
      funcs.addPass(affine::createAffineScalarReplacementPass());
      funcs.addPass(createCanonicalizerPass());
      funcs.addPass(createCSEPass());
    }
    // Step 5: lower affine to SCF
    pm->addPass(createLowerAffinePass());
    // pm->addPass(bufferization::createBufferLoopHoistingPass());
    // pm->addPass(bufferization::createBufferHoistingPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 6: cnm → upmem
    pm->addPass(cnm::createConvertCnmToUPMEMPass({}));
    pm->addPass(createCSEPass());
    pm->addPass(createUPMEMDedupKernelsPass());
    pm->addPass(createCSEPass());
    {
      // This needs to apply after cnm->upmem bc of some assumptions we make
      // there.
      auto &funcs = pm->nest<func::FuncOp>();
      funcs.addPass(affine::createLoopUnrollPass(4));
    }
    return pm;
  }

  // --- InferencePlugin interface ---

  std::unique_ptr<cinm::InferencePlugin> clone() const override {
    auto c = std::make_unique<UpmemInferencePlugin>(platform, mramTiling,
                                                    simulator->clone());
    // c->rankIx = rankIx;
    c->dpuIx = dpuIx;
    c->taskletIx = taskletIx;
    return c;
  }

  void warmUp(mlir::MLIRContext *ctx) override {
    if (!pipeline)
      pipeline = buildPipeline(ctx);
    simulator->warmUp();
  }

  void handleOpConstraints(cinm::CinmTilingInterface op,
                           ConstraintEditor &editor);

  void initializeSpace(cinm::ComputeBlockOp refClone,
                       cinm::ConfigSpace &space) override {
    auto dpuCountParam = cinm::makeRange(
        "dpus", 1, platform.getMaxNumRanks() * platform.getMaxNumDpusPerRank());
    // auto rankParam = cinm::makeRange("ranks", 1, platform.getMaxNumRanks());
    // auto dpuParam = cinm::makeRange("dpus", 1,
    // platform.getMaxNumDpusPerRank());
    auto taskletParam =
        cinm::makeRange("tasklets", 1, platform.getMaxNumTasklets());

    auto nameInventor = cinm::utils::NameInventor::getNameInventor(
        refClone.getOperation(), "wtile");
    MLIRContext *ctx = refClone->getContext();

    refClone.getBody().walk([&](mlir::Operation *op) {
      auto tileable = llvm::dyn_cast<cinm::CinmTilingInterface>(op);
      if (!tileable)
        return;

      llvm::SmallVector<int64_t> dimSizes;
      tileable.getTilableDimSizes(dimSizes);

      llvm::SmallVector<cinm::SearchParam> tilingFactors;
      llvm::SmallVector<StringRef> paramNames;
      buildTilingParams(dimSizes, nameInventor, tilingFactors, paramNames);
      if (mramTiling) {
        auto nameInventor2 = cinm::utils::NameInventor::getNameInventor(
            refClone.getOperation(), "mtile");
        buildTilingParams(dimSizes, nameInventor2, tilingFactors, paramNames);
      }
      auto firstDim = space.params.size();
      ConstraintEditor editor(space, this, dpuCountParam, taskletParam,
                              std::move(tilingFactors), dimSizes, firstDim);

      handleOpConstraints(tileable, editor);

      for (auto &parm : std::move(editor.tilingFactors)) {
        space.addDim(std::move(parm));
      }

      op->setAttr(kTileParamNamesAttr,
                  OpBuilder(ctx).getStrArrayAttr(paramNames));
    });

    // this->rankIx = space.addDim(std::move(rankParam));
    this->dpuIx = space.addDim(std::move(dpuCountParam));
    this->taskletIx = space.addDim(std::move(taskletParam));
  }

  Maybe<double> evaluate(cinm::TrialInfo &trial) override {
    auto conf = trial.conf();
    int64_t dpus = conf[this->dpuIx], tasklets = conf[this->taskletIx];

    MLIRContext *ctx = trial.computeBlock->getContext();
    mlir::Location loc = trial.computeBlock->getLoc();
    trial.computeBlock.setAcceleratorAttr(
        upmem::UpmemAcceleratorAttr::get(platform, 1, dpus, tasklets));

    if (mramTiling) {
      // then bypass the lowering completely, we run the simulator on a template
      // TODO when the CNM pipeline allows, remove this and run the regular
      // pipeline
      double result = std::numeric_limits<double>::infinity();
      trial.computeBlock.getBody().walk([&](mlir::Operation *op) {
        auto paramNamesAttr = op->getAttrOfType<ArrayAttr>(kTileParamNamesAttr);
        if (!paramNamesAttr)
          return;

        llvm::SmallVector<int64_t> tileSizes;
        for (auto nameAttr : paramNamesAttr)
          tileSizes.push_back(
              trial.conf()[llvm::cast<StringAttr>(nameAttr).strref()]);

        if (auto gemv = llvm::dyn_cast_or_null<cinm::GemvOp>(op)) {
          auto shape = gemv.getLhs().getType().getShape();
          result =
              simulateFullGemv(shape[0], shape[1], tileSizes[2], tileSizes[3],
                               tileSizes[0], tileSizes[1], 1, dpus, tasklets);
        }
      });
      return result;
    }

    applyTileSizes(trial);

    if (!pipeline)
      pipeline = buildPipeline(ctx);

    {
      // This handler suppresses errors caused during trials, as they are
      // normal.
      ScopedDiagnosticHandler scopedHandler(ctx, [](Diagnostic &diag) {
        LLVM_DEBUG(llvm::dbgs()
                       << "[cinm-inference]   pipeline failed:\n      ";
                   diag.print(llvm::dbgs()); llvm::dbgs() << "\n";);

        return success();
      });
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference]   running pipeline\n");
      if (mlir::failed(pipeline->run(trial.module.get()))) {
        LLVM_DEBUG(trial.module->print(llvm::dbgs());
                   llvm::dbgs() << "\n========\n";);

        return emitSilenceableFailure(loc, "Pipeline failed");
      }
    }

    // computeBlock is gone after the pipeline; simulate the lowered module.
    return simulator->simulate(trial.computeBlock.getBody());
  }

private:
  static void
  buildTilingParams(llvm::ArrayRef<int64_t> dimSizes,
                    cinm::utils::NameInventor &nameInventor,
                    llvm::SmallVector<cinm::SearchParam> &tilingFactors,
                    llvm::SmallVector<StringRef> &paramNames) {
    for (int64_t size : dimSizes) {
      StringRef paramName = nameInventor.getUniqueName();
      if (ShapedType::isDynamic(size)) {
        tilingFactors.emplace_back(cinm::makePow2Range(paramName, 0, 10));
      } else {
        tilingFactors.emplace_back(paramName, cinm::IntRange{1, size})
            .keepDivisorsOf(size);
      }
      paramNames.push_back(paramName);
    }
  }

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

void UpmemInferencePlugin::handleOpConstraints(cinm::CinmTilingInterface op,
                                               ConstraintEditor &editor) {

  if (auto gemv = llvm::dyn_cast_or_null<cinm::GemvOp>(op.getOperation())) {
    editor.addStaticConstraint([&](auto &rd, auto &t, auto &, auto dims) {
      auto m = dims[0];
      if (!ShapedType::isDynamic(m)) {
        rd.keepDivisorsOf(m);
        // d.keepDivisorsOf(m);
        t.keepDivisorsOf(m);
      }
      auto k = dims[1];
      if (!ShapedType::isDynamic(k)) {
        rd.keepDivisorsOf(k);
        // d.keepDivisorsOf(k);
        // tasklets are broadcast
        // t.keepDivisorsOf(k);
      }
    });
    auto wramLevel = platform.getWramLevel();
    auto mramLevel = platform.getMramLevel();
    auto eltTy = gemv.getLhs().getType().getElementType();
    bool mramTiling = this->mramTiling;

    editor.addDynamicConstraint([wramLevel, mramLevel, eltTy,
                                 mramTiling](auto totalDpus, auto tasklets,
                                             auto tiles, auto dims) {
      const int64_t wramRowTile = tiles[0]; // WRAM rows per tasklet
      const int64_t wramColTile = tiles[1]; // WRAM cols per tasklet

      // Per-tasklet WRAM: A tile (wramRowTile×wramColTile) + x
      // (wramColTile) + y (wramRowTile)
      if (wramRowTile * wramColTile + wramColTile + wramRowTile >
          wramLevel.getSizeInElements(eltTy))
        return false;

      if (!mramTiling)
        return true;

      const int64_t mramRowTile =
          tiles[2]; // MRAM rows per DPU  (= ranks × dpusPerRank combined)
      const int64_t mramColTile = tiles[3]; // MRAM cols per DPU
      const int64_t M = dims[0], K = dims[1];

      // WRAM tiles must divide their MRAM counterparts.
      if (mramRowTile % wramRowTile != 0 || mramColTile % wramColTile != 0 ||
          mramRowTile < wramRowTile || mramColTile < wramColTile)
        return false;

      if (!ShapedType::isDynamic(M)) {
        // All DPUs and tasklets together cover
        // (totalDpus × tasklets × mramRowTile) rows per outer loop
        // iteration.
        if (M % (totalDpus * tasklets * mramRowTile) != 0 ||
            totalDpus * tasklets * mramRowTile > M)
          return false;
      }
      if (!ShapedType::isDynamic(K)) {
        if (K % mramColTile != 0 || mramColTile > K)
          return false;
      }

      // MRAM per DPU: A (tasklets×mramRowTile×mramColTile) + x
      // (mramColTile)
      //               + y (tasklets×mramRowTile)
      if (tasklets * mramRowTile * mramColTile + mramColTile +
              tasklets * mramRowTile >
          mramLevel.getSizeInElements(eltTy))
        return false;

      return true;
    });
  }
}
void ConstraintEditor::addStaticConstraint(
    function_ref<void(cinm::SearchParam &rd, cinm::SearchParam &t,
                      SmallVectorImpl<cinm::SearchParam> &tilingFactors,
                      ArrayRef<int64_t> tiledDims)>
        callback) {

  callback(rd, t, tilingFactors, tiledDimensions);
}

void ConstraintEditor::addDynamicConstraint(
    std::function<bool(int64_t rd, int64_t t, ArrayRef<int64_t> tilingFactors,
                       ArrayRef<int64_t> tiledDims)>
        callback) {
  // It is tricky to get the lifetimes right with dynamic constraints
  // so that's why we use this wrapper here.

  auto tfStart = this->tfStart;
  auto tfEnd = tfStart + tilingFactors.size();
  auto plugin = this->plugin;
  std::vector<int64_t> dimsCopy(tiledDimensions);
  space.addConstraint(
      [=, dimsCopy = std::move(dimsCopy)](const cinm::ConfWrapper conf) {
        // auto r = conf[plugin->rankIx], d = conf[plugin->dpuIx],
        auto rd = conf[plugin->dpuIx], t = conf[plugin->taskletIx];
        ArrayRef<int64_t> range(&conf.conf[tfStart], &conf.conf[tfEnd]);
        return callback(rd, t, range, dimsCopy);
      });
}

// ===----------------------------------------------------------------------===//
// Pass
// ===----------------------------------------------------------------------===//
} // namespace

/// UPMEM-specific inference options. Wraps the generic InferenceOptions and
/// provides a place to add UPMEM-specific knobs in the future.
struct UpmemInferenceOptions {
  cinm::InferenceOptions inference;
  bool annotateOpCosts = false;
  // todo wire these two through pass options
  bool useMRAMTiling = true;
  std::string simulator = "cycleaccurate";
};

struct UpmemInferAcceleratorPass
    : impl::UpmemInferAcceleratorPassBase<UpmemInferAcceleratorPass> {
  using Base::Base;

  UpmemInferenceOptions buildOptions() const {
    UpmemInferenceOptions upmemOpts;
    auto &o = upmemOpts.inference;
    o.maxEvals = maxEvals;
    o.nInit = nInit;
    o.rngSeed = rngSeed;
    o.kappa = kappa;
    o.epochs = epochs;
    o.nEnsemble = nEnsemble;
    o.hidden = hidden;
    o.depth = depth;
    o.neighborDepth = neighborDepth;
    o.neighborFrontierOnly = neighborFrontierOnly;
    o.exhaustiveSearch = exhaustiveSearch;
    o.nValidation = nValidation;
    o.validationInterval = validationInterval;
    o.objectiveScale = objectiveScale;
    upmemOpts.annotateOpCosts = annotateOpCosts;
    o.dumpDir = dumpDir;
    return upmemOpts;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DiagnosedSilenceableFailure failed = DiagnosedSilenceableFailure::success();

    UpmemInferenceOptions upmemOpts = buildOptions();
    auto dataDumpDir = std::move(upmemOpts.inference.dumpDir);

    cinm::utils::NameInventor inferenceNamer(&getContext(), "infer_");

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

      UpmemInferencePlugin plugin(
          platform, upmemOpts.useMRAMTiling,
          createSimulator(upmemOpts.simulator, upmemOpts.annotateOpCosts));

      if (!dataDumpDir.empty()) {
        auto parentFunc = computeOp->getParentOfType<SymbolOpInterface>();
        StringRef nameHint = parentFunc && parentFunc.getNameAttr()
                                 ? parentFunc.getName()
                                 : "op";
        auto name = inferenceNamer.getUniqueName(nameHint);
        LLVM_DEBUG(llvm::dbgs() << "===== START INFERENCE " << name << " =====";
                   llvm::dbgs() << "==================";);

        auto path = std::filesystem::path(dataDumpDir) / name.str();
        if (!upmemOpts.inference.exhaustiveSearch)
          path /= "seed_" + std::to_string(upmemOpts.inference.rngSeed);
        upmemOpts.inference.dumpDir = path;
      }
      TRY_IN_WALK(failed, cinm::inferAcceleratorConfig(computeOp, plugin,
                                                       upmemOpts.inference));
      return WalkResult::skip();
    });

    if (!failed.succeeded()) {
      (void)failed.checkAndReport();
      signalPassFailure();
    }
  }
};

} // namespace mlir::upmem
