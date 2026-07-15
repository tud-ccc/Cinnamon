#include "SimulatorBase.h"
#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include "upmem_cost_model/Types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <llvm/Support/LogicalResult.h>
#include <memory>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
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
using mlir::cinm::SpaceBuilder;
using mlir::cinm::SpaceVar;
using mlir::cinm::utils::Maybe;

/// UPMEM-specific inference options. Wraps the generic InferenceOptions and
/// provides a place to add UPMEM-specific knobs in the future.
struct UpmemInferenceOptions {
  cinm::InferenceOptions inference;
  bool annotateOpCosts = false;
  bool useMRAMTiling = true;
  UpmemSimulatorId simulator = UpmemSimulatorId::CYCLE_ACCURATE;
  std::chrono::milliseconds evalTimeoutMs = std::chrono::milliseconds(2000);
};

static void addAffineOpts(OpPassManager &pm) {
  // pm.addPass(affine::createLoopUnrollPass(1, true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(affine::createAffineFoldMemRefAliasOps());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(affine::createRaiseMemrefToAffine());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(affine::createAffineExpandIndexOpsAsAffinePass());
  // pm.addPass(affine::createLoopFusionPass());
  pm.addPass(createSROA());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(affine::createRaiseMemrefToAffine());
  pm.addPass(affine::createAffineScalarReplacementPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(affine::createAffineLoopInvariantCodeMotionPass());
  pm.addPass(createSROA());
  pm.addPass(affine::createAffineScalarReplacementPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  // pm->addPass(bufferization::createBufferLoopHoistingPass());
  // pm->addPass(bufferization::createBufferHoistingPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

struct UpmemInferencePlugin : cinm::InferencePlugin {
  upmem::UpmemPlatformAttr platform;
  const UpmemInferenceOptions &opts;
  std::unique_ptr<UpmemSimulator> simulator;

  // SpaceVars for dpus and tasklets, assigned during initializeSpace.
  SpaceVar dpusVar_, taskletsVar_;

  // Simulation callbacks registered by per-op handlers. Each receives the
  // current configuration and the (per-thread) simulator; evaluate() sums
  // their return values. The simulator is passed at call time so that clones
  // (which have their own simulator) work without lambda modification.
  using SimFn = std::function<Maybe<double>(
      const cinm::ConfWrapper &, UpmemSimulator &, cinm::TrialInfo &)>;
  std::vector<SimFn> simulators_;

  std::unique_ptr<PassManager> pipeline;

  static constexpr llvm::StringLiteral kTileParamNamesAttr =
      "upmem.tile_param_names";

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform,
                       const UpmemInferenceOptions &opts,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), opts(opts), simulator(std::move(sim)) {}

  void registerSimulator(SimFn &&fn) { simulators_.push_back(std::move(fn)); }

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
      addAffineOpts(funcs);
      funcs.addPass(createLowerAffinePass());
      // pm->addPass(bufferization::createBufferLoopHoistingPass());
      // pm->addPass(bufferization::createBufferHoistingPass());
      funcs.addPass(createCanonicalizerPass());
      funcs.addPass(createCSEPass());
      funcs.addPass(arith::createIntRangeOptimizationsPass());
      funcs.addPass(createCanonicalizerPass());
      funcs.addPass(createCSEPass());
      funcs.addPass(createCanonicalizerPass());
    }
    // Step 5: lower affine to SCF
    pm->addPass(createLowerAffinePass());
    // pm->addPass(bufferization::createBufferLoopHoistingPass());
    // pm->addPass(bufferization::createBufferHoistingPass());
    pm->addPass(arith::createIntRangeOptimizationsPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    pm->addPass(createCanonicalizerPass());
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
    auto c = std::make_unique<UpmemInferencePlugin>(platform, opts,
                                                    simulator->clone());
    c->dpusVar_ = dpusVar_;
    c->taskletsVar_ = taskletsVar_;
    c->simulators_ = simulators_;
    return c;
  }

  void warmUp(mlir::MLIRContext *ctx) override {
    if (!pipeline)
      pipeline = buildPipeline(ctx);
    simulator->warmUp();
  }

  void printStats() const override { simulator->printStats(); }

  void handleGemv(cinm::GemvOp gemv, SpaceBuilder &b);
  void handleReduce(cinm::ReduceOp op, SpaceBuilder &b);
  void handleEltwise(cinm::ElementwiseOp op, SpaceBuilder &b);

  void initializeSpace(cinm::ComputeBlockOp refClone,
                       cinm::ConfigSpace &space) override {
    SpaceBuilder b;
    simulators_.clear();
    dpusVar_ = b.intRange(
        "dpus", 1, platform.getMaxNumRanks() * platform.getMaxNumDpusPerRank());
    taskletsVar_ = b.intRange("tasklets", 1, platform.getMaxNumTasklets());

    refClone.getBody().walk([&](mlir::Operation *op) {
      if (auto gemv = llvm::dyn_cast<cinm::GemvOp>(op))
        handleGemv(gemv, b);
      else if (auto red = llvm::dyn_cast<cinm::ReduceOp>(op))
        handleReduce(red, b);
      // else if (auto ew = llvm::dyn_cast<cinm::ElementwiseOp>(op))
      //   handleEltwise(ew, b);
    });

    b.buildInto(space);
  }

  static DType cmDtyFromMlirDty(Type ty) {
    if (ty.isF32())
      return DType::F32;
    if (ty.isF64())
      return DType::F64;
    if (ty.isInteger(8))
      return DType::I8;
    if (ty.isInteger(16))
      return DType::I16;
    if (ty.isInteger(32))
      return DType::I32;
    if (ty.isInteger(64))
      return DType::I64;
    assert(false && "unsuported datatye");
  }
  static DiagnosedSilenceableFailure
  runPipeline(PassManager *pipeline, Location loc, ModuleOp module) {
    ScopedDiagnosticHandler scopedHandler(
        pipeline->getContext(), [](Diagnostic &diag) {
          LLVM_DEBUG(llvm::dbgs()
                         << "[cinm-inference]   pipeline failed:\n      ";
                     diag.print(llvm::dbgs()); llvm::dbgs() << "\n";);
          return success();
        });
    if (mlir::failed(pipeline->run(module))) {
      LLVM_DEBUG(module->print(llvm::dbgs()); llvm::dbgs() << "\n========\n";);
      return mlir::emitSilenceableFailure(loc, "Pipeline failed");
    }
    return DiagnosedSilenceableFailure::success();
  }

  Maybe<double> evaluate(cinm::TrialInfo &trial) override {
    auto conf = trial.conf();
    int64_t dpus = dpusVar_[conf];
    int64_t tasklets = taskletsVar_[conf];

    MLIRContext *ctx = trial.computeBlock->getContext();
    mlir::Location loc = trial.computeBlock->getLoc();
    trial.computeBlock.setPlatformAttr({});
    trial.computeBlock.setAcceleratorAttr(
        upmem::UpmemAcceleratorAttr::get(platform, 1, dpus, tasklets));

    // if (opts.useMRAMTiling) {
      // Bypass the lowering pipeline: call each op's registered simulator.
      double total = 0.0;
      for (auto &sim : simulators_)
        total += TRY_GET(sim(conf, *simulator, trial));
      if (opts.annotateOpCosts) {
        OpBuilder b(ctx);
        trial.computeBlock->setAttr(kSimCostAttr, b.getF64FloatAttr(total));
      }
      return total;
    // }

    // applyTileSizes(trial);

    // if (!pipeline)
    //   pipeline = buildPipeline(ctx);

    // TRY(runPipeline(pipeline.get(), loc, trial.module.get()));

    // auto total = TRY_GET(simulator->simulate(trial.computeBlock.getBody()));
    // if (opts.annotateOpCosts) {
    //   OpBuilder b(ctx);
    //   trial.computeBlock->setAttr(kSimCostAttr, b.getF64FloatAttr(total));
    // }
    // return total;
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

void UpmemInferencePlugin::handleGemv(cinm::GemvOp gemv, SpaceBuilder &b) {
  auto lhsTy = gemv.getLhs().getType();
  const int64_t M = lhsTy.getShape()[0], K = lhsTy.getShape()[1];
  auto eltTy = lhsTy.getElementType();
  auto wramLevel = platform.getWramLevel();
  auto mramLevel = platform.getMramLevel();
  const bool mramTiling = opts.useMRAMTiling;
  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;

  // Hardware dimensions.
  b.mustDivide(tasklets, M); // tasklets must divide M

  // WRAM tile dims: each is a divisor of its corresponding problem dimension.
  // K = dpuCols * mramCols * k1  ⟹  dpuCols | K  and  dpuCols | dpus
  auto wramRow = b.divisorsOf("wramRow", M);
  auto wramCol = b.divisorsOf("wramCol", K);
  auto dpuCols = b.divisorsOf("dpuCols", K);

  // Per-tasklet WRAM must fit: A tile (wr×wc) + x slice (wc) + y slot (wr)
  b.require(tasklets * wramRow * wramCol + wramCol + tasklets * wramRow <=
            wramLevel.getSizeInElements(eltTy));

  // Attributes used by applyTileSizes() in the non-MRAM pipeline path.
  gemv->setAttr(kTileParamNamesAttr,
                OpBuilder(gemv->getContext())
                    .getStrArrayAttr({wramRow.name(), wramCol.name()}));

  auto mramRow = b.divisorsOf("mramRow", M);
  auto mramCol = b.divisorsOf("mramCol", K);
  if (!mramTiling) {
    // In this mode we imitate cinm 1.0 behavior and do not tile on MRAM,
    // equivalently this means the MRAM and WRAM tiles have the same dimensions.
    // This corresponds to constraints:
    // - mramRow := wramRow * tasklets
    // - mramCol := wramCol
    b.require([=](auto c) -> bool {
      return mramRow[c] == wramRow[c] * tasklets[c] && mramCol[c] == wramCol[c];
    });
  }

  b.require(M / ((dpus / dpuCols) * mramRow));
  b.require(mramRow / (tasklets * wramRow));
  b.require(mramCol / wramCol);
  b.require(K / (dpuCols * mramCol));

  // Per-DPU MRAM must fit: A (T×mr×mc) + x (mc) + y (T×mr)
  b.require(mramRow * mramCol + mramCol + mramRow <=
            mramLevel.getSizeInElements(eltTy));

  // Simulation template for the MRAM fast path (bypasses the lowering
  // pipeline).
  registerSimulator([=](const cinm::ConfWrapper &c, UpmemSimulator &sim,
                        cinm::TrialInfo &trial) -> Maybe<double> {
    auto bufferizePm =
        std::make_unique<PassManager>(trial.computeBlock.getContext());
    {
      bufferization::OneShotBufferizePassOptions opts;
      opts.unknownTypeConversion =
          bufferization::LayoutMapOption::IdentityLayoutMap;
      bufferizePm->addPass(bufferization::createOneShotBufferizePass(opts));
    }
    TRY(runPipeline(bufferizePm.get(), gemv->getLoc(), trial.module.get()));

    IRRewriter rewriter(trial.module->getContext());
    rewriter.setInsertionPointToStart(&trial.computeBlock.getBody().front());

    trial.computeBlock->walk([&](cinm::GemvOp op) {
      generateGemv(op, rewriter, dpus[c] / dpuCols[c], dpuCols[c],
                   mramRow[c], mramCol[c], wramRow[c], wramCol[c],
                   tasklets[c]);
    });

    auto cleanupPm =
        std::make_unique<PassManager>(trial.computeBlock.getContext());
    {
      auto &dpuPm = cleanupPm->nest<upmem::DpuProgramOp>();
      addAffineOpts(dpuPm);
      dpuPm.addPass(createLowerAffinePass());
      dpuPm.addPass(createCanonicalizerPass());
      dpuPm.addPass(createCSEPass());
    }
    {
      auto &funcs = cleanupPm->nest<func::FuncOp>();
      funcs.addPass(createConvertLinalgToAffineLoopsPass());
      addAffineOpts(funcs);
    }

    TRY(runPipeline(cleanupPm.get(), gemv->getLoc(), trial.module.get()));

    return sim.simulate(trial.computeBlock.getBody());
  });
}

void UpmemInferencePlugin::handleReduce(cinm::ReduceOp op, SpaceBuilder &b) {
  auto type = op.getInput().getType();
  if (op.getDimension() != type.getShape().size() - 1) {
    // For now only support when the reduction dimension is the last one
    return;
  }
  auto parShape = type.getShape().drop_back();
  const auto M = computeProduct(parShape);
  const auto K = type.getShape()[op.getDimension()];

  auto eltTy = type.getElementType();
  auto wramLevel = platform.getWramLevel();
  auto mramLevel = platform.getMramLevel();
  const bool mramTiling = opts.useMRAMTiling;
  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;

  auto taskletCols = b.divisorsOf("taskletCols", tasklets);

  // WRAM tile dims: each is a divisor of its corresponding problem dimension.
  // K = dpuCols * mramCols * k1  ⟹  dpuCols | K  and  dpuCols | dpus
  auto wramRow = b.divisorsOf("wramRow", M);
  auto wramCol = b.divisorsOf("wramCol", K);
  auto dpuCols = b.divisorsOf("dpuCols", K);

  // Per-tasklet WRAM must fit: wramTile * tasklets + tasklets
  b.require(wramCol * wramRow * tasklets + tasklets <=
            wramLevel.getSizeInElements(eltTy));

  // Attributes used by applyTileSizes() in the non-MRAM pipeline path.
  // fixme here
  // op->setAttr(kTileParamNamesAttr,
  //             OpBuilder(op->getContext()).getStrArrayAttr({wramTile.name()}));

  auto mramRow = b.divisorsOf("mramRow", M);
  auto mramCol = b.divisorsOf("mramCol", K);
  b.require(M / ((dpus / dpuCols) * mramRow));
  b.require(mramRow / ((tasklets / taskletCols) * wramRow));
  b.require(mramCol / (taskletCols * wramCol));
  b.require(K / (dpuCols * mramCol));

  if (!mramTiling) {
    // In this mode we imitate cinm 1.0 behavior and do not tile on MRAM,
    // equivalently this means the MRAM and WRAM tiles have the same dimensions.
    // This corresponds to constraints:
    // - mramRow := wramRow * (tasklets / taskletCols)
    // - mramCol := wramCol * taskletCols
    //
    // (tasklets / taskletCols) is taskletRows
    b.require([=](auto c) -> bool {
      return mramRow[c] == (wramRow[c] * tasklets[c] / taskletCols[c]) &&
             mramCol[c] == wramCol[c] * taskletCols[c];
    });
  }

  // Per-DPU MRAM must fit: A (T×mr×mc) + y (T×mr)
  b.require(mramRow * mramCol + mramRow <= mramLevel.getSizeInElements(eltTy));

  // Simulation template for the MRAM fast path (bypasses the lowering
  // pipeline).
  if (op.getDimension() == type.getShape().size() - 1) {

    // todo register simulator for specific op, here we assume
    //  that there is a single op in the compute block
    registerSimulator([=](const cinm::ConfWrapper &c, UpmemSimulator &sim,
                          cinm::TrialInfo &trial) -> Maybe<double> {
      auto bufferizePm =
          std::make_unique<PassManager>(trial.computeBlock.getContext());
      {
        bufferization::OneShotBufferizePassOptions opts;
        opts.unknownTypeConversion =
            bufferization::LayoutMapOption::IdentityLayoutMap;
        // opts.bufferizeFunctionBoundaries = true;
        // opts.functionBoundaryTypeConversion =
        //     bufferization::LayoutMapOption::IdentityLayoutMap;
        bufferizePm->addPass(bufferization::createOneShotBufferizePass(opts));
      }
      TRY(runPipeline(bufferizePm.get(), op->getLoc(), trial.module.get()));

      IRRewriter rewriter(trial.module->getContext());
      rewriter.setInsertionPointToStart(&trial.computeBlock.getBody().front());

      trial.computeBlock->walk([&](cinm::ReduceOp op) {
        generateTailReduction(op, rewriter, dpus[c] / dpuCols[c], dpuCols[c],
                              mramRow[c], mramCol[c], wramRow[c], wramCol[c],
                              tasklets[c] / taskletCols[c], taskletCols[c]);
      });
      auto cleanupPm =
          std::make_unique<PassManager>(trial.computeBlock.getContext());
      {
        auto &dpuPm = cleanupPm->nest<upmem::DpuProgramOp>();
        addAffineOpts(dpuPm);
        // dpuPm.addPass(affine::createLoopUnrollPass(
        //     -1, false, [](affine::AffineForOp forOp) -> unsigned int {
        //       auto tc =
        //       dyn_cast<LoopLikeOpInterface>(*forOp).getStaticTripCount(); if
        //       (tc && tc->getZExtValue() <= 4) {
        //         // In an upmem DPU program, we want to either unroll in
        //         // full and have static (immediate) index patterns, or not
        //         // unroll. This is because the IRAM is shared with the WRAM.
        //         return tc->getZExtValue();
        //       }
        //       return 1; // do not unroll
        //     }));

        dpuPm.addPass(createLowerAffinePass());
        dpuPm.addPass(createCanonicalizerPass());
        dpuPm.addPass(createCSEPass());
      }
      {
        auto &funcs = cleanupPm->nest<func::FuncOp>();
        funcs.addPass(createConvertLinalgToAffineLoopsPass());
        addAffineOpts(funcs);
        // funcs.addPass(affine::createAffineVectorize(
        //     affine::AffineVectorizeOptions{.vectorSizes = {8},
        //                                    .fastestVaryingPattern = {},
        //                                    .vectorizeReductions = true}));
      }

      TRY(runPipeline(cleanupPm.get(), op->getLoc(), trial.module.get()));

      return sim.simulate(trial.computeBlock.getBody());
    });
  }
}

// ===----------------------------------------------------------------------===//
// Pass
// ===----------------------------------------------------------------------===//
} // namespace

struct UpmemInferAcceleratorPass
    : impl::UpmemInferAcceleratorPassBase<UpmemInferAcceleratorPass> {
  using Base::Base;

  UpmemInferenceOptions buildOptions() const {
    UpmemInferenceOptions upmemOpts;
    auto &o = upmemOpts.inference;
    o.maxEvals = maxEvals;
    o.nInit = nInit;
    o.rngSeed = rngSeed;
    o.nSeeds = nSeeds;
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
    o.numWorkers = numWorkers;
    upmemOpts.annotateOpCosts = annotateOpCosts;
    upmemOpts.useMRAMTiling = useMRAMTiling;
    upmemOpts.simulator = simulator;
    upmemOpts.evalTimeoutMs = std::chrono::milliseconds(evalTimeoutMs);
    o.dumpDir = dumpDir;
    if (!evalSolution.empty())
      o.evalSingleSolution =
          cinm::Configuration(evalSolution.begin(), evalSolution.end());
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

      UpmemInferencePlugin plugin(platform, upmemOpts,
                                  createSimulator(upmemOpts.simulator,
                                                  upmemOpts.annotateOpCosts,
                                                  upmemOpts.evalTimeoutMs));

      if (!dataDumpDir.empty()) {
        auto parentFunc = computeOp->getParentOfType<SymbolOpInterface>();
        StringRef nameHint = parentFunc && parentFunc.getNameAttr()
                                 ? parentFunc.getName()
                                 : "op";
        auto name = inferenceNamer.getUniqueName(nameHint);
        LLVM_DEBUG(llvm::dbgs() << "===== START INFERENCE " << name << " =====";
                   llvm::dbgs() << "==================";);

        auto path = std::filesystem::path(dataDumpDir) / name.str();
        // Multi-seed mode appends its own seed_<value>/ per seed, so pass the
        // base (per-op) dir. Single-seed BO gets the seed_<rngSeed>/ suffix
        // here.
        if (!upmemOpts.inference.exhaustiveSearch &&
            upmemOpts.inference.nSeeds <= 1)
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
