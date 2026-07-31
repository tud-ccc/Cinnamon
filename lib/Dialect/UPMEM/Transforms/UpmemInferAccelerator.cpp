#include <cinm-mlir/Conversion/CinmPasses.h>
#include <cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h>
#include <cinm-mlir/Conversion/LinalgToCnm/LinalgToCnm.h>
#include <cinm-mlir/Conversion/CommonPatterns.h>
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h>
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmBase.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h>
#include <cinm-mlir/Dialect/Cinm/Transforms/Passes.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cinm-mlir/Utils/DebugPasses.h>
#include <upmem_cost_model/Types.h>

#include "SimulatorBase.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Transforms/Passes.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Pipelines/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
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

#define DEBUG_TYPE "cinm-inference"

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMINFERACCELERATORPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {
using mlir::cinm::SpaceBuilder;
using mlir::cinm::SpaceVar;
using mlir::cinm::utils::Maybe;

/// A search-space quantity resolved against a configuration. Type-erased so a
/// recorded parameter can be a derived expression rather than a bare variable
/// -- what a pass consumes is rarely what the search declares.
using SpaceValue = std::function<int64_t(const cinm::ConfWrapper &)>;

/// Type-erase any space expression into a SpaceValue. Handles copy the
/// variables' shared index cells, so this stays valid across
/// SpaceBuilder::buildInto().
template <typename E>
static SpaceValue spaceValue(const cinm::SpaceExprBase<E> &expr) {
  E copy = static_cast<const E &>(expr);
  return [copy](const cinm::ConfWrapper &c) { return copy.eval(c); };
}

/// UPMEM-specific inference options. Wraps the generic InferenceOptions and
/// provides a place to add UPMEM-specific knobs in the future.
struct UpmemInferenceOptions {
  cinm::InferenceOptions inference;
  bool annotateOpCosts = false;
  bool useMRAMTiling = true;
  UpmemLoweringPath lowering = UpmemLoweringPath::TEMPLATES;
  UpmemSimulatorId simulator = UpmemSimulatorId::CYCLE_ACCURATE;
  std::chrono::milliseconds evalTimeoutMs = std::chrono::milliseconds(2000);
  // Pin dpus/tasklets to a fixed value instead of searching over them.
  // -1 means "search normally".
  int64_t fixedDpus = -1;
  int64_t fixedTasklets = -1;
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
  using SimFn = std::function<Maybe<SimCost>(
      const cinm::ConfWrapper &, UpmemSimulator &, cinm::TrialInfo &)>;
  std::vector<SimFn> simulators_;

  std::unique_ptr<PassManager> frontPipeline;
  std::unique_ptr<PassManager> backPipeline;

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
  /// Everything up to and including bufferization. Split from the rest so the
  /// leaf tile sizes can be stamped on the launch bodies in between: those ops
  /// are created by --convert-cinm-to-cnm and so do not exist yet when the
  /// search space is built.
  static std::unique_ptr<PassManager> buildFrontPipeline(MLIRContext *ctx) {
    auto pm = std::make_unique<PassManager>(ctx);

    // Step 1: cinm → linalg. The distribution below reads indexing maps, so
    // it needs the ops in structured form; this is also where fusion will go
    // (design §G8), which is why it runs on the generic branch only.
    pm->addPass(cinm::createConvertCinmOpsToLinalgPass());
    pm->addPass(createCanonicalizerPass());

    // Step 2: distribute onto the workgroup, with the buffers in MRAM. The
    // launch bodies then compute on MRAM, and --upmem-tile-mram-buffers stages
    // them down to WRAM. No separate tiling round: `cnm.tile_sizes` is a block
    // size per iteration dimension and the workgroup takes the whole tile
    // space at once.
    {
      mlir::ConvertLinalgToCnmPassOptions cnmOpts;
      cnmOpts.bufferLevel = "mram";
      // Splitting a reduction rewrites the op and adds an iteration
      // dimension; the leaf tile sizes have to follow it, or
      // --upmem-tile-mram-buffers below silently stages the whole tile.
      cnmOpts.perDimAttrs = {UPMEMDialect::LEAF_TILE_SIZES_NAME.str()};
      pm->addPass(cnm::createConvertLinalgToCnmPass(cnmOpts));
    }
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
    // Linalg is *not* lowered to loops here: --upmem-tile-mram-buffers needs
    // to see the launch bodies as linalg ops on memrefs. That happens at the
    // start of the back pipeline instead.
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

    return pm;
  }

  /// Everything after the launch bodies have been staged down to the leaf
  /// level.
  static std::unique_ptr<PassManager> buildBackPipeline(MLIRContext *ctx) {
    auto pm = std::make_unique<PassManager>(ctx);

    // Staging has to see linalg on memrefs, so it runs after bufferization and
    // before linalg is lowered to loops.
    pm->addPass(createUpmemTileMRAMBuffersPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    pm->addPass(createConvertLinalgToAffineLoopsPass());
    // Keep the reduction accumulator in a register. Straight out of linalg the
    // innermost loop reloads and restores the output element on every
    // iteration; the hand-written templates carry it in an scf.for iter_arg by
    // construction, so without this the generic path pays two extra memory ops
    // per multiply-accumulate.
    pm->addNestedPass<func::FuncOp>(affine::createAffineScalarReplacementPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 6: cnm → upmem
    pm->addPass(cnm::createCnmEnsureScatterGatherContiguousPass());
    pm->addPass(cnm::createConvertCnmToUPMEMPass({}));
    pm->addPass(bufferization::createBufferLoopHoistingPass());
    auto nested = pm->nestAny();
    bufferization::buildBufferDeallocationPipeline(nested); //fixme
    pm->addPass(createCSEPass());
    pm->addPass(createUPMEMDedupKernelsPass());
    pm->addPass(createCSEPass());
    {
      // This needs to apply after cnm->upmem bc of some assumptions we make
      // there.
      auto &funcs = pm->nest<func::FuncOp>();
      funcs.addPass(affine::createLoopUnrollPass(4));
    }
    // The affine dialect is an artefact of lowering linalg above; the DPU
    // kernels have to leave here free of it, because the C translator that
    // consumes them does not register affine (the hand-written templates emit
    // scf directly, so this only bites the generic path). Last, so the affine
    // passes above still see affine loops.
    pm->addPass(createLowerAffinePass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    return pm;
  }

  // --- InferencePlugin interface ---

  std::unique_ptr<cinm::InferencePlugin> clone() const override {
    auto c = std::make_unique<UpmemInferencePlugin>(platform, opts,
                                                    simulator->clone());
    c->dpusVar_ = dpusVar_;
    c->taskletsVar_ = taskletsVar_;
    c->simulators_ = simulators_;
    // A clone evaluates trials, so it needs the search parameters recorded
    // when the space was built.
    c->opParams_ = opParams_;
    c->refBlock_ = refBlock_;
    return c;
  }

  void warmUp(mlir::MLIRContext *ctx) override {
    if (!frontPipeline) {
      frontPipeline = buildFrontPipeline(ctx);
      backPipeline = buildBackPipeline(ctx);
    }
    simulator->warmUp();
  }

  void printStats() const override { simulator->printStats(); }

  void handleGemv(cinm::GemvOp gemv, SpaceBuilder &b);
  void handleReduce(cinm::ReduceOp op, SpaceBuilder &b);
  void handleEltwise(cinm::ElementwiseOp op, SpaceBuilder &b);

  /// Record the search parameters `op`'s lowering needs. `op` belongs to the
  /// reference clone; see opParams_ for how it is found again in a trial.
  void recordParams(Operation *op, ArrayRef<SpaceValue> outerTile,
                    ArrayRef<SpaceValue> leafTile) {
    OpSearchParams params;
    params.outerTile.assign(outerTile.begin(), outerTile.end());
    params.leafTile.assign(leafTile.begin(), leafTile.end());
    opParams_.push_back({walkIndexOf(op), std::move(params)});
  }

  void initializeSpace(cinm::ComputeBlockOp refClone,
                       cinm::ConfigSpace &space) override {
    SpaceBuilder b;
    simulators_.clear();
    opParams_.clear();
    refBlock_ = refClone;
    const int64_t maxDpus =
        platform.getMaxNumRanks() * platform.getMaxNumDpusPerRank();
    const int64_t maxTasklets = platform.getMaxNumTasklets();
    dpusVar_ = opts.fixedDpus > 0
                   ? b.intRange("dpus", opts.fixedDpus, opts.fixedDpus)
                   : b.intRange("dpus", 1, maxDpus);
    taskletsVar_ =
        opts.fixedTasklets > 0
            ? b.intRange("tasklets", opts.fixedTasklets, opts.fixedTasklets)
            : b.intRange("tasklets", 1, maxTasklets);

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

  /// Position of `op` in a pre-order walk of the reference compute block.
  unsigned walkIndexOf(Operation *op) {
    unsigned index = 0, found = 0;
    refBlock_.getBody().walk([&](Operation *candidate) {
      if (candidate == op)
        found = index;
      ++index;
    });
    return found;
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

  Maybe<SimCost> evaluate(cinm::TrialInfo &trial) override {
    auto conf = trial.conf();
    int64_t dpus = dpusVar_[conf];
    int64_t tasklets = taskletsVar_[conf];

    MLIRContext *ctx = trial.computeBlock->getContext();
    // mlir::Location loc = trial.computeBlock->getLoc();
    trial.computeBlock.setPlatformAttr({});
    trial.computeBlock.setAcceleratorAttr(
        upmem::UpmemAcceleratorAttr::get(platform, 1, dpus, tasklets));

    if (opts.lowering == UpmemLoweringPath::TEMPLATES) {
      // Each op's hand-written generator produces the program directly.
      SimCost total;
      for (auto &sim : simulators_)
        total += TRY_GET(sim(conf, *simulator, trial));
      annotateCost(ctx, trial, total.total());
      return total;
    }

    TRY(runGenericLowering(trial));
    SimCost total = TRY_GET(simulator->simulate(trial.computeBlock.getBody()));
    annotateCost(ctx, trial, total.total());
    return total;
  }

  /// Lower `trial` through the real pass pipeline, stamping each stage's
  /// search parameters immediately before the pass that reads them.
  DiagnosedSilenceableFailure runGenericLowering(cinm::TrialInfo &trial) {
    mlir::Location loc = trial.computeBlock->getLoc();
    MLIRContext *ctx = trial.computeBlock->getContext();
    if (!frontPipeline) {
      frontPipeline = buildFrontPipeline(ctx);
      backPipeline = buildBackPipeline(ctx);
    }

    stampSearchParams(trial);
    TRY(runPipeline(frontPipeline.get(), loc, trial.module.get()));
    TRY(runPipeline(backPipeline.get(), loc, trial.module.get()));
    return DiagnosedSilenceableFailure::success();
  }

  void annotateCost(MLIRContext *ctx, cinm::TrialInfo &trial,
                    double cost) const {
    if (!opts.annotateOpCosts)
      return;
    OpBuilder b(ctx);
    trial.computeBlock->setAttr(kSimCostAttr, b.getF64FloatAttr(cost));
  }

private:
  /// The search parameters one op's lowering needs.
  struct OpSearchParams {
    /// Block size per iteration dimension for --convert-linalg-to-cnm, which
    /// decides how much of the iteration space each workgroup leaf gets.
    SmallVector<SpaceValue> outerTile;
    /// Block size per iteration dimension for --upmem-tile-mram-buffers, which
    /// decides how the launch body walks its buffers through the leaf memory
    /// level.
    SmallVector<SpaceValue> leafTile;
  };

  /// Search parameters per op, keyed by the op's position in a pre-order walk
  /// of the compute block rather than by Operation*: makeTrialInfo deep-clones
  /// the module without retaining an IRMapping, so a trial's ops are different
  /// pointers. The clone is structurally identical, so walk position is a
  /// stable correspondence.
  SmallVector<std::pair<unsigned, OpSearchParams>> opParams_;
  cinm::ComputeBlockOp refBlock_;

  /// Resolve the recorded parameters against this trial's configuration and
  /// stamp them on the ops that consume them.
  ///
  /// Both levels are stamped here, before anything runs, because
  /// --convert-cinm-ops-to-linalg carries discardable attributes onto the
  /// linalg op it produces and --convert-linalg-to-cnm carries them again into
  /// the launch body. Nothing has to find the op again half way down the
  /// pipeline.
  void stampSearchParams(cinm::TrialInfo &trial) const {
    llvm::DenseMap<unsigned, const OpSearchParams *> byIndex;
    for (auto &[index, params] : opParams_)
      byIndex[index] = &params;

    auto resolve = [&](ArrayRef<SpaceValue> exprs) {
      SmallVector<int64_t> sizes;
      for (const SpaceValue &expr : exprs)
        sizes.push_back(expr(trial.conf()));
      return sizes;
    };

    unsigned index = 0;
    trial.computeBlock.getBody().walk([&](Operation *op) {
      auto it = byIndex.find(index++);
      if (it == byIndex.end())
        return;
      MLIRContext *ctx = op->getContext();
      if (!it->second->outerTile.empty())
        op->setAttr(cnm::CnmDialect::TILE_SIZES_NAME,
                    DenseI64ArrayAttr::get(ctx, resolve(it->second->outerTile)));
      if (!it->second->leafTile.empty())
        op->setAttr(UPMEMDialect::LEAF_TILE_SIZES_NAME,
                    DenseI64ArrayAttr::get(ctx, resolve(it->second->leafTile)));
    });
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

void UpmemInferencePlugin::handleGemv(cinm::GemvOp gemv, SpaceBuilder &b) {
  auto lhsTy = gemv.getLhs().getType();
  const int64_t M = lhsTy.getShape()[0], K = lhsTy.getShape()[1];
  auto eltTy = lhsTy.getElementType();
  auto wramLevel = platform.getWramLevel();
  auto mramLevel = platform.getMramLevel();
  const bool mramTiling = opts.useMRAMTiling;
  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;

  auto taskletCols = b.divisorsOf("taskletCols", tasklets);

  // Hardware dimensions.
  b.mustDivide(tasklets, M); // tasklets must divide M

  // WRAM tile dims: each is a divisor of its corresponding problem dimension.
  // K = dpuCols * mramCols * k1  ⟹  dpuCols | K  and  dpuCols | dpus
  auto wramRow = b.divisorsOf("wramRow", M);
  auto wramCol = b.divisorsOf("wramCol", K);
  auto dpuCols = b.divisorsOf("dpuCols", K);

  // Per-tasklet WRAM must fit: A tile (wr×wc, same total size whether split
  // or not) + x slice (now taskletCols*wc, shared but column-split) + y
  // slots (T×wr, same total size) + merge scratch for the MRAM-resident
  // running total ((T/taskletCols)×wr, one wr-slice per row group)
  b.require(tasklets * wramRow * wramCol + taskletCols * wramCol +
                tasklets * wramRow + (tasklets / taskletCols) * wramRow <=
            wramLevel.getSizeInElements(eltTy));

  auto mramRow = b.divisorsOf("mramRow", M);
  auto mramCol = b.divisorsOf("mramCol", K);

  // What the generic pipeline needs, projected from the template's parameters
  // (design §G2). The two paths search one space; they differ only in how they
  // read it.
  //
  // mramRow/mramCol are per-*DPU*, and the tasklets of a DPU subdivide that
  // tile -- see the mramRow/mramCol constraints below. A CNM leaf is a
  // tasklet, so a leaf's share of the iteration space is
  //
  //     b_m = mramRow / taskletRows = mramRow * taskletCols / tasklets
  //     b_k = mramCol / taskletCols
  //
  // Both divisions are exact given those constraints; multiplying before
  // dividing keeps them exact here too.
  //
  // The generic path's own requirement -- that the tile counts fill the
  // workgroup exactly -- then follows from the constraints below rather than
  // being an extra restriction:
  //
  //     (M/b_m) * (K/b_k) = (dpuRows*taskletRows) * (dpuCols*taskletCols)
  //                       = dpus * tasklets
  //
  // --convert-linalg-to-cnm checks that anyway, so a wrong projection fails
  // loudly instead of silently mis-tiling. That check is what M7 lacked.
  recordParams(gemv,
               {spaceValue(mramRow * taskletCols / tasklets),
                spaceValue(mramCol / taskletCols)},
               {spaceValue(wramRow), spaceValue(wramCol)});

  if (!mramTiling) {
    // In this mode we imitate cinm 1.0 behavior and do not tile on MRAM,
    // equivalently this means the MRAM and WRAM tiles have the same dimensions.
    // This corresponds to constraints:
    // - mramRow := wramRow * (tasklets / taskletCols)
    // - mramCol := wramCol * taskletCols
    b.require(
        [=](auto c) -> bool {
          return mramRow[c] == (wramRow[c] * tasklets[c] / taskletCols[c]) &&
                 mramCol[c] == wramCol[c] * taskletCols[c];
        },
        "mramRow == wramRow * tasklets / taskletCols && "
        "mramCol == wramCol * taskletCols");
  }

  b.require(M / ((dpus / dpuCols) * mramRow));
  b.require(mramRow / ((tasklets / taskletCols) * wramRow));
  b.require(mramCol / (taskletCols * wramCol));
  b.require(K / (dpuCols * mramCol));

  // Per-DPU MRAM must fit: A (T×mr×mc) + x (mc) + y (T×mr)
  b.require(mramRow * mramCol + mramCol + mramRow <=
            mramLevel.getSizeInElements(eltTy));

  // Simulation template for the MRAM fast path (bypasses the lowering
  // pipeline).
  registerSimulator([=](const cinm::ConfWrapper &c, UpmemSimulator &sim,
                        cinm::TrialInfo &trial) -> Maybe<SimCost> {
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
      generateGemv(op, rewriter, dpus[c] / dpuCols[c], dpuCols[c], mramRow[c],
                   mramCol[c], wramRow[c], wramCol[c],
                   tasklets[c] / taskletCols[c], taskletCols[c]);
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

    return TRY_GET(sim.simulate(trial.computeBlock.getBody()));
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

  auto mramRow = b.divisorsOf("mramRow", M);
  auto mramCol = b.divisorsOf("mramCol", K);

  // See handleGemv for the projection; the decomposition is identical.
  recordParams(op,
               {spaceValue(mramRow * taskletCols / tasklets),
                spaceValue(mramCol / taskletCols)},
               {spaceValue(wramRow), spaceValue(wramCol)});

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
    b.require(
        [=](auto c) -> bool {
          return mramRow[c] == (wramRow[c] * tasklets[c] / taskletCols[c]) &&
                 mramCol[c] == wramCol[c] * taskletCols[c];
        },
        "mramRow == wramRow * tasklets / taskletCols && "
        "mramCol == wramCol * taskletCols (MRAM trip count == 1)");
  }

  // Per-DPU MRAM must fit: A (T×mr×mc) + y (T×mr)
  b.require(mramRow * mramCol + mramRow <= mramLevel.getSizeInElements(eltTy));

  // Simulation template for the MRAM fast path (bypasses the lowering
  // pipeline).
  if (op.getDimension() == type.getShape().size() - 1) {

    // todo register simulator for specific op, here we assume
    //  that there is a single op in the compute block
    registerSimulator([=](const cinm::ConfWrapper &c, UpmemSimulator &sim,
                          cinm::TrialInfo &trial) -> Maybe<SimCost> {
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

      return TRY_GET(sim.simulate(trial.computeBlock.getBody()));
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
    o.sampleN = sampleN;
    o.sampleMaxCostMs = sampleMaxCostMs;
    o.nValidation = nValidation;
    o.validationInterval = validationInterval;
    o.objectiveScale = objectiveScale;
    o.numWorkers = numWorkers;
    o.dumpFullPool = dumpFullPool;
    upmemOpts.annotateOpCosts = annotateOpCosts;
    upmemOpts.useMRAMTiling = useMRAMTiling;
    upmemOpts.lowering = lowering;
    upmemOpts.fixedDpus = fixedDpus;
    upmemOpts.fixedTasklets = fixedTasklets;
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
        // here. Neither exhaustive search nor random sampling are seeded BO
        // runs, so both dump straight to the base dir.
        if (!upmemOpts.inference.exhaustiveSearch &&
            !upmemOpts.inference.sampleN && upmemOpts.inference.nSeeds <= 1)
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
