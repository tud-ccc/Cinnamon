#include <armadillo>
#include <cinm-mlir/Conversion/CinmPasses.h>
#include <cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h>
#include <cinm-mlir/Conversion/CommonPatterns.h>
#include <cinm-mlir/Conversion/LinalgToCnm/LinalgToCnm.h>
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
#include <upmem_cost_model/Types.h>

#include "SimulatorBase.h"

#include <algorithm>
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
#include <mlir/Dialect/Bufferization/Pipelines/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
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
using mlir::cinm::ConfigurationVector;
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

/// `n!`. Only ever called on an iteration rank, which is 1 (elementwise), 2
/// (gemv) or 3 (gemm) in everything we lower today.
static int64_t factorial(unsigned n) {
  assert(n <= 20 && "factorial would overflow");
  int64_t result = 1;
  for (unsigned i = 2; i <= n; ++i)
    result *= i;
  return result;
}

/// UPMEM-specific inference options. Wraps the generic InferenceOptions and
/// provides a place to add UPMEM-specific knobs in the future.
struct UpmemInferenceOptions {
  cinm::InferenceOptions inference;
  bool annotateOpCosts = false;
  bool useMRAMTiling = true;
  bool debugPrintsInPipeline = false;
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
  pm.addPass(affine::createAffineLoopInvariantCodeMotionPass());
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

/// `cinm.op.gemv` -> `gemv`, for use in search-parameter names.
static std::string shortOpName(StringRef opName) {
  auto [prefix, last] = opName.rsplit('.');
  return (last.empty() ? opName : last).str();
}

/// Whether the space declares parameters for `op` -- whether, in other words,
/// it is an op the pipeline will distribute onto the workgroup.
///
/// This used to be "carries a `cinm.lowered_from` marker", which
/// `--convert-cinm-ops-to-linalg` sets on exactly the ops that inherit a cinm
/// op's computation. That premise does not survive the pipeline: elementwise
/// fusion builds a fresh `linalg.generic` from several ops and, like upstream
/// rewrites generally, does not carry discardable attributes onto it. A fused
/// body then looked like nothing but ops produced *alongside* a computation:
/// the space declared no parameters, nothing was stamped, and
/// `--convert-linalg-to-cnm` skipped an op with no tile sizes -- so the block
/// lowered to a host-side loop nest with no diagnostic at all.
///
/// The test is structural instead, which no future rewrite can invalidate by
/// dropping an attribute. The marker keeps the one job it can still do: naming
/// the parameters.
static bool isDistributionCandidate(Operation *op) {
  auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalgOp.hasPureTensorSemantics())
    return false;
  // What the marker was really distinguishing: an op whose result only ever
  // lands in another linalg op's `outs` is that op's *init* -- a `linalg.fill`
  // seeding an accumulator, typically. Distributing it would spread the
  // initialization as though it were the computation.
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses()) {
      auto consumer = llvm::dyn_cast<linalg::LinalgOp>(use.getOwner());
      if (!consumer || !consumer.isDpsInit(&use))
        return true;
    }
  return false;
}

/// What to call `op`'s parameters: the cinm op it was lowered from while that
/// is still recorded, and otherwise the linalg op it *is*. A fused op comes
/// from several cinm ops, so naming it after any one of them would be a guess;
/// `generic.D00` at least says what it is.
static std::string searchNameFor(Operation *op) {
  if (auto origin =
          op->getAttrOfType<StringAttr>(cinm::CinmDialect::LOWERED_FROM_NAME))
    return shortOpName(origin.getValue());
  return shortOpName(op->getName().getStringRef());
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

  std::unique_ptr<PassManager> convertPipeline;
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
  /// cinm -> linalg, run on its own so the search parameters can be stamped
  /// in between. The space is stated in terms of an *iteration space*, and
  /// only linalg carries one; the walk positions recorded when the space was
  /// built therefore address the converted ops, not the cinm ops. This is also
  /// where fusion will go (design §G8), which is why it is on the generic
  /// branch only.
  static std::unique_ptr<PassManager> buildConvertPipeline(MLIRContext *ctx,
                                                           bool debug) {
    auto pm = std::make_unique<PassManager>(ctx);
    pm->addPass(cinm::createConvertCinmOpsToLinalgPass());
    pm->addPass(mlir::createConvertTensorToLinalgPass());
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-cinm-to-linalg"}));
    // pm->addPass(linalg::createLinalgGeneralizeNamedOpsPass());
    pm->addPass(createLinalgElementwiseOpFusionPass());
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-fusion"}));
    // Scalar values used directly inside a linalg op need to be
    // promoted to operands of the linalg op, so that the body
    // of the generic op becomes IsolatedFromAbove.
    pm->addPass(cnm::createCnmIsolateLinalgCapturesPass());
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-isolate-captures"}));
    return pm;
  }

  static std::unique_ptr<PassManager> buildFrontPipeline(MLIRContext *ctx,
                                                         bool debug) {
    auto pm = std::make_unique<PassManager>(ctx);

    // Step 2: distribute onto the workgroup, with the buffers in MRAM. The
    // launch bodies then compute on MRAM, and --upmem-tile-mram-buffers stages
    // them down to WRAM. No separate tiling round: `cnm.tile_sizes` is a block
    // size per iteration dimension and the workgroup takes the whole tile
    // space at once.
    if (debug)
      pm->addPass(createPrintIRPass({.label = "before-linalg-to-cnm"}));
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
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-linalg-to-cnm"}));
    pm->addPass(cnm::createCnmHoistWorkgroupsPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 3: bufferize
    pm->addPass(bufferization::createEmptyTensorEliminationPass());
    pm->addPass(cnm::createCnmScatterOptimizationsPass());
    pm->addPass(createCSEPass());
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "before-bufferization"}));
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
  static std::unique_ptr<PassManager> buildBackPipeline(MLIRContext *ctx,
                                                        bool debug) {
    auto pm = std::make_unique<PassManager>(ctx);

    // Staging has to see linalg on memrefs, so it runs after bufferization and
    // before linalg is lowered to loops.
    pm->addPass(createUpmemTileMRAMBuffersPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-tile-mram-buffers"}));
    pm->addPass(createConvertLinalgToAffineLoopsPass());
    // Keep the reduction accumulator in a register. Straight out of linalg the
    // innermost loop reloads and restores the output element on every
    // iteration; the hand-written templates carry it in an scf.for iter_arg by
    // construction, so without this the generic path pays two extra memory ops
    // per multiply-accumulate.
    pm->addNestedPass<func::FuncOp>(
        affine::createAffineScalarReplacementPass());
    auto &funcPm = pm->nest<func::FuncOp>();
    addAffineOpts(funcPm);
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 6: cnm → upmem
    pm->addPass(cnm::createCnmEnsureScatterGatherContiguousPass());
    pm->addPass(cnm::createConvertCnmToUPMEMPass({}));
    // Right after the conversion, so the rest of the back pipeline sees the
    // narrowest transfer form each map allows -- in particular the occupancy
    // check and the cost model, which read the ops' shapes.
    pm->addPass(createUpmemSpecializeTransfersPass());
    pm->addPass(bufferization::createBufferLoopHoistingPass());
    {
      // auto &nested = pm->nestAny();
      // bufferization::buildBufferDeallocationPipeline(nested); // fixme
    }
    pm->addPass(createCSEPass());
    pm->addPass(createUPMEMDedupKernelsPass());
    pm->addPass(createCSEPass());
    {
      // This needs to apply after cnm->upmem bc of some assumptions we make
      // there.
      // auto &funcs = pm->nest<func::FuncOp>();
      // funcs.addPass(affine::createLoopUnrollPass(4));
    }
    // The affine dialect is an artefact of lowering linalg above; the DPU
    // kernels have to leave here free of it, because the C translator that
    // consumes them does not register affine (the hand-written templates emit
    // scf directly, so this only bites the generic path). Last, so the affine
    // passes above still see affine loops.
    {
      auto &dpuPm = pm->nest<ModuleOp>().nest<DpuProgramOp>();
      addAffineOpts(dpuPm);
      dpuPm.addPass(createLowerAffinePass());
    }
    // The C translator addresses a buffer as base pointer + one linear
    // offset, so it can only express a single subview. Staging a tile of a
    // tasklet's slice naturally produces two nested ones; compose them.
    pm->addPass(memref::createFoldMemRefAliasOpsPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    // Last, deliberately: what a configuration occupies is a property of the
    // program every pass above has finished optimizing, not of the
    // configuration itself (design §H3). A trial that does not fit fails here
    // and the search moves on, rather than being discovered when the DPU
    // binary fails to link.
    pm->addPass(createUpmemCheckOccupancyPass());
    if (debug)
      LLVM_DEBUG(pm->dump(););
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
      convertPipeline = buildConvertPipeline(ctx, opts.debugPrintsInPipeline);
      frontPipeline = buildFrontPipeline(ctx, opts.debugPrintsInPipeline);
      backPipeline = buildBackPipeline(ctx, opts.debugPrintsInPipeline);
    }
    simulator->warmUp();
  }

  void printStats() const override { simulator->printStats(); }

  /// Register the hand-written template for an op, deriving the generator's
  /// arguments from the generic block sizes (design §H5). Only called when
  /// the template path is selected; the templates declare no variables of
  /// their own.
  void registerGemvTemplate(SpaceBuilder &b, ArrayRef<SpaceVar> blocks,
                            ArrayRef<SpaceVar> leaves,
                            ArrayRef<int64_t> extents, Type eltTy);
  void registerReduceTemplate(SpaceBuilder &b, ArrayRef<SpaceVar> blocks,
                              ArrayRef<SpaceVar> leaves,
                              ArrayRef<int64_t> extents, Type eltTy);

  /// The search space for one op, stated in the parameters the passes
  /// actually consume (design §G2, §H5): one block size per iteration
  /// dimension for the workgroup distribution, and one for the leaf level.
  ///
  /// This is the whole space. `dpuRows`/`dpuCols`/`taskletRows`/`taskletCols`
  /// and the MRAM/WRAM tile pairs are not parameters -- they are a *reading*
  /// of these numbers that the template path derives when it needs them.
  void handleLinalgOp(linalg::LinalgOp op, StringRef namePrefix,
                      unsigned walkIndex, SpaceBuilder &b);

  /// Record the search parameters `op`'s lowering needs. `op` belongs to the
  /// reference clone; see opParams_ for how it is found again in a trial.
  void recordParams(unsigned walkIndex, ArrayRef<SpaceValue> outerTile,
                    ArrayRef<SpaceValue> leafTile, SpaceValue order = {}) {
    OpSearchParams params;
    params.outerTile.assign(outerTile.begin(), outerTile.end());
    params.leafTile.assign(leafTile.begin(), leafTile.end());
    params.order = std::move(order);
    opParams_.push_back({walkIndex, std::move(params)});
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

    // The space is derived from the *linalg* form of the block. Block sizes
    // are indexed by iteration dimension and only linalg states an iteration
    // space; deriving them from cinm ops instead would mean maintaining a
    // second, hand-written notion of each op's iteration space. That notion
    // already disagrees: `getTilableDimSizes` reports one flattened dimension
    // for an elementwise op where its linalg form has one per rank.
    //
    // A throwaway copy is converted here purely to read those iteration
    // spaces. Trials are clones of the *unconverted* reference and run the
    // same conversion as their first pipeline step, so a walk position in
    // this copy addresses the same op in a trial.
    OwningOpRef<ModuleOp> converted(
        llvm::cast<ModuleOp>(refClone->getParentOfType<ModuleOp>()->clone()));
    {
      auto pm = buildConvertPipeline(refClone->getContext(),
                                     opts.debugPrintsInPipeline);
      if (failed(pm->run(converted.get()))) {
        refClone->emitError("could not convert the compute block to linalg, "
                            "so no search space can be derived from it");
        return;
      }
    }

    cinm::ComputeBlockOp convertedBlock;
    converted->walk([&](cinm::ComputeBlockOp op) { convertedBlock = op; });
    if (!convertedBlock) {
      refClone->emitError("the converted reference has no compute block");
      return;
    }

    // Name each op's parameters after the cinm op it came from, e.g.
    // `gemv.M0`. Count the kinds first so that a block with two gemvs gets
    // `gemv0`/`gemv1` while the common single-op case stays unadorned.
    llvm::StringMap<unsigned> kindCount;
    convertedBlock.getBody().walk([&](Operation *op) {
      if (isDistributionCandidate(op))
        ++kindCount[searchNameFor(op)];
    });

    // Walk the *body*, which is exactly what stampSearchParams walks in a
    // trial, so the recorded positions mean the same thing on both sides.
    llvm::StringMap<unsigned> kindSeen;
    unsigned walkIndex = 0;
    convertedBlock.getBody().walk([&](Operation *op) {
      unsigned here = walkIndex++;
      if (!isDistributionCandidate(op))
        return;
      std::string kind = searchNameFor(op);
      std::string prefix =
          kindCount[kind] > 1 ? kind + std::to_string(kindSeen[kind]++) : kind;
      handleLinalgOp(llvm::cast<linalg::LinalgOp>(op), prefix, here, b);
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
    // Diagnostics are swallowed because a rejected trial is ordinary during a
    // search -- but the first error is kept, so that the reason surfaces in
    // the failure a pinned `eval-solution` reports. Without it every rejection
    // reads "Pipeline failed", including the occupancy check's.
    std::string reason;
    ScopedDiagnosticHandler scopedHandler(
        pipeline->getContext(), [&reason](Diagnostic &diag) {
          if (reason.empty() && diag.getSeverity() == DiagnosticSeverity::Error)
            reason = diag.str();
          LLVM_DEBUG(llvm::dbgs()
                         << "[cinm-inference]   pipeline failed:\n      ";
                     diag.print(llvm::dbgs()); llvm::dbgs() << "\n";);
          return success();
        });
    if (mlir::failed(pipeline->run(module))) {
      LLVM_DEBUG(module->print(llvm::dbgs()); llvm::dbgs() << "\n========\n";);
      if (reason.empty())
        return mlir::emitSilenceableFailure(loc, "Pipeline failed");
      return mlir::emitSilenceableFailure(loc, "Pipeline failed: " + reason);
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
      convertPipeline = buildConvertPipeline(ctx, opts.debugPrintsInPipeline);
      frontPipeline = buildFrontPipeline(ctx, opts.debugPrintsInPipeline);
      backPipeline = buildBackPipeline(ctx, opts.debugPrintsInPipeline);
    }

    // Convert first, then stamp: the recorded walk positions address the
    // linalg ops, which is the form the space was derived from.
    TRY(runPipeline(convertPipeline.get(), loc, trial.module.get()));
    if (failed(stampSearchParams(trial)))
      return DiagnosedSilenceableFailure::definiteFailure();
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
    /// Which tile dimension varies fastest across the leaves, as a rank among
    /// the distinct orders (design §G3). Empty for ops with nothing to order.
    SpaceValue order;
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
  LogicalResult stampSearchParams(cinm::TrialInfo &trial) const {
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
    size_t stamped = 0;
    trial.computeBlock.getBody().walk([&](Operation *op) {
      auto it = byIndex.find(index++);
      if (it == byIndex.end())
        return;
      MLIRContext *ctx = op->getContext();
      if (!it->second->outerTile.empty())
        op->setAttr(
            cnm::CnmDialect::TILE_SIZES_NAME,
            DenseI64ArrayAttr::get(ctx, resolve(it->second->outerTile)));
      if (!it->second->leafTile.empty())
        op->setAttr(UPMEMDialect::LEAF_TILE_SIZES_NAME,
                    DenseI64ArrayAttr::get(ctx, resolve(it->second->leafTile)));
      // The index form rather than the permutation: the order is stated over
      // the dimensions the op has *after* --convert-linalg-to-cnm splits its
      // reductions, which have not been created yet, whereas the rank is the
      // same number here and there.
      if (it->second->order)
        op->setAttr(cnm::CnmDialect::WORKGROUP_DIM_ORDER_INDEX_NAME,
                    IntegerAttr::get(IntegerType::get(ctx, 64),
                                     it->second->order(trial.conf())));
      ++stamped;
    });

    // The trial is a clone of the reference, converted by the same pipeline,
    // so every recorded position must have been found. If not, the two have
    // drifted apart and stamping the wrong ops would mis-tile silently.
    if (stamped != opParams_.size())
      return trial.computeBlock->emitOpError()
             << "stamped " << stamped << " of " << opParams_.size()
             << " recorded ops; the trial no longer matches the reference the "
                "search space was built from";
    return success();
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

/// Human-readable names for an op's iteration dimensions, so the space reads
/// as `gemv.M0` -- the workgroup block on M -- rather than `op0.block0`.
/// Falls back to D0, D1, ... for ops without an established convention.
static SmallVector<std::string> iterationDimNames(StringRef origin,
                                                  unsigned rank) {
  auto fixed = [&](ArrayRef<StringRef> names)
      -> std::optional<SmallVector<std::string>> {
    if (names.size() != rank)
      return std::nullopt;
    SmallVector<std::string> out;
    for (StringRef name : names)
      out.push_back(name.str());
    return out;
  };

  std::optional<SmallVector<std::string>> named;
  if (origin == cinm::GemvOp::getOperationName())
    named = fixed({"M", "K"});
  else if (origin == cinm::GemmOp::getOperationName())
    named = fixed({"M", "N", "K"});
  else if (origin == cinm::BatchGemvOp::getOperationName())
    named = fixed({"B", "M", "K"});
  else if (origin == cinm::BatchGemmOp::getOperationName())
    named = fixed({"B", "M", "N", "K"});
  else if (origin == cinm::ReduceOp::getOperationName() && rank >= 2) {
    // This generator only handles a trailing reduction, so the leading
    // dimensions are the parallel ones.
    SmallVector<std::string> out;
    if (rank == 2) {
      out.push_back("M");
    } else {
      for (unsigned dim = 0; dim + 1 < rank; ++dim)
        out.push_back("P" + std::to_string(dim));
    }
    out.push_back("K");
    named = std::move(out);
  }
  if (named)
    return *named;

  SmallVector<std::string> out;
  for (unsigned dim = 0; dim < rank; ++dim)
    out.push_back("D" + std::to_string(dim));
  return out;
}

/// Loop extents of a linalg op, read off its operands. Only valid for
/// projected-permutation indexing maps, which is what --convert-linalg-to-cnm
/// requires anyway.
static FailureOr<SmallVector<int64_t>> linalgLoopExtents(linalg::LinalgOp op) {
  SmallVector<int64_t> extents(op.getNumLoops(), ShapedType::kDynamic);
  for (auto [operand, map] :
       llvm::zip(op->getOpOperands(), op.getIndexingMapsArray())) {
    if (!map.isProjectedPermutation())
      return failure();
    auto shape = cast<ShapedType>(operand.get().getType()).getShape();
    for (auto [pos, expr] : llvm::enumerate(map.getResults()))
      extents[cast<AffineDimExpr>(expr).getPosition()] = shape[pos];
  }
  if (llvm::any_of(extents, ShapedType::isDynamic))
    return failure();
  return extents;
}

/// The iteration dimensions each operand is indexed by.
static SmallVector<SmallVector<unsigned>>
linalgOperandDims(linalg::LinalgOp op) {
  SmallVector<SmallVector<unsigned>> dims;
  for (AffineMap map : op.getIndexingMapsArray()) {
    SmallVector<unsigned> operandDims;
    for (AffineExpr expr : map.getResults())
      operandDims.push_back(cast<AffineDimExpr>(expr).getPosition());
    dims.push_back(std::move(operandDims));
  }
  return dims;
}

void UpmemInferencePlugin::handleLinalgOp(linalg::LinalgOp op,
                                          StringRef namePrefix,
                                          unsigned walkIndex, SpaceBuilder &b) {
  FailureOr<SmallVector<int64_t>> extents = linalgLoopExtents(op);
  if (failed(extents))
    return; // Not distributable; contributes no parameters.

  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;
  Type eltTy = cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();

  // `<op>.<dim><level>`: level 0 is the block a workgroup leaf gets, level 1
  // the block it walks that in at the leaf memory level. So a gemv declares
  // gemv.M0, gemv.K0, gemv.M1, gemv.K1. These are the user-facing names that
  // eval-solution refers to.
  //
  // The marker is gone on anything a rewrite rebuilt -- a fused op, above all
  // -- and both readers of it degrade rather than fail: the dimensions fall
  // back to `D0, D1, ...`, and no template claims an op whose origin is not
  // recorded, which is right, since the templates implement particular cinm
  // ops and a fused op is no longer one of them.
  auto originAttr =
      op->getAttrOfType<StringAttr>(cinm::CinmDialect::LOWERED_FROM_NAME);
  StringRef origin = originAttr ? originAttr.getValue() : StringRef();
  SmallVector<std::string> dimNames =
      iterationDimNames(origin, extents->size());

  SmallVector<SpaceVar> blocks, leaves;
  for (auto [dim, extent] : llvm::enumerate(*extents)) {
    std::string base = (namePrefix + "." + dimNames[dim]).str();
    blocks.push_back(b.divisorsOf(base + "0", extent));
    leaves.push_back(b.divisorsOf(base + "1", blocks.back()));
  }

  // The tile counts must fill the workgroup exactly (design §G2). This is the
  // one structural constraint; everything else about the distribution follows
  // from the block sizes and the op's own indexing maps.
  SmallVector<int64_t> extentsCopy(*extents);
  b.require(
      [=](auto &c, auto &valid) {
        auto tiles = c.ones();
        cinm::ParmVector extentRow(c.size());
        for (auto [extent, block] : llvm::zip_equal(extentsCopy, blocks)) {
          extentRow.fill(extent);
          // Every block domain is a subset of [1, extent] (SpaceBuilder::
          // divisorsOf), so the divisor is never zero and the division needs
          // no guard. `%` is arma's elementwise multiply throughout, so the
          // test below is `quotient * block == extent`, i.e. exact division.
          const cinm::ParmVector &blockRow = block[c];
          cinm::ParmVector quotient = extentRow / blockRow;
          valid %= (quotient % blockRow == extentRow);
          tiles %= quotient;
        }
        valid %= (tiles == (dpus[c] % tasklets[c]));
      },
      "prod(extent / block) == dpus * tasklets");

  // Capacity, as a *necessary* condition only (design §H4). Assume maximal
  // sharing -- every operand stored once per DPU -- so the bound can never
  // reject a configuration that would have fitted. What actually fits depends
  // on decisions taken during lowering (which operands end up shared, how
  // promotion sizes its staging buffers, where buffers are hoisted), so the
  // exact test is done on the lowered program instead.
  auto operandDims = linalgOperandDims(op);
  auto footprint =
      [operandDims](ArrayRef<SpaceVar> sizes,
                    const ConfigurationVector &c) -> cinm::ParmVector {
    cinm::ParmVector total = c.zeros();
    for (const auto &dims : operandDims) {
      auto elements = c.ones();
      for (unsigned dim : dims)
        elements %= sizes[dim][c]; // elementwise multiply.
      total += elements;
    }
    return total;
  };

  const int64_t mramElements = platform.getMramLevel().getSizeInElements(eltTy);
  const int64_t wramElements = platform.getWramLevel().getSizeInElements(eltTy);
  b.require([=](auto &c,
                auto &valid) { valid %= footprint(blocks, c) <= mramElements; },
            "sum of per-leaf operand tiles <= MRAM (assuming maximal sharing)");
  b.require([=](auto &c,
                auto &valid) { valid %= footprint(leaves, c) <= wramElements; },
            "sum of leaf tiles <= WRAM (assuming maximal sharing)");
  if (!opts.useMRAMTiling) {
    // Note: this is only required for benchmarks that compare
    // against CINM1 codegen. To be removed.
    b.require(
        [=](auto &c, auto &valid) {
          valid %= footprint(blocks, c) <= wramElements;
        },
        "MRAM tile should be equal to WRAM tile (no tiling in MRAM)");
  }

  // Which tile dimension varies fastest across the leaves (design §G3). The
  // one parameter here that is not a size: it decides what the leaves sharing
  // a hardware node share rather than replicate, which the block sizes cannot
  // state. `<op>.order` ranks the distinct orders lexicographically, with the
  // default rule at 0.
  //
  // The domain is `numLoops!` because the reduction split leaves at most one
  // distributed dimension per iteration dimension: a split dimension carries
  // the tile count and its remainder is left with 1. How many there actually
  // are depends on the block sizes, so the rest of the range is pruned by a
  // predicate -- an index the op has no order for is a configuration the space
  // does not offer, not a trial that fails.
  SpaceValue order;
  if (extents->size() >= 2) {
    // The templates implement one fixed mapping and read the rest of this
    // space through a projection (§H5). Declaring the variable for them too
    // keeps one space and one set of parameter names across both paths;
    // pinning it to 0 keeps the search from spending trials on a parameter
    // that path ignores.
    int64_t numOrders = opts.lowering == UpmemLoweringPath::TEMPLATES
                            ? 1
                            : factorial(extents->size());
    SpaceVar orderVar =
        b.intRange((namePrefix + ".order").str(), 0, numOrders - 1);
    b.require(
        [=](const ConfigurationVector &c, arma::urowvec &valid) {
          arma::urowvec distributed(c.size(), arma::fill::zeros);
          cinm::ParmVector extentRow(c.size());
          for (auto [extent, block] : llvm::zip_equal(extentsCopy, blocks)) {
            extentRow.fill(extent);
            // Divisor is never zero -- see the tile-count constraint above.
            const cinm::ParmVector &blockRow = block[c];
            cinm::ParmVector quotient = extentRow / blockRow;
            arma::urowvec dimDistributed =
                (quotient % blockRow == extentRow) % (quotient > 1);
            distributed += dimDistributed;
          }

          // factorial() has no simple vectorized form, but `distributed`
          // counts dimensions spread over the workgroup, so it is bounded by
          // the number of iteration dimensions -- a handful at most. Walk
          // that range directly, carrying the factorial forward from the
          // previous stop, and settle every lane at a stop in one shot,
          // instead of recomputing factorial(distributed[i]) per lane.
          // Enumerating the bound also beats discovering the distinct values
          // with arma::unique, which would sort the whole row (O(n log n)) to
          // recover at most numDims+1 stops.
          const cinm::ParmVector &orderRow = orderVar[c];
          const uint64_t maxDistributed =
              static_cast<uint64_t>(extentsCopy.size());
          uint64_t runningFactorial = 1; // == stop!
          for (uint64_t stop = 0; stop <= maxDistributed; ++stop) {
            if (stop > 0)
              runningFactorial *= stop;
            // Lanes already cleared by an earlier constraint stay cleared:
            // %= multiplies into the existing 0.
            arma::uvec lanes = arma::find(distributed == stop);
            valid.elem(lanes) %= (orderRow.elem(lanes) < runningFactorial);
          }
        },
        "order < (number of dimensions spread over the workgroup)!");
    order = spaceValue(orderVar);
  }

  SmallVector<SpaceValue> outerTile, leafTile;
  for (const SpaceVar &var : blocks)
    outerTile.push_back(spaceValue(var));
  for (const SpaceVar &var : leaves)
    leafTile.push_back(spaceValue(var));
  recordParams(walkIndex, outerTile, leafTile, order);

  // The hand-written templates read this same space through a projection.
  // Dispatch on which cinm op this came from rather than on the linalg op's
  // type: the generators still rewrite the cinm op in the trial, so the two
  // must agree on what they are looking at.
  if (opts.lowering == UpmemLoweringPath::TEMPLATES) {
    if (origin == cinm::GemvOp::getOperationName())
      registerGemvTemplate(b, blocks, leaves, *extents, eltTy);
    else if (origin == cinm::ReduceOp::getOperationName())
      registerReduceTemplate(b, blocks, leaves, *extents, eltTy);
  }
}

/// Read the generic block sizes as the gemv template generator's arguments
/// (design §H5). §G3 fixes the layout -- the tasklets of a DPU split the
/// parallel dimension -- which pins the reading exactly:
///
///   taskletRows = min(tasklets, mTiles)   taskletCols = tasklets/taskletRows
///   mramRow     = blockM * taskletRows    mramCol     = blockK * taskletCols
///   dpuRows     = mTiles / taskletRows    dpuCols     = kTiles / taskletCols
///   wramRow     = leafM                   wramCol     = leafK
struct GemvTemplateArgs {
  int64_t dpuRows, dpuCols, taskletRows, taskletCols;
  int64_t mramRow, mramCol, wramRow, wramCol;
};

static std::optional<GemvTemplateArgs>
readAsGemvTemplate(int64_t M, int64_t K, int64_t dpus, int64_t tasklets,
                   int64_t blockM, int64_t blockK, int64_t leafM,
                   int64_t leafK) {
  if (blockM <= 0 || blockK <= 0 || M % blockM || K % blockK)
    return std::nullopt;
  const int64_t mTiles = M / blockM, kTiles = K / blockK;
  const int64_t taskletRows = std::min<int64_t>(tasklets, mTiles);
  if (taskletRows <= 0 || tasklets % taskletRows)
    return std::nullopt;
  const int64_t taskletCols = tasklets / taskletRows;
  if (mTiles % taskletRows || kTiles % taskletCols)
    return std::nullopt;
  const int64_t dpuRows = mTiles / taskletRows, dpuCols = kTiles / taskletCols;
  if (dpus != dpuRows * dpuCols)
    return std::nullopt;
  return GemvTemplateArgs{dpuRows,
                          dpuCols,
                          taskletRows,
                          taskletCols,
                          blockM * taskletRows,
                          blockK * taskletCols,
                          leafM,
                          leafK};
}

void UpmemInferencePlugin::registerGemvTemplate(SpaceBuilder &b,
                                                ArrayRef<SpaceVar> blocks,
                                                ArrayRef<SpaceVar> leaves,
                                                ArrayRef<int64_t> extents,
                                                Type eltTy) {
  // linalg.matvec iterates (m, k).
  const int64_t M = extents[0], K = extents[1];
  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;
  SpaceVar blockM = blocks[0], blockK = blocks[1];
  SpaceVar leafM = leaves[0], leafK = leaves[1];
  const int64_t wramElements = platform.getWramLevel().getSizeInElements(eltTy);
  const int64_t mramElements = platform.getMramLevel().getSizeInElements(eltTy);

  auto derive = [=](const cinm::ConfWrapper &c) {
    return readAsGemvTemplate(M, K, dpus[c], tasklets[c], blockM[c], blockK[c],
                              leafM[c], leafK[c]);
  };

  // Under `lowering=templates` there is nothing else to run, so restricting
  // the space to what the template layout can express is right here -- unlike
  // the capacity bounds in handleLinalgOp, which must stay necessary-only.
  b.require([=](const cinm::ConfWrapper &c) { return derive(c).has_value(); },
            "the gemv template layout can express this configuration");
  b.require(
      [=](const cinm::ConfWrapper &c) -> bool {
        auto d = derive(c);
        return d && d->wramRow > 0 && d->wramCol > 0 &&
               d->mramRow % (d->taskletRows * d->wramRow) == 0 &&
               d->mramCol % (d->taskletCols * d->wramCol) == 0;
      },
      "the WRAM tile divides the MRAM tile");
  b.require(
      [=](const cinm::ConfWrapper &c) -> bool {
        auto d = derive(c);
        if (!d)
          return false;
        // Per-tasklet WRAM: A tile + the column-split x slice + y slots +
        // merge scratch for the MRAM-resident running total.
        const int64_t t = tasklets[c];
        return t * d->wramRow * d->wramCol + d->taskletCols * d->wramCol +
                   t * d->wramRow + (t / d->taskletCols) * d->wramRow <=
               wramElements;
      },
      "the gemv template's WRAM working set fits");
  b.require(
      [=](const cinm::ConfWrapper &c) -> bool {
        auto d = derive(c);
        // Per-DPU MRAM: A (mr x mc) + x (mc) + y (mr).
        return d && d->mramRow * d->mramCol + d->mramCol + d->mramRow <=
                        mramElements;
      },
      "the gemv template's MRAM working set fits");

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
    TRY(runPipeline(bufferizePm.get(), trial.computeBlock->getLoc(),
                    trial.module.get()));

    IRRewriter rewriter(trial.module->getContext());
    rewriter.setInsertionPointToStart(&trial.computeBlock.getBody().front());

    auto args = derive(c);
    if (!args)
      return Maybe<SimCost>(emitSilenceableFailure(
          trial.computeBlock->getLoc(),
          "the gemv template cannot express this configuration"));
    trial.computeBlock->walk([&](cinm::GemvOp op) {
      generateGemv(op, rewriter, args->dpuRows, args->dpuCols, args->mramRow,
                   args->mramCol, args->wramRow, args->wramCol,
                   args->taskletRows, args->taskletCols);
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

    TRY(runPipeline(cleanupPm.get(), trial.computeBlock->getLoc(),
                    trial.module.get()));

    return TRY_GET(sim.simulate(trial.computeBlock.getBody()));
  });
}

void UpmemInferencePlugin::registerReduceTemplate(SpaceBuilder &b,
                                                  ArrayRef<SpaceVar> blocks,
                                                  ArrayRef<SpaceVar> leaves,
                                                  ArrayRef<int64_t> extents,
                                                  Type eltTy) {
  // The template treats a reduction as a 2-D (M, K) problem: M is the product
  // of the parallel extents, K the reduction extent. linalg.reduce iterates
  // the input's dimensions in order, and this generator only handles a
  // trailing reduction, so the parallel dimensions are the leading ones.
  if (extents.size() < 2)
    return;
  const int64_t M = computeProduct(extents.drop_back());
  const int64_t K = extents.back();
  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;
  SmallVector<SpaceVar> parBlocks(blocks.drop_back());
  SmallVector<SpaceVar> parLeaves(leaves.drop_back());
  SpaceVar blockK = blocks.back(), leafK = leaves.back();
  const int64_t wramElements = platform.getWramLevel().getSizeInElements(eltTy);
  const int64_t mramElements = platform.getMramLevel().getSizeInElements(eltTy);

  auto derive = [=](const cinm::ConfWrapper &c) {
    int64_t blockM = 1, leafM = 1;
    for (const SpaceVar &var : parBlocks)
      blockM *= var[c];
    for (const SpaceVar &var : parLeaves)
      leafM *= var[c];
    return readAsGemvTemplate(M, K, dpus[c], tasklets[c], blockM, blockK[c],
                              leafM, leafK[c]);
  };

  b.require([=](const cinm::ConfWrapper &c) { return derive(c).has_value(); },
            "the reduction template layout can express this configuration");
  b.require(
      [=](const cinm::ConfWrapper &c) -> bool {
        auto d = derive(c);
        return d && d->wramRow > 0 && d->wramCol > 0 &&
               d->mramRow % (d->taskletRows * d->wramRow) == 0 &&
               d->mramCol % (d->taskletCols * d->wramCol) == 0;
      },
      "the WRAM tile divides the MRAM tile");
  b.require(
      [=](const cinm::ConfWrapper &c) -> bool {
        auto d = derive(c);
        return d && d->wramCol * d->wramRow * tasklets[c] + tasklets[c] <=
                        wramElements;
      },
      "the reduction template's WRAM working set fits");
  b.require(
      [=](const cinm::ConfWrapper &c) -> bool {
        auto d = derive(c);
        // Per-DPU MRAM: input (mr x mc) + output (mr).
        return d && d->mramRow * d->mramCol + d->mramRow <= mramElements;
      },
      "the reduction template's MRAM working set fits");

  {

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
      TRY(runPipeline(bufferizePm.get(), trial.computeBlock->getLoc(),
                      trial.module.get()));

      IRRewriter rewriter(trial.module->getContext());
      rewriter.setInsertionPointToStart(&trial.computeBlock.getBody().front());

      auto args = derive(c);
      if (!args)
        return Maybe<SimCost>(emitSilenceableFailure(
            trial.computeBlock->getLoc(),
            "the reduction template cannot express this configuration"));
      trial.computeBlock->walk([&](cinm::ReduceOp op) {
        generateTailReduction(op, rewriter, args->dpuRows, args->dpuCols,
                              args->mramRow, args->mramCol, args->wramRow,
                              args->wramCol, args->taskletRows,
                              args->taskletCols);
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

      TRY(runPipeline(cleanupPm.get(), trial.computeBlock->getLoc(),
                      trial.module.get()));

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

  UpmemInferenceOptions buildOptions() {
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
    o.dumpDir = dumpDir;
    upmemOpts.annotateOpCosts = annotateOpCosts;
    upmemOpts.useMRAMTiling = useMRAMTiling;
    upmemOpts.lowering = lowering;
    upmemOpts.fixedDpus = fixedDpus;
    upmemOpts.fixedTasklets = fixedTasklets;
    upmemOpts.simulator = simulator;
    upmemOpts.evalTimeoutMs = std::chrono::milliseconds(evalTimeoutMs);
    upmemOpts.debugPrintsInPipeline = debugPipeline;
    if (!evalSolution.empty()) {
      llvm::StringMap<cinm::ParmValue> named;
      for (StringRef entry : evalSolution) {
        auto [name, value] = entry.split('=');
        cinm::ParmValue parsed;
        if (name.empty() || value.getAsInteger(10, parsed)) {
          getOperation()->emitError()
              << "eval-solution entry '" << entry
              << "' is not a `name=value` pair; configurations are named, not "
                 "positional";
          signalPassFailure();
          break;
        }
        named[name.trim()] = parsed;
      }
      o.evalSingleSolution = std::move(named);
    }
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
