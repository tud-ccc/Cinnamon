#include <cinm-mlir/Conversion/CinmPasses.h>
#include <cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h>
#include <cinm-mlir/Conversion/CommonPatterns.h>
#include <cinm-mlir/Conversion/LinalgToCnm/LinalgToCnm.h>
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h>
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/FusionEdges.h>
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h>
#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmBase.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h>
#include <cinm-mlir/Dialect/Cinm/Transforms/Passes.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOccupancy.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>

#include "SimulatorBase.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

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
#include <mlir/IR/IRMapping.h>
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
#include <upmem_cost_model/ScatterGatherCm.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMINFERACCELERATORPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {
using mlir::cinm::IntVar;
using mlir::cinm::PermVar;
using mlir::cinm::SpaceBuilder;
using mlir::cinm::utils::Maybe;

/// What `UpmemPlatformAttr::getName()` returns: the name a scope's
/// `cinm.available_platforms` list uses to offer this backend.
static constexpr llvm::StringLiteral kUpmemPlatformName = "upmem";

/// UPMEM-specific inference options. Wraps the generic InferenceOptions and
/// provides a place to add UPMEM-specific knobs in the future.
struct UpmemInferenceOptions {
  cinm::InferenceOptions inference;
  bool annotateOpCosts = false;
  bool useMRAMTiling = true;
  // Scatter specialisation (constant scatter -> broadcast, on-device init
  // of uniform buffers) fires at the cnm level and again at the upmem
  // level; this switch disables both sites, or the A1 ablation
  // under-reports the capability.
  bool scatterSpecialisation = true;
  bool fusionEdges = true;
  bool debugPrintsInPipeline = false;
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
  pm.addPass(affine::createLoopFusionPass());
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
          op->getAttrOfType<StringAttr>(cinm::CinmDialect::DEBUG_TAG_NAME))
    return shortOpName(origin.getValue());
  return shortOpName(op->getName().getStringRef());
}

static SmallVector<SmallVector<unsigned>> linalgOperandDims(linalg::LinalgOp);
static FailureOr<SmallVector<int64_t>> linalgLoopExtents(linalg::LinalgOp);

struct UpmemInferencePlugin : cinm::InferencePlugin {
  upmem::UpmemPlatformAttr platform;
  const UpmemInferenceOptions &opts;
  std::unique_ptr<UpmemSimulator> simulator;

  // Handles for dpus and tasklets, assigned during initializeSpace.
  IntVar dpusVar_, taskletsVar_;

  std::unique_ptr<PassManager> frontPipeline;
  std::unique_ptr<PassManager> backPipeline;

  /// Names of the search parameters an op's lowering consumes, stamped on the
  /// op itself when the space is built (see initializeSpace). The reference is
  /// what every trial is cloned from, so a trial inherits them and
  /// stampSearchParams only has to resolve each name against the trial's
  /// configuration -- no correspondence between reference ops and trial ops
  /// has to be maintained on the side.
  /// @{
  /// One parameter name per iteration dimension, resolving to
  /// `cnm.tile_sizes`.
  static constexpr llvm::StringLiteral kOuterTileParamsAttr =
      "upmem.outer_tile_params";
  /// One parameter name per iteration dimension, resolving to
  /// `upmem.leaf_tile_sizes`.
  static constexpr llvm::StringLiteral kLeafTileParamsAttr =
      "upmem.leaf_tile_params";
  /// Single parameter name, resolving to the workgroup dimension order index.
  static constexpr llvm::StringLiteral kOrderParamAttr = "upmem.order_param";
  /// @}

  UpmemInferencePlugin(upmem::UpmemPlatformAttr platform,
                       const UpmemInferenceOptions &opts,
                       std::unique_ptr<UpmemSimulator> sim)
      : platform(platform), opts(opts), simulator(std::move(sim)) {}

  bool supportsMultithreading() const override {
    return simulator && simulator->supportsMultithreading();
  }

  /// The resource the graph level divides between compute blocks is the DPU
  /// count: cost profiles are indexed by it.
  StringRef sharedResourceParam() const override { return "dpus"; }

  int64_t sharedResourceMax() const override { return platform.getMaxDpus(); }

  /// Iteration-space sizes (product of loop extents) of every op the
  /// pipeline would distribute in `block`, read off a throwaway linalg
  /// conversion -- the same one the search space itself is derived from, so
  /// the menu and the space agree about what gets distributed.
  SmallVector<int64_t>
  distributedIterationSizes(cinm::ComputeBlockOp block) const {
    MLIRContext *ctx = block->getContext();
    OpBuilder b(ctx);
    Location loc = block.getLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    auto func = func::FuncOp::create(
        loc, "menu_probe",
        FunctionType::get(ctx, SmallVector<Type>(block->getOperandTypes()),
                          SmallVector<Type>(block->getResultTypes())));
    module->push_back(func);
    Block *entry = func.addEntryBlock();
    b.setInsertionPointToStart(entry);
    IRMapping mapping;
    for (auto [operand, arg] :
         llvm::zip(block->getOperands(), entry->getArguments()))
      mapping.map(operand, arg);
    auto *clone = b.clone(*block, mapping);
    func::ReturnOp::create(b, loc, clone->getResults());

    auto pm = buildConvertPipeline(ctx, /*debug=*/false);
    if (failed(pm->run(*module)))
      return {};

    SmallVector<int64_t> sizes;
    module->walk([&](Operation *op) {
      if (!isDistributionCandidate(op))
        return;
      auto extents = linalgLoopExtents(llvm::cast<linalg::LinalgOp>(op));
      if (failed(extents))
        return;
      int64_t product = 1;
      for (int64_t extent : *extents)
        product *= extent;
      sizes.push_back(product);
    });
    return sizes;
  }

  /// The DPU counts worth profiling `block` at. The workgroup must be filled
  /// exactly -- the tiles of every distributed op multiply out to
  /// dpus * tasklets -- and tasklets = 1 is always admissible, so the
  /// divisibility-feasible DPU counts are exactly the divisors of each op's
  /// iteration-space size: divisors of their gcd for the block. Among those
  /// the menu prefers multiples of the allocation granularity (rank-sized
  /// sets keep host<->DPU transfers rank-parallel), falling back to plain
  /// divisors when the problem size admits no such multiple, and thins to a
  /// bounded count. Divisibility is necessary, not sufficient: a menu value
  /// the pinned search still finds infeasible (capacity) becomes a hole in
  /// the profile.
  SmallVector<int64_t>
  sharedResourceMenu(cinm::ComputeBlockOp block) const override {
    const int64_t maxDpus = sharedResourceMax();
    const int64_t granularity =
        std::max<int64_t>(1, opts.inference.allocationGranularity);

    SmallVector<int64_t> sizes = distributedIterationSizes(block);
    if (sizes.empty())
      return {};
    int64_t g = 0;
    for (int64_t size : sizes)
      g = std::gcd(g, size);

    SmallVector<int64_t> divisors;
    for (int64_t d = 1; d <= std::min(g, maxDpus); ++d)
      if (g % d == 0)
        divisors.push_back(d);

    SmallVector<int64_t> menu;
    for (int64_t d : divisors)
      if (d % granularity == 0)
        menu.push_back(d);
    if (menu.empty())
      for (int64_t d : divisors)
        if (d >= granularity)
          menu.push_back(d);
    if (menu.empty())
      menu = divisors;

    // Sorted divisors are distributed roughly geometrically, so index-spaced
    // thinning approximates log spacing and keeps both endpoints.
    constexpr size_t kMaxMenu = 16;
    if (menu.size() > kMaxMenu) {
      SmallVector<int64_t> thinned;
      for (size_t i = 0; i < kMaxMenu; ++i) {
        size_t idx = (i * (menu.size() - 1)) / (kMaxMenu - 1);
        if (thinned.empty() || thinned.back() != menu[idx])
          thinned.push_back(menu[idx]);
      }
      menu = std::move(thinned);
    }
    return menu;
  }

  /// Footprint at a configuration, per memory level and split by operand
  /// staticness, mirroring exactly the capacity bounds the search space
  /// posts (see handleLinalgOp): per level, `tasklets × Σ_operands
  /// Π_dims tile[d]` with that level's tile sizes, no sharing assumed. The
  /// tile sizes are read back through the parameter names stamped as
  /// kOuterTileParamsAttr (MRAM) and kLeafTileParamsAttr (WRAM); staticness
  /// resolves through the trial's function-argument attributes
  /// (isStaticValue). Only MRAM holds anything between inferences -- WRAM
  /// tiles are re-staged by DMA on every use -- so WRAM footprints are all
  /// dynamic and never bind the co-residency packing. An operand that is not
  /// directly a block argument (a `linalg.fill` accumulator, a fused
  /// intermediate) is charged as dynamic, which errs toward under-pinning.
  cinm::ResidencyInfo measureResidency(cinm::TrialInfo &trial) override {
    cinm::ResidencyInfo out;
    auto valueOf = [&](StringRef name) -> int64_t {
      for (size_t d = 0; d < trial.space->numDims(); ++d)
        if (trial.space->dimName(d) == name)
          return trial.config[d];
      return -1;
    };
    const int64_t tasklets = valueOf("tasklets");
    const int64_t dpus = valueOf("dpus");

    cinm::LevelResidency mram{
        platform.getMramLevel().getName().getValue().str(), 0, 0};
    cinm::LevelResidency wram{
        platform.getWramLevel().getName().getValue().str(), 0, 0};

    cinm::ComputeBlockOp block = trial.computeBlock;
    block.getBody().walk([&](linalg::LinalgOp op) {
      auto mramParams = op->getAttrOfType<ArrayAttr>(kOuterTileParamsAttr);
      auto wramParams = op->getAttrOfType<ArrayAttr>(kLeafTileParamsAttr);
      if (!mramParams || !wramParams)
        return;
      auto tileSizes = [&](ArrayAttr params) {
        SmallVector<int64_t> sizes;
        for (Attribute name : params)
          sizes.push_back(valueOf(llvm::cast<StringAttr>(name).getValue()));
        return sizes;
      };
      SmallVector<int64_t> mramTile = tileSizes(mramParams);
      SmallVector<int64_t> wramTile = tileSizes(wramParams);

      auto operandDims = linalgOperandDims(op);
      for (auto [opnd, dims] : llvm::zip(op->getOpOperands(), operandDims)) {
        auto shaped = llvm::cast<ShapedType>(opnd.get().getType());
        const int64_t eltBytes =
            std::max<int64_t>(1, shaped.getElementTypeBitWidth() / 8);
        int64_t mramElts = 1, wramElts = 1;
        for (unsigned dim : dims) {
          mramElts *= mramTile[dim];
          wramElts *= wramTile[dim];
        }
        // WRAM is scratch: every tile there is re-staged per use, so it is
        // dynamic whatever the operand's staticness.
        wram.dynBytes += tasklets * wramElts * eltBytes;

        const int64_t perDpuBytes = tasklets * mramElts * eltBytes;
        auto arg = llvm::dyn_cast<BlockArgument>(opnd.get());
        const bool isStatic = arg && arg.getOwner()->getParentOp() == block &&
                              cinm::isStaticValue(arg);
        if (isStatic) {
          mram.staticBytes += perDpuBytes;
          // What a timeshared placement would pay per inference to restore
          // these weights: the whole tensor through the scatter model.
          out.weightScatterMs += upmem_cm::scatterBlockCostMs(
              dpus, (shaped.getNumElements()) * eltBytes);
        } else {
          mram.dynBytes += perDpuBytes;
        }
      }
    });
    out.levels.push_back(std::move(mram));
    out.levels.push_back(std::move(wram));
    return out;
  }

  /// One upmem.alloc_dpus for the whole group, shaped by the group's winning
  /// configuration. The load is NOT emitted here: which binary the set holds
  /// is each member's decision at lowering time (see CnmToUPMEM), and the
  /// program symbol does not even exist yet when this runs.
  Value materializeWorkgroupAlloc(
      OpBuilder &builder, Location loc,
      const llvm::StringMap<cinm::ParmValue> &config) override {
    int64_t dpus = config.lookup("dpus");
    int64_t tasklets = config.lookup("tasklets");
    if (dpus <= 0 || tasklets <= 0)
      return Value();
    auto ty =
        upmem::DeviceHierarchyType::get(builder.getContext(), dpus, tasklets);
    return upmem::AllocDPUsOp::create(builder, loc, ty).getResult();
  }

  void materializeWorkgroupFree(OpBuilder &builder, Location loc,
                                Value workgroup) override {
    upmem::FreeDPUsOp::create(builder, loc, workgroup);
  }

  /// cinm -> linalg. Run once, on the reference the trials are cloned from,
  /// because its result does not depend on the configuration: the space is
  /// stated in terms of an *iteration space*, and only linalg carries one, so
  /// the search has to see the converted form anyway. This is also where
  /// fusion happens (design §G8), which is what makes the iteration spaces the
  /// space is built from the ones the pipeline will actually distribute.
  static std::unique_ptr<PassManager> buildConvertPipeline(MLIRContext *ctx,
                                                           bool debug) {
    auto pm = std::make_unique<PassManager>(ctx);
    pm->addPass(cinm::createConvertCinmOpsToLinalgPass());
    pm->addPass(mlir::createConvertTensorToLinalgPass());
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-cinm-to-linalg"}));
    // The layout decision below restates the iteration space, which a named
    // op cannot express -- its maps and iterator kinds are implied by its
    // name. Generalizing here keeps that concern out of the distribution
    // pass itself.
    pm->addPass(createLinalgGeneralizeNamedOpsPass());
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

  /// linalg -> cnm -> bufferized, everything up to and including
  /// bufferization. Split from the back pipeline only because the latter has
  /// to see the launch bodies as linalg on memrefs.
  static std::unique_ptr<PassManager>
  buildFrontPipeline(MLIRContext *ctx, const UpmemInferenceOptions &opts) {
    bool debug = opts.debugPrintsInPipeline;
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
      // Lay each leaf's buffers out for the tile --upmem-tile-mram-buffers
      // will stage below, so that one staged chunk is a contiguous run and
      // the transfer is a single DMA.
      cnmOpts.leafTileAttr = UPMEMDialect::LEAF_TILE_SIZES_NAME.str();
      pm->addPass(cnm::createConvertLinalgToCnmPass(cnmOpts));
    }
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-linalg-to-cnm"}));
    pm->addPass(cnm::createCnmHoistWorkgroupsPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Two ops the configuration distributes the same way send the value
    // between them through the host and back to the leaf it came from. Cancel
    // that round trip and run the two as one launch, which is also what makes
    // the leaf level able to fuse them later (docs/LaunchFusionDesign.md).
    // Opportunistic: a configuration whose schedules do not agree -- one that
    // splits the producer's reduction across the workgroup, in particular --
    // presents no round trip and is left alone. Before bufferization, so that
    // --cnm-scatter-optimizations and the bufferizer see the merged form.
    pm->addPass(cnm::createCnmFuseLaunchesPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());

    // Step 3: bufferize
    pm->addPass(bufferization::createEmptyTensorEliminationPass());
    if (opts.scatterSpecialisation)
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
  static std::unique_ptr<PassManager>
  buildBackPipeline(MLIRContext *ctx, const UpmemInferenceOptions &opts) {
    bool debug = opts.debugPrintsInPipeline;
    auto pm = std::make_unique<PassManager>(ctx);

    // Staging has to see linalg on memrefs, so it runs after bufferization and
    // before linalg is lowered to loops.
    pm->addPass(createUpmemTileMRAMBuffersPass());
    pm->addPass(createCanonicalizerPass());
    pm->addPass(createCSEPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-tile-mram-buffers"}));
    pm->addPass(createLinalgGeneralizeNamedOpsPass());
    pm->addPass(createLinalgElementwiseOpFusionPass());
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
    if (debug)
      pm->addPass(createPrintIRPass({.label = "before-cnm-sg-contiguous"}));
    pm->addPass(cnm::createCnmEnsureScatterGatherContiguousPass(
        {.packFragmented = true, .staticOnly = true}));
    pm->addPass(createCanonicalizerPass());
    if (debug)
      pm->addPass(createPrintIRPass({.label = "after-cnm-sg-contiguous"}));
    pm->addPass(cnm::createConvertCnmToUPMEMPass({}));
    // Right after the conversion, so the rest of the back pipeline sees the
    // narrowest transfer form each map allows -- in particular the occupancy
    // check and the cost model, which read the ops' shapes. The broadcast
    // narrowing is the upmem-level half of scatter specialisation; the
    // block-collapsing rewrites are pure transfer-form narrowing and stay on
    // either way.
    {
      UpmemSpecializeTransfersPassOptions specOpts;
      specOpts.useBcXferCodegen = opts.scatterSpecialisation;
      pm->addPass(createUpmemSpecializeTransfersPass(specOpts));
    }
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
      // The DPU compiler unrolls nothing by itself, so a short innermost loop
      // pays a counter increment, a branch and an address computation per
      // operand on every iteration -- about half the instructions of a
      // multiply-accumulate body. Unrolling here rather than asking the DPU
      // compiler for it keeps the cost model reading the code that runs.
      //
      // `unrollUpToFactor` is what makes the factor a *bound*: it unrolls by
      // min(trip count, factor), so a loop shorter than 64 comes out fully
      // unrolled instead of untouched -- plain `unroll-factor=64` fails
      // outright on anything shorter (loopUnrollByFactor bails when the trip
      // count is below the factor). Only innermost loops are considered, and
      // only once, so an outer loop is never unrolled around a body this has
      dpuPm.addPass(affine::createLoopUnrollPass(/*unrollFactor=*/129,
                                                 /*unrollUpToFactor=*/true));
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
    return c;
  }

  void warmUp(mlir::MLIRContext *ctx) override {
    if (!frontPipeline) {
      frontPipeline = buildFrontPipeline(ctx, opts);
      backPipeline = buildBackPipeline(ctx, opts);
    }
    simulator->warmUp();
  }

  void printStats() const override { simulator->printStats(); }

  /// The search space for one op, stated in the parameters the passes
  /// actually consume (design §G2, §H5): one block size per iteration
  /// dimension for the workgroup distribution, and one for the leaf level.
  /// Also stamps the parameter names on `op` itself, see
  /// kOuterTileParamsAttr.
  ///
  /// What it declared is reported back, because the fusion edges between two
  /// ops are stated over both ops' parameters and so can only be declared once
  /// every op has been through here.
  std::optional<cinm::DistributedOpInfo>
  handleLinalgOp(linalg::LinalgOp op, StringRef namePrefix, SpaceBuilder &b);

  void initializeSpace(cinm::ComputeBlockOp refClone,
                       cinm::SpaceBuilder &b) override {
    const int64_t maxDpus = platform.getMaxDpus();
    const int64_t maxTasklets = platform.getMaxNumTasklets();
    dpusVar_ = opts.fixedDpus > 0
                   ? b.intRange("dpus", opts.fixedDpus, opts.fixedDpus)
                   : b.intRange("dpus", 1, maxDpus);
    b.describe("dpus", "number of DPUs the workgroup spans (ATiM: product of "
                       "blockIdx extents)");
    taskletsVar_ =
        opts.fixedTasklets > 0
            ? b.intRange("tasklets", opts.fixedTasklets, opts.fixedTasklets)
            : b.intRange("tasklets", 1, maxTasklets);
    b.describe("tasklets", "tasklets per DPU (ATiM: threadIdx extent)");

    // handleLinalgOp declares a tiling factor per iteration dimension per
    // level the platform reports, but only two of them have a consumer: the
    // outermost feeds `cnm.tile_sizes` and the innermost
    // `upmem.leaf_tile_sizes`. A level in between would get search parameters
    // no pass ever reads -- dimensions that multiply the space and change
    // nothing about the program -- so refuse the platform rather than
    // silently offer them.
    if (platform.getLevels().size() != 2) {
      emitError(refClone->getLoc(), "this platform declares ")
          << platform.getLevels().size()
          << " memory level(s); the UPMEM inference plugin can only supply "
             "tiling factors for two of them (the workgroup distribution and "
             "the leaf level)";
      return;
    }

    // The space is derived from the *linalg* form of the block. Block sizes
    // are indexed by iteration dimension and only linalg states an iteration
    // space; deriving them from cinm ops instead would mean maintaining a
    // second, hand-written notion of each op's iteration space. That notion
    // already disagrees: `getTilableDimSizes` reports one flattened dimension
    // for an elementwise op where its linalg form has one per rank.
    //
    // The reference itself is converted, not a throwaway copy of it: the
    // conversion does not depend on the configuration, so doing it once here
    // both saves every trial from repeating it and lets the parameters be
    // stamped straight onto the ops the space was read from. Trials are clones
    // of what this leaves behind, so they inherit the annotations and start
    // where the search space starts.
    MLIRContext *ctx = refClone->getContext();
    Location loc = refClone->getLoc();
    ModuleOp refModule = refClone->getParentOfType<ModuleOp>();
    {
      auto pm = buildConvertPipeline(ctx, opts.debugPrintsInPipeline);
      if (failed(pm->run(refModule))) {
        emitError(loc, "could not convert the compute block to linalg, "
                       "so no search space can be derived from it");
        return;
      }
    }

    // Not `refClone`: the pipeline above may have replaced the compute block
    // op (canonicalization rebuilds it to drop an unused block argument), so
    // the handle the framework passed in can be dangling by now.
    cinm::ComputeBlockOp block;
    refModule.walk([&](cinm::ComputeBlockOp op) { block = op; });
    if (!block) {
      emitError(loc, "the converted reference has no compute block");
      return;
    }

    // Name each op's parameters after the cinm op it came from, e.g.
    // `gemv.M0`. Count the kinds first so that a block with two gemvs gets
    // `gemv0`/`gemv1` while the common single-op case stays unadorned.
    llvm::StringMap<unsigned> kindCount;
    block.getBody().walk([&](Operation *op) {
      if (isDistributionCandidate(op))
        ++kindCount[searchNameFor(op)];
    });

    llvm::StringMap<unsigned> kindSeen;
    SmallVector<cinm::DistributedOpInfo, 2> distributed;
    block.getBody().walk([&](Operation *op) {
      if (!isDistributionCandidate(op))
        return;
      std::string kind = searchNameFor(op);
      std::string prefix =
          kindCount[kind] > 1 ? kind + std::to_string(kindSeen[kind]++) : kind;
      if (auto info =
              handleLinalgOp(llvm::cast<linalg::LinalgOp>(op), prefix, b))
        distributed.push_back(std::move(*info));
    });

    // Whether a consumer can pick its operand up from the leaf the producer
    // left it on is a joint property of the two ops' tilings, so it is stated
    // once both have declared theirs. See declareFusionEdges.
    if (opts.fusionEdges)
      cinm::declareFusionEdges(distributed, b);
  }

  /// Whether to print why an individual trial's pipeline failed.
  ///
  /// During a search a rejected trial is ordinary, not an event: exhaustive
  /// and sampling runs reject thousands of them, and evaluate them on many
  /// threads at once. llvm::dbgs() offers no atomicity beyond a single write,
  /// so these multi-part messages (a prefix, the diagnostic, then a whole
  /// module dump) interleave mid-line across threads and shred the log --
  /// including the parts of it that are worth reading.
  ///
  /// They are wanted in exactly one case: a configuration pinned with
  /// eval-solution, where "why was this rejected" is the question being asked.
  /// Everywhere else the reason still reaches the caller through the
  /// silenceable failure below, it is just not printed per trial.
  bool logTrialDiagnostics() const {
    return opts.inference.evalSingleSolution.has_value();
  }

  DiagnosedSilenceableFailure runPipeline(PassManager *pipeline, Location loc,
                                          ModuleOp module) const {
    // Diagnostics are swallowed because a rejected trial is ordinary during a
    // search -- but the first error is kept, so that the reason surfaces in
    // the failure a pinned `eval-solution` reports. Without it every rejection
    // reads "Pipeline failed", including the occupancy check's.
    std::string reason;
    const bool verbose = logTrialDiagnostics();
    ScopedDiagnosticHandler scopedHandler(
        pipeline->getContext(), [&reason, verbose](Diagnostic &diag) {
          if (reason.empty() && diag.getSeverity() == DiagnosticSeverity::Error)
            reason = diag.str();
          if (verbose)
            LLVM_DEBUG(llvm::dbgs()
                           << "[cinm-inference]   pipeline failed:\n      ";
                       diag.print(llvm::dbgs()); llvm::dbgs() << "\n";);
          return success();
        });
    if (mlir::failed(pipeline->run(module))) {
      if (verbose)
        LLVM_DEBUG(module->print(llvm::dbgs());
                   llvm::dbgs() << "\n========\n";);
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
    trial.computeBlock.setPlatformAttr({});
    trial.computeBlock.setAcceleratorAttr(
        upmem::UpmemAcceleratorAttr::get(platform, dpus, tasklets));

    TRY(runLowering(trial));
    SimCost total = TRY_GET(simulator->simulate(trial.computeBlock.getBody()));
    annotateCost(ctx, trial, total.total());
    return total;
  }

  /// Lower `trial` through the real pass pipeline. The trial starts in the
  /// linalg form the search space was built from (see initializeSpace), so
  /// only the configuration-dependent stages are left.
  DiagnosedSilenceableFailure runLowering(cinm::TrialInfo &trial) {
    mlir::Location loc = trial.computeBlock->getLoc();
    MLIRContext *ctx = trial.computeBlock->getContext();
    if (!frontPipeline) {
      frontPipeline = buildFrontPipeline(ctx, opts);
      backPipeline = buildBackPipeline(ctx, opts);
    }

    // 1. Resolve this configuration's tiling factors and iteration orders onto
    // the ops the space named when it was built.
    stampSearchParams(trial);
    // 2. Convert linalg to CNM (using the workgroup tiling factors
    // and iteration order parameters), bufferize, perform affine
    // optimizations and canonicalizations.
    TRY(runPipeline(frontPipeline.get(), loc, trial.module.get()));
    // 3. Perform tiling of the MRAM kernel into a WRAM program. This uses
    // other tiling factors. Convert the CNM IR to upmem, perform more
    // canonicalizations and transfer op specializations. Finally lower the
    // kernel from affine to SCF.
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
  /// Resolve the parameter names stamped on the reference (see
  /// kOuterTileParamsAttr) against this trial's configuration, and replace
  /// them with the values the passes read.
  ///
  /// Both levels are resolved here, before anything runs, because
  /// --convert-linalg-to-cnm carries discardable attributes into the launch
  /// body it creates. Nothing has to find the op again half way down the
  /// pipeline.
  void stampSearchParams(cinm::TrialInfo &trial) const {
    cinm::ConfWrapper conf = trial.conf();
    // A name the space does not have reads as 0, which would stamp a tile size
    // of 0 and mis-tile silently. It cannot happen -- every name stamped below
    // was declared on the same SpaceBuilder -- so assert rather than handle it.
    auto value = [&](StringRef name) {
      assert(trial.space->findParam(name) >= 0 &&
             "op names a search parameter the space does not declare");
      return conf[name];
    };
    auto resolve = [&](ArrayAttr names) {
      SmallVector<int64_t> sizes;
      for (Attribute name : names)
        sizes.push_back(value(llvm::cast<StringAttr>(name).getValue()));
      return sizes;
    };

    trial.computeBlock.getBody().walk([&](Operation *op) {
      MLIRContext *ctx = op->getContext();
      OpBuilder b(ctx);
      if (auto names = op->getAttrOfType<ArrayAttr>(kOuterTileParamsAttr))
        op->setAttr(cnm::CnmDialect::TILE_SIZES_NAME,
                    b.getDenseI64ArrayAttr(resolve(names)));
      if (auto names = op->getAttrOfType<ArrayAttr>(kLeafTileParamsAttr))
        op->setAttr(UPMEMDialect::LEAF_TILE_SIZES_NAME,
                    b.getDenseI64ArrayAttr(resolve(names)));
      // The order, as the permutation it is. Only the inversion happens here:
      // the space says where each dimension went, and the attribute lists the
      // dimensions in axis order.
      //
      // Over the dimensions the op has *now*, which is not the set
      // --convert-linalg-to-cnm will tile: splitting a distributed reduction
      // prepends a dimension. That pass rewrites the attribute as it splits,
      // so nothing here has to predict it. The rank form the attribute also
      // accepts exists for exactly that reason and is why this used to have to
      // reconstruct the post-split dimension order -- the same reconstruction,
      // written twice, agreeing by inspection.
      if (auto name = op->getAttrOfType<StringAttr>(kOrderParamAttr)) {
        cinm::Permutation order =
            trial.space->getAs<cinm::Permutation>(conf.conf, name.getValue());
        SmallVector<int64_t> byAxis(order.size());
        for (size_t dim = 0; dim < order.size(); ++dim)
          byAxis[order[dim]] = static_cast<int64_t>(dim);
        op->setAttr(cnm::CnmDialect::WORKGROUP_DIM_ORDER_NAME,
                    b.getDenseI64ArrayAttr(byAxis));
      }
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

/// Names for the operands of `op`, in the order linalgOperandDims() lists
/// them, so a constraint over one operand's tile says which operand it is.
static SmallVector<std::string> linalgOperandNames(linalg::LinalgOp op) {
  SmallVector<std::string> names;
  int64_t inputs = op.getNumDpsInputs();
  for (int64_t idx = 0, end = op->getNumOperands(); idx < end; ++idx)
    names.push_back(idx < inputs ? "input " + std::to_string(idx)
                                 : "output " + std::to_string(idx - inputs));
  return names;
}

std::optional<cinm::DistributedOpInfo>
UpmemInferencePlugin::handleLinalgOp(linalg::LinalgOp op, StringRef namePrefix,
                                     SpaceBuilder &b) {
  FailureOr<SmallVector<int64_t>> extents = linalgLoopExtents(op);
  if (failed(extents))
    return std::nullopt; // Not distributable; contributes no parameters.

  auto dpus = dpusVar_;
  auto tasklets = taskletsVar_;
  Type eltTy = cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();

  // Determine the names of the search params. Each dimension gets one
  // parameter per memory level (a tiling factor).
  auto originAttr =
      op->getAttrOfType<StringAttr>(cinm::CinmDialect::DEBUG_TAG_NAME);
  StringRef origin = originAttr ? originAttr.getValue() : StringRef();
  SmallVector<std::string> dimNames =
      iterationDimNames(origin, extents->size());

  /// One tiling factor per level and dim.
  ArrayRef<cinm::CinmLevelDefAttr> levels = platform.getLevels();
  SmallVector<SmallVector<IntVar>> perLevel(levels.size());
  for (auto [dim, extent] : llvm::enumerate(*extents)) {
    std::string base = (namePrefix + "." + dimNames[dim]).str();
    for (auto [levelIdx, level] : llvm::enumerate(levels)) {
      std::string name = base + "." + level.getName().getValue().str();
      perLevel[levelIdx].push_back(
          levelIdx == 0 ? b.divisorsOf(name, extent)
                        : b.divisorsOf(name, perLevel[levelIdx - 1].back()));
      b.describe(name,
                 (level.getName().getValue() + " tile size of iteration dim " +
                  dimNames[dim] + " (extent " + std::to_string(extent) +
                  ") of " + (origin.empty() ? "this op" : origin) +
                  (levelIdx == 0
                       ? "; per LEAF -- across all dims, prod(extent/tile) == "
                         "dpus*tasklets, so one such tile per (dpu, tasklet) "
                         "worker, not per DPU; must divide the extent"
                       : "; must divide the previous level's tile"))
                     .str());
    }
  }
  // todo is this generic enough for CNM? I think so
  SmallVector<IntVar> blocks = perLevel.front();
  SmallVector<IntVar> leaves = perLevel.back();

  // The tile counts must fill the workgroup exactly. This is the
  // one structural constraint; everything else about the distribution follows
  // from the block sizes and the op's own indexing maps.
  SmallVector<int64_t> extentsCopy(*extents);
  SmallVector<cinm::IntExpr> tilesPerDim;
  for (auto [extent, block] : llvm::zip_equal(extentsCopy, blocks))
    tilesPerDim.push_back(extent / block);
  b.require(cinm::prod(tilesPerDim) == dpus * tasklets,
            "prod(extent / block) == dpus * tasklets");

  // Capacity. A level's size is what one DPU has, and a DPU runs `tasklets`
  // tiles, so the footprint charged here is one private copy of every operand
  // tile per tasklet: no sharing. A configuration that passes therefore has
  // room whatever lowering decides to share.
  //
  // Tasklets can in fact share an operand whose dimensions no tasklet index
  // reaches -- the vector of a gemv, say -- so this rejects configurations
  // that would have fitted. That is the direction the bound is meant to err
  // in; the tight test is still done on the lowered program.
  auto operandDims = linalgOperandDims(op);
  // The tile of each operand at one level, as a count of elements: the
  // product of that level's tiling factors over the dimensions the operand is
  // indexed by. This is both the unit the capacity bound sums and the unit a
  // transfer moves, which is why the DMA constraint below shares it.
  auto operandTiles = [operandDims](ArrayRef<IntVar> sizes) {
    SmallVector<cinm::IntExpr> operands;
    for (const auto &dims : operandDims) {
      SmallVector<cinm::IntExpr> factors;
      for (unsigned dim : dims)
        factors.push_back(sizes[dim]);
      operands.push_back(cinm::prod(std::move(factors)));
    }
    return operands;
  };
  auto footprint = [&operandTiles, tasklets](ArrayRef<IntVar> sizes) {
    return tasklets * (kStackReserveBytes + cinm::sum(operandTiles(sizes)));
  };

  // One bound per level, against the capacity the platform declares for it.
  for (auto [levelIdx, level] : llvm::enumerate(levels))
    b.require(footprint(perLevel[levelIdx]) <= level.getSizeInElements(eltTy),
              ("tasklets * sum of operand tiles <= " +
               level.getName().getValue() + " (assuming no sharing)")
                  .str());
  if (!opts.useMRAMTiling) {
    // Note: this is only required for benchmarks that compare
    // against CINM1 codegen. To be removed.
    b.require(footprint(blocks) <= levels.back().getSizeInElements(eltTy),
              "MRAM tile should be equal to WRAM tile (no tiling in MRAM)");
  }

  // DMA granularity. Every operand tile is staged by one transfer -- host to
  // MRAM at the outer level, MRAM to WRAM at the leaf -- and the engine
  // addresses whole granules only. A tile that is a fraction of one cannot be
  // moved correctly: the per-tasklet slices of a shared buffer sit at
  // `tasklet * tileBytes`, so a fractional tile puts every other slice at a
  // misaligned address, and the length has to be rounded up into the slice
  // next door. The C emitter refuses such a program, so without this a
  // configuration of this shape costs a trial and yields nothing.
  //
  // Stated per level, since the granule is the level's own declared property.
  // Both UPMEM levels declare 8 bytes, which makes the outer constraint
  // implied -- a dimension's outer tile is a multiple of its leaf tile -- but
  // it is the host transfer that the outer one describes, so it is posted on
  // its own terms rather than left to follow.
  const int64_t eltBits = eltTy.getIntOrFloatBitWidth();
  SmallVector<std::string> operandNames = linalgOperandNames(op);
  for (auto [levelIdx, level] : llvm::enumerate(levels)) {
    const int64_t granuleBits = level.getAlignment() * 8;
    // The smallest tile that is a whole number of granules. Counted in bits
    // so that an element narrower than a byte stays exact.
    const int64_t elemsPerGranule =
        granuleBits / std::gcd(granuleBits, eltBits);
    if (elemsPerGranule <= 1)
      continue;
    for (auto [name, tile] :
         llvm::zip_equal(operandNames, operandTiles(perLevel[levelIdx])))
      b.require(cinm::divides(cinm::ParmValue(elemsPerGranule), tile),
                (name + "'s " + level.getName().getValue() +
                 " tile must be a whole number of " +
                 std::to_string(level.getAlignment()) +
                 "-byte DMA granules, i.e. a multiple of " +
                 std::to_string(elemsPerGranule) + " elements")
                    .str());
  }

  // Which tile dimension varies fastest across the leaves (design §G3). The
  // one parameter here that is not a size: it decides what the leaves sharing
  // a hardware node share rather than replicate, which the block sizes cannot
  // state. Its type is what stops the DSL doing arithmetic on it.
  //
  // The items ordered are the dimensions this configuration actually spreads
  // over the workgroup, which is not known at declaration: a dimension cut
  // into one tile takes no workgroup axis. So the activity is handed over as
  // an expression per dimension and SpaceBuilder posts what follows from it --
  // an ordering of the active items is a notion the framework has, and what it
  // costs in the encoding is not this file's business.
  std::optional<PermVar> order;
  if (extents->size() >= 2) {
    SmallVector<cinm::BoolExpr> distributed;
    for (auto [extent, block] : llvm::zip_equal(extentsCopy, blocks))
      distributed.push_back(extent / block > 1);
    std::string orderName = (namePrefix + ".order").str();
    order = b.permutation(orderName, distributed);
    b.describe(orderName,
               "which iteration dim occupies which workgroup axis "
               "(cnm.workgroup_dim_order). An item is active only when its "
               "dim is actually distributed (extent / " +
                   levels.front().getName().getValue().str() +
                   " tile > 1); see the encoding note");
    b.labelItems(orderName, dimNames);
  }

  // Record which parameters this op's lowering consumes, on the op itself.
  // Every trial is cloned from this reference, so the association survives
  // into each trial without a side table keyed on anything a rewrite could
  // invalidate; stampSearchParams only resolves the names.
  OpBuilder builder(op.getContext());
  auto paramNames = [&](ArrayRef<IntVar> vars) {
    SmallVector<Attribute> names;
    for (const IntVar &var : vars)
      names.push_back(builder.getStringAttr(var.name()));
    return builder.getArrayAttr(names);
  };
  op->setAttr(kOuterTileParamsAttr, paramNames(blocks));
  op->setAttr(kLeafTileParamsAttr, paramNames(leaves));
  if (order)
    op->setAttr(kOrderParamAttr, builder.getStringAttr(order->name()));

  return cinm::DistributedOpInfo{op, namePrefix.str(), std::move(extentsCopy),
                                 std::move(perLevel), order};
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
    o.boBatchSize = boBatchSize;
    o.acquisition = acquisition;
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
    o.dumpSpaceOnly = dumpSpaceOnly;
    o.nSolveWorkers = nSolveWorkers;
    o.graphAllocation = graphAllocation;
    o.programReloadMs = programReloadMs;
    o.latencyObjective = latencyObjective;
    o.allocationGranularity = allocationGranularity;
    upmemOpts.annotateOpCosts = annotateOpCosts;
    upmemOpts.useMRAMTiling = useMRAMTiling;
    upmemOpts.scatterSpecialisation = enableScatterSpecialisation;
    upmemOpts.fusionEdges = fusionEdges;
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
    o.evalSolutionForce = evalSolutionForce;
    return upmemOpts;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    UpmemInferenceOptions upmemOpts = buildOptions();

    // Which blocks are searched, and in what grouping, is the framework's
    // business (see GraphInference.h); this backend only says what "a UPMEM
    // block" is -- a block whose scope offers the `upmem` platform -- and how
    // to search one.
    auto makePlugin = [&upmemOpts](cinm::CinmPlatformAttrInterface platform)
        -> std::unique_ptr<cinm::InferencePlugin> {
      auto upmemPlatform = llvm::dyn_cast<upmem::UpmemPlatformAttr>(platform);
      if (!upmemPlatform)
        return nullptr;
      return std::make_unique<UpmemInferencePlugin>(
          upmemPlatform, upmemOpts,
          createSimulator(upmemOpts.simulator, upmemOpts.annotateOpCosts,
                          upmemOpts.evalTimeoutMs));
    };

    auto result = cinm::inferAcceleratorConfigs(
        module, kUpmemPlatformName, makePlugin, upmemOpts.inference);
    if (!result.succeeded()) {
      (void)result.checkAndReport();
      signalPassFailure();
    }
  }
};

} // namespace mlir::upmem
