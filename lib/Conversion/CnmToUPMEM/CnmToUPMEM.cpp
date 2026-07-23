#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <cinm-mlir/Dialect/UPMEM/Transforms/Utils.h>
#include <cinm-mlir/Utils/CinmUtils.h>
#include <cstdint>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CONVERTCNMTOUPMEMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace {

struct Opts {
  bool cinm1codegen = false;
  bool useSgXferCodegen = true;
  bool useBcXferCodegen = true;
  bool useMramNoInit = true;
};

template <typename T> T reduceMul(ArrayRef<T> arr) {
  T result{1};
  for (const T &elem : arr) {
    result *= elem;
  }
  return result;
}

MemRefType convertTensorToMemref(ShapedType ty) {
  if (isa<MemRefType>(ty))
    return cast<MemRefType>(ty);

  return MemRefType::get(ty.getShape(), ty.getElementType());
}

// In CNM the affine map has 1 dim for rank, 1 for dpu, 1 for tasklet.
// In upmem it has only one dim for rank and another for dpu. Dimensions
// of the buffer shape are zero (they are the offset of the buffer start).
//
// TODO: whether zeroing out the last dimension is valid should be checked by
// the verifier before we reach this point.
//  That works only if the input affine map either:
//  - does not use t (broadcast), or
//  - uses t at the smallest varying dim, ie t is only used in the last result
//  and as `+ t` (not eg + 6 * t).
static AffineMap adaptAffineMapCnmToUpmem(AffineMap map,
                                          cnm::BufferType bufTy) {
  assert(map.getNumDims() == 3);
  auto rankDim = getAffineDimExpr(0, map.getContext());
  auto dpuDim = getAffineDimExpr(1, map.getContext());
  auto cst0 = getAffineConstantExpr(0, map.getContext());
  SmallVector<AffineExpr> exprs;
  for (auto e : map.getResults()) {
    exprs.push_back(e.replaceDims({rankDim, dpuDim, cst0}));
  }
  for (auto _ : bufTy.getShape()) {
    exprs.push_back(cst0);
  }
  return AffineMap::get(2, 0, std::move(exprs), map.getContext());
}

// Same as adaptAffineMapCnmToUpmem, but keeps the tasklet dim (dim 2) instead
// of zeroing it out. Used for the upmem.scatter (rank, dpu, tasklet) form,
// where the tasklet dim is evaluated per-tasklet by the UPMEM SDK's scatter
// transfer API rather than being folded into a single flat per-DPU memcpy.
static AffineMap keepTaskletDimAffineMapCnmToUpmem(AffineMap map,
                                                   cnm::BufferType bufTy) {
  assert(map.getNumDims() == 3);
  auto cst0 = getAffineConstantExpr(0, map.getContext());
  SmallVector<AffineExpr> exprs(map.getResults());
  for (auto _ : bufTy.getShape()) {
    exprs.push_back(cst0);
  }
  return AffineMap::get(3, 0, std::move(exprs), map.getContext());
}

// Linearizes `map` (assumed to already have one result per dimension of
// `hostBufferTy`) into a single element-offset expression in the map's
// dims, using `hostBufferTy`'s layout to convert a multi-dim index into an
// element offset. The base offset of a strided layout is dropped, since only
// relative (stride) information matters for the contiguity check below.
// Returns failure if the layout isn't identity or a static StridedLayoutAttr.
static FailureOr<AffineExpr> linearizeToElementOffset(AffineMap map,
                                                      MemRefType hostBufferTy) {
  MLIRContext *ctx = map.getContext();
  ArrayRef<int64_t> shape = hostBufferTy.getShape();
  AffineMap layoutMap;
  if (hostBufferTy.getLayout().isIdentity()) {
    layoutMap =
        AffineMap::get(shape.size(), 0, linearizeIndices(ctx, shape), ctx);
  } else if (auto strided =
                dyn_cast<StridedLayoutAttr>(hostBufferTy.getLayout())) {
    AffineExpr linear = getAffineConstantExpr(0, ctx);
    for (auto [i, stride] : llvm::enumerate(strided.getStrides())) {
      if (ShapedType::isDynamic(stride))
        return failure();
      linear = linear + getAffineDimExpr(i, ctx) * stride;
    }
    layoutMap = AffineMap::get(shape.size(), 0, linear, ctx);
  } else {
    return failure();
  }

  MutableAffineMap composed(layoutMap.compose(map));
  composed.simplify();
  assert(composed.getAffineMap().getNumResults() == 1);
  return composed.getAffineMap().getResult(0);
}

// Returns the constant coefficient of `dim` in `expr`, or nullopt if `expr`
// doesn't vary with `dim` in a simple affine (constant-coefficient) way.
static std::optional<int64_t>
getAffineExprDimCoefficient(AffineExpr expr, unsigned dim, unsigned numDims) {
  MLIRContext *ctx = expr.getContext();
  SmallVector<AffineExpr> substAt0, substAt1;
  for (unsigned i = 0; i < numDims; ++i) {
    substAt0.push_back(getAffineDimExpr(i, ctx));
    substAt1.push_back(getAffineDimExpr(i, ctx));
  }
  substAt0[dim] = getAffineConstantExpr(0, ctx);
  substAt1[dim] = getAffineConstantExpr(1, ctx);

  AffineExpr diff =
      expr.replaceDims(substAt1) - expr.replaceDims(substAt0);
  diff = simplifyAffineExpr(diff, numDims, 0);
  if (auto cst = dyn_cast<AffineConstantExpr>(diff))
    return cst.getValue();
  return std::nullopt;
}

// Whether, for every (rank, dpu), the per-tasklet blocks addressed by
// `scatterMap(rank, dpu, tasklet)` -- each `blockSizeInItems` elements long
// -- are laid out back-to-back in `hostBufferTy`, in tasklet order, forming
// one contiguous run of `numTasklets * blockSizeInItems` elements. This is
// exactly the condition under which collapsing to the classic (rank, dpu)
// upmem.scatter form (a single flat memcpy per DPU) is correct. We check
// this by verifying that the element offset (linearized using the host
// buffer's actual layout) is affine in the tasklet dim with a coefficient of
// exactly `blockSizeInItems`.
static bool taskletBlocksAreContiguous(AffineMap scatterMap,
                                       cnm::BufferType bufTy,
                                       MemRefType hostBufferTy,
                                       int64_t blockSizeInItems) {
  AffineMap extended = keepTaskletDimAffineMapCnmToUpmem(scatterMap, bufTy);
  FailureOr<AffineExpr> offset =
      linearizeToElementOffset(extended, hostBufferTy);
  if (failed(offset))
    return false;
  std::optional<int64_t> coeff =
      getAffineExprDimCoefficient(*offset, /*dim=*/2, /*numDims=*/3);
  return coeff.has_value() && *coeff == blockSizeInItems;
}

// Whether `scatterMap` (the CNM (rank, dpu, tasklet) -> host index map) does
// not depend on any of its three dimensions and evaluates to the origin --
// i.e. every DPU's tasklets all read the exact same region of the host
// buffer, starting at its very first element. This is strictly stronger than
// the tasklet-only broadcast used to decide MRAM layout (see
// isMramBroadcastOverThreads), which only requires the tasklet dim to be
// unused: `upmem.broadcast` (unlike `upmem.scatter`) has no affine map at
// all, so every DPU in the hierarchy must get byte-for-byte identical data
// straight from the start of the host buffer.
static bool isGloballyBroadcast(AffineMap scatterMap) {
  auto unusedDims = getUnusedDimsBitVector({scatterMap});
  if (!unusedDims[0] || !unusedDims[1] || !unusedDims[2])
    return false;
  if (!scatterMap.isConstant())
    return false;
  // todo in the future detect offset != 0 and turn it into a subview then broadcast
  return llvm::all_of(scatterMap.getConstantResults(),
                      [](int64_t v) { return v == 0; });
}

static LogicalResult convertCnmGatherToUpmem(RewriterBase &rewriter,
                                             cnm::GatherOp op,
                                             upmem::AllocDPUsOp upmemWgAlloc,
                                             StringAttr refToBuffer) {

  rewriter.setInsertionPoint(op);
  Value outputBuf = op.getOutputBuf();
  bool isBufferized = isa<BaseMemRefType>(op.getOutputBuf().getType());
  if (!isBufferized) {
    outputBuf = memref::AllocOp::create(
        rewriter, op->getLoc(),
        convertTensorToMemref(op.getOutputBuf().getType()));
  }

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const int64_t transferCount = op.getTransferCountInItems() * numTasklets;

  upmem::GatherOp::create(
      rewriter, op->getLoc(), outputBuf, refToBuffer, transferCount,
      adaptAffineMapCnmToUpmem(op.getGatherMap(), op.getBuffer().getType()),
      upmemWgAlloc.getResult());

  if (!isBufferized) {
    Value outputAsTensor = createOrFoldUnrealizedConversionCast(
        op->getLoc(), rewriter, op.getOutput().getType(), outputBuf);

    rewriter.replaceAllUsesWith(op.getOutput(), outputAsTensor);
  }
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult convertCnmScatterToUpmem(RewriterBase &rewriter,
                                              cnm::ScatterOp op,
                                              bool isBroadcast,
                                              upmem::AllocDPUsOp upmemWgAlloc,
                                              StringAttr refToBuffer,
                                              const Opts &opts) {

  rewriter.setInsertionPoint(op);
  const Value tensor = op.getInput();
  const ShapedType inputTy = op.getInput().getType();
  const MemRefType hostBufferTy = convertTensorToMemref(inputTy);

  const Value inputAsMemref = createOrFoldUnrealizedConversionCast(
      op.getLoc(), rewriter, hostBufferTy, tensor);

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const int64_t blockSizeInItems = op.getTransferCountInItems();

  // The classic upmem.scatter (rank, dpu) form performs a single flat memcpy
  // per DPU of numTasklets*blockSizeInItems elements: correct only when
  // those per-tasklet blocks are actually contiguous in the host buffer (see
  // adaptAffineMapCnmToUpmem). When they aren't, instead of forcing a
  // staging copy upstream, we can use the (rank, dpu, tasklet) form, which
  // uses the UPMEM SDK's scatter transfer API (dpu_push_sg_xfer) to gather
  // each tasklet's (still individually contiguous) block directly from its
  // real, possibly non-contiguous, location.
  bool useTaskletForm =
      !isBroadcast && numTasklets > 1 && opts.useSgXferCodegen &&
      !taskletBlocksAreContiguous(op.getScatterMap(), op.getBuffer().getType(),
                                  hostBufferTy, blockSizeInItems);

  // When every DPU's tasklets read byte-for-byte identical data straight
  // from the start of the host buffer (isGloballyBroadcast), and that data
  // is already exactly `hostBuffer`'s own contents (no slicing needed), we
  // can skip the affine map entirely and use upmem.broadcast -- one runtime
  // call broadcasting the whole buffer to every DPU, rather than a per-DPU
  // scatter transfer that happens to always fetch the same bytes. This is
  // gated behind !cinm1codegen the same way WRAM sharing already is (see
  // wramIsShared above): cinm1-codegen's DPU-side code doesn't expect this
  // shortcut, only the plain upmem.scatter (rank, dpu) form.
  bool useBroadcastOp = isBroadcast && opts.useBcXferCodegen &&
                       hostBufferTy.hasStaticShape() &&
                       hostBufferTy.getNumElements() == blockSizeInItems &&
                       isGloballyBroadcast(op.getScatterMap());

  if (useTaskletForm) {
    AffineMap upmemMap = keepTaskletDimAffineMapCnmToUpmem(
        op.getScatterMap(), op.getBuffer().getType());
    int64_t transferCount = blockSizeInItems;
    upmem::ScatterOnTaskletsOp::create(rewriter, op->getLoc(), inputAsMemref,
                                       refToBuffer, transferCount, upmemMap,
                                       upmemWgAlloc.getResult(),
                                       static_cast<int64_t>(numTasklets));
  } else if (useBroadcastOp) {
    upmem::BroadcastOp::create(rewriter, op->getLoc(), inputAsMemref,
                               refToBuffer, upmemWgAlloc.getResult());
  } else {
    AffineMap upmemMap =
        adaptAffineMapCnmToUpmem(op.getScatterMap(), op.getBuffer().getType());
    int64_t transferCount =
        isBroadcast ? blockSizeInItems : blockSizeInItems * numTasklets;
    upmem::ScatterOp::create(rewriter, op->getLoc(), inputAsMemref,
                             refToBuffer, transferCount, upmemMap,
                             upmemWgAlloc.getResult());
  }

  rewriter.eraseOp(op);
  return success();
}

// static MemRefType withMemrefMemspace(MemRefType fromTy, Attribute memspace) {
//   return MemRefType::get(fromTy.getShape(), fromTy.getElementType(),
//                          fromTy.getLayout(), memspace);
// }

// Whether `wramBuffer` is a single copy shared by every tasklet (as opposed
// to a private per-tasklet WRAM buffer). MRAM sharing (see
// isMramBroadcastOverThreads) and WRAM sharing are independent decisions:
// under cinm1-codegen, WRAM is never shared, but MRAM can still be a single
// copy that every tasklet loads from into its own private WRAM buffer.
static bool isWramShared(TypedValue<MemRefType> wramBuffer) {
  return isa_and_nonnull<upmem::StaticAllocOp>(wramBuffer.getDefiningOp());
}

static void createTransfer(RewriterBase &rewriter, bool toWram, Location loc,
                           upmem::StaticAllocOp mramBuf,
                           TypedValue<MemRefType> wramBuffer) {

  auto mramBufTy = mramBuf.getBuffer().getType();
  auto wramBufTy = wramBuffer.getType();
  // The MRAM buffer has an extra leading tasklet dimension whenever it isn't
  // itself broadcast over threads (see isMramBroadcastOverThreads).
  bool mramHasTaskletDim = mramBufTy.getRank() == wramBufTy.getRank() + 1;
  Value mramBufToScatter;

  auto taskletId = upmem::TaskletDimOp::create(rewriter, loc);

  Operation *insertionPointReset = nullptr;
  if (mramHasTaskletDim) {
    // scatter over tasklets
    SmallVector<OpFoldResult, 4> offsets(mramBufTy.getRank(),
                                         rewriter.getIndexAttr(0));
    offsets[0] = taskletId.getResult();

    SmallVector<OpFoldResult, 4> sizes;
    sizes.push_back(rewriter.getIndexAttr(1));
    for (auto size : wramBufTy.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(size));
    }

    llvm::SmallVector<OpFoldResult, 4> strides(mramBufTy.getRank(),
                                               rewriter.getIndexAttr(1));

    auto [baseStrides, baseOffset] = mramBufTy.getStridesAndOffset();

    // this is the type of the tile. We cannot let it be inferred as it may be
    // rank-reduced.
    MemRefType viewType = MemRefType::get(
        wramBufTy.getShape(), wramBufTy.getElementType(),
        rewriter.getAttr<StridedLayoutAttr>(
            ShapedType::kDynamic, ArrayRef<long>(baseStrides).drop_front()),
        mramBufTy.getMemorySpace());

    mramBufToScatter = memref::SubViewOp::create(
        rewriter, loc, viewType, mramBuf.getBuffer(), offsets, sizes, strides);
  } else if (isWramShared(wramBuffer)) {
    // MRAM buffer corresponds exactly to WRAM buffer, and WRAM is shared:
    // this is a full broadcast (every tasklet reads the same WRAM copy).
    mramBufToScatter = mramBuf.getBuffer();

    // In that case we need to make only thread 0 call
    // for the transfer
    auto cst0 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(taskletId.getResult().getType()));
    auto isTaskletZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, taskletId, cst0);
    auto scfIf = scf::IfOp::create(rewriter, loc, isTaskletZero, false);
    // Create a barrier so that all threads wait for the transfer to finish.
    // This is only ok if the transfer is from MRAM to WRAM, then threads are
    // waiting for their inputs to be loaded into WRAM.
    auto barrier = upmem::BarrierOp::create(rewriter, loc);
    // If we are transferring back to MRAM, then the barrier needs to be instead
    // _before_ the transfer, that way we make sure all threads are done before
    // writing back.
    if (!toWram) {
      barrier->remove();
      rewriter.setInsertionPoint(scfIf);
      rewriter.insert(barrier);
    }

    // Position the rewriter so that the transfer is written inside the
    // conditional block
    rewriter.setInsertionPointToStart(&scfIf.getThenRegion().front());
    insertionPointReset = scfIf;
  } else {
    // MRAM is broadcast (single shared copy) but WRAM is private per-tasklet
    // (cinm1-codegen): every tasklet independently loads its own copy from
    // the same MRAM location into its own private WRAM buffer. No subview,
    // no restriction to a single tasklet, no barrier needed.
    assert(toWram && "a broadcast MRAM buffer should never be an output "
                     "(outputs always have a gather, disqualifying MRAM "
                     "broadcast -- see isMramBroadcastOverThreads)");
    mramBufToScatter = mramBuf.getBuffer();
  }

  if (toWram)
    upmem::LocalTransferOp::create(rewriter, loc, mramBufToScatter, wramBuffer);
  else
    upmem::LocalTransferOp::create(rewriter, loc, wramBuffer, mramBufToScatter);

  if (insertionPointReset)
    rewriter.setInsertionPointAfter(insertionPointReset);
}

// Whether every scatter into `alloc` addresses the buffer without using the
// thread dimension of the affine map (and there is no gather reading it
// back, which would require distinguishable per-tasklet results). This is
// purely a property of the CNM scatter maps, independent of whether WRAM
// ends up shared or private for this buffer: even when WRAM is private per
// tasklet (cinm1-codegen), a single shared MRAM copy is enough, since every
// tasklet can load the same MRAM location into its own private WRAM buffer.
static bool isMramBroadcastOverThreads(cnm::AllocOp alloc) {
  for (auto user : alloc->getUsers()) {
    if (llvm::isa<cnm::GatherOp>(user))
      return false;
    if (auto scatter = llvm::dyn_cast_or_null<cnm::ScatterOp>(user)) {
      auto map = scatter.getScatterMap();
      if (map.getNumDims() != 3)
        return false;
      auto unusedDims = getUnusedDimsBitVector({map});
      if (!unusedDims[2]) {
        // threads dim is used so all threads see the same buffer
        return false;
      }
    }
  }

  return true;
}

static LogicalResult convertCnmLaunchToUpmem(cnm::LaunchOp launch,
                                             RewriterBase &rewriter, Opts opts,
                                             SymbolTable rootModule,
                                             ModuleOp dpuKernelModule) {

  rewriter.clearInsertionPoint();

  auto wg = launch.getWg().getType().getShape();
  if (wg.size() != 3)
    return launch.emitOpError("Should have working group with 3 entries");

  const auto upmemTy =
      rewriter.getType<upmem::DeviceHierarchyType>(wg[0], wg[1], wg[2]);

  auto dpuProgram = upmem::DpuProgramOp::create(
      rewriter, launch->getLoc(), "program", upmemTy.getNumTaskletsPerDpu());
  dpuProgram.getBody().emplaceBlock();
  SymbolTable symTable(dpuKernelModule);
  symTable.insert(dpuProgram);

  auto programPath = upmem::getSymbolPath(rootModule, dpuProgram);
  assert(llvm::succeeded(programPath));

  auto wgAlloc = cast<cnm::WorkgroupOp>(launch.getWg().getDefiningOp());
  rewriter.setInsertionPoint(wgAlloc);
  auto upmemWgAlloc = upmem::AllocDPUsOp::create(rewriter, wgAlloc->getLoc(),
                                                 upmemTy, *programPath);

  llvm::MapVector<Value, upmem::StaticAllocOp> buffersToMramBuf;
  // llvm::MapVector<Value, upmem::StaticAllocOp> buffersToSharedWramBuf;
  llvm::MapVector<Value, TypedValue<MemRefType>> buffersToWramBufValue;

  rewriter.setInsertionPointToStart(&dpuProgram.getBody().front());

  // create named static MRAM buffers for each cnm.alloc operation, put them in
  // the dpu program
  SmallVector<AllocOp> allocsToDelete;

  SymbolTable dpuProgramSymTable(dpuProgram);
  auto mramMemspaceAttr =
      rewriter.getAttr<upmem::DpuMemSpaceAttr>(upmem::DpuMemSpace::MRAM);
  auto wramMemspaceAttr =
      rewriter.getAttr<upmem::DpuMemSpaceAttr>(upmem::DpuMemSpace::WRAM);

  for (auto user : launch.getWg().getUsers()) {
    if (auto alloc = llvm::dyn_cast_or_null<cnm::AllocOp>(user)) {
      allocsToDelete.push_back(alloc);

      auto bufferType = alloc.getType();

      SmallVector<int64_t> bufShape(bufferType.getShape());

      // the pwram buffer has the shape we expect
      MemRefType memrefTy =
          MemRefType::get(bufShape, bufferType.getElementType(),
                          MemRefLayoutAttrInterface{}, wramMemspaceAttr);

      // MRAM sharing only depends on the scatter maps (isMramBroadcastOverThreads);
      // WRAM sharing additionally requires that cinm1-codegen isn't forcing
      // private per-tasklet WRAM buffers.
      bool mramIsBroadcast = isMramBroadcastOverThreads(alloc);
      bool wramIsShared = !opts.cinm1codegen && mramIsBroadcast;

      if (wramIsShared) {
        // If all threads see the same buffer (broadcast), then we only
        // create one static buffer in WRAM.
        auto wrambuf =
            upmem::StaticAllocOp::create(rewriter, alloc->getLoc(), memrefTy,
                                         upmem::DpuMemSpace::WRAM, "buf", true);
        dpuProgramSymTable.insert(wrambuf); // this renames it to a unique name
        buffersToWramBufValue[alloc.getResult()] = wrambuf.getBuffer();
      } else {
        // WRAM is private - each tasklet gets its own buffer.
        auto pwramBuf = upmem::PrivateWRAMAllocOp::create(
            rewriter, alloc.getLoc(), memrefTy);

        buffersToWramBufValue[alloc.getResult()] = pwramBuf.getBuffer();
      }

      if (!mramIsBroadcast) {
        // the mram buffer type has tasklet dimension prepended - unless the
        // buffer is broadcasted.
        bufShape.insert(bufShape.begin(), upmemTy.getNumTaskletsPerDpu());
      }

      memrefTy = MemRefType::get(bufShape, bufferType.getElementType(),
                                 MemRefLayoutAttrInterface{}, mramMemspaceAttr);

      auto mrambuf =
          upmem::StaticAllocOp::create(rewriter, alloc->getLoc(), memrefTy,
                                       upmem::DpuMemSpace::MRAM, "buf", opts.useMramNoInit);
      dpuProgramSymTable.insert(mrambuf); // this renames it to a unique name
      buffersToMramBuf[alloc.getResult()] = mrambuf;
    }
  }

  // then, replace all scatter/gather with the upmem equivalents

  for (auto user : launch.getWg().getUsers()) {

    if (auto scatter = llvm::dyn_cast_or_null<cnm::ScatterOp>(user)) {
      upmem::StaticAllocOp alloc = buffersToMramBuf.lookup(scatter.getBuffer());
      // transferCount only needs the numTasklets multiplier when the MRAM
      // buffer itself has a per-tasklet leading dimension, i.e. isn't
      // broadcast -- this is independent of whether WRAM ends up shared.
      bool isBroadcast =
          alloc && alloc.getBuffer().getType().getRank() ==
                       static_cast<int64_t>(
                           scatter.getBuffer().getType().getShape().size());

      if (!alloc || failed(convertCnmScatterToUpmem(
                        rewriter, scatter, isBroadcast, upmemWgAlloc,
                        alloc.getSymNameAttr(), opts))) {
        return failure();
      }
    }
    if (auto gather = llvm::dyn_cast_or_null<cnm::GatherOp>(user)) {
      auto alloc = buffersToMramBuf.lookup(gather.getBuffer());
      if (!alloc ||
          failed(convertCnmGatherToUpmem(rewriter, gather, upmemWgAlloc,
                                         alloc.getSymNameAttr()))) {
        return failure();
      }
    }
  }

  // At this point we have replaced (and deleted) the scatter and gather.
  // We still need to move the body of the launch into the new DPU program,
  // and delete all remaining CNM ops.

  // these memrefs map to the pwram bufs. TODO we need to transfer from mram to
  // pwram
  IRMapping mapping;
  for (auto [cnmBuf, memref] : llvm::zip_equal(
           llvm::concat<Value>(launch.getInputs(), launch.getOutBuffers()),
           launch.getBody().getArguments())) {

    auto wrambuf = buffersToWramBufValue.lookup(cnmBuf);
    mapping.map(memref, wrambuf);
  }

  // todo support moving tiles of the mram buffer into pwram
  rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  for (auto [buf, mramBuf] : buffersToMramBuf) {
    auto wramBuf = buffersToWramBufValue[buf];
    createTransfer(rewriter, true, buf.getLoc(), mramBuf, wramBuf);
    rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  }

  // copy the old ops
  for (auto &op : launch.getBody().front().without_terminator()) {
    rewriter.clone(op, mapping);
  }

  // transfer buffers back to mram
  for (auto buf : launch.getOutBuffers()) {
    auto wramBuf = buffersToWramBufValue[buf];
    auto mramBuf = buffersToMramBuf[buf];

    createTransfer(rewriter, false, buf.getLoc(), mramBuf, wramBuf);
    rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  }

  upmem::ReturnOp::create(rewriter, launch->getLoc());

  rewriter.setInsertionPoint(launch);
  upmem::WaitForOp::create(rewriter, launch->getLoc(),
                           upmemWgAlloc.getResult());

  // cleanup

  rewriter.eraseOp(launch);
  for (auto op : allocsToDelete) {
    if (op->getResults().use_empty()) {
      rewriter.eraseOp(op);
    } else {
      op->emitOpError("should have no uses");
      return failure();
    }
  }

  for (auto user : wgAlloc.getResult().getUsers()) {
    if (auto free = llvm::dyn_cast_or_null<cnm::FreeWorkgroupOp>(user)) {
      rewriter.setInsertionPoint(free);
      upmem::FreeDPUsOp::create(rewriter, free->getLoc(),
                                upmemWgAlloc.getResult());
      rewriter.eraseOp(free);
    }
  }

  if (wgAlloc.getResult().use_empty()) {
    rewriter.eraseOp(wgAlloc);
    return success();
  } else {
    return wgAlloc.getResult().user_begin()->emitOpError(
        "Unexpected workgroup usage");
  }
}

struct ConvertCnmTerminatorToUPMEM
    : public OpConversionPattern<cnm::TerminatorOp> {
  using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op); // gets generated by ConvertCnmLaunchToUPMEM
    return success();
  }
};

} // namespace

struct ConvertCnmToUPMEMPass
    : public impl::ConvertCnmToUPMEMPassBase<ConvertCnmToUPMEMPass> {
  using Base::Base;

  void runOnOperation() final {
    Operation *rootOp = getOperation();
    Opts opts{.cinm1codegen = cinm1Codegen,
              .useSgXferCodegen = useSgXferCodegen,
              .useBcXferCodegen = useBcXferCodegen,
              .useMramNoInit = !cinm1Codegen};

    // Determine kernel module name: prefer per-op annotation, else option.
    std::string kmName = kernelModuleName;
    if (auto attr = rootOp->getAttrOfType<StringAttr>("upmem.kernel_module"))
      kmName = attr.getValue().str();

    // Find the enclosing ModuleOp (or use rootOp itself if it is one).
    ModuleOp parentModule = llvm::dyn_cast<ModuleOp>(rootOp);
    if (!parentModule)
      parentModule = rootOp->getParentOfType<ModuleOp>();
    if (!parentModule) {
      mlir::emitError(rootOp->getLoc(), "No parent ModuleOp found");
      signalPassFailure();
      return;
    }

    auto sym = SymbolTable::lookupSymbolIn(parentModule, kmName);
    ModuleOp dpuKernelModule = llvm::dyn_cast_or_null<ModuleOp>(sym);
    if (!dpuKernelModule && sym) {
      mlir::emitError(sym->getLoc(), "Should be a module");
      signalPassFailure();
      return;
    }
    if (!dpuKernelModule) {
      OpBuilder builder(&getContext());
      builder.setInsertionPointToEnd(&parentModule.getBodyRegion().front());
      dpuKernelModule =
          ModuleOp::create(builder, parentModule->getLoc(), kmName);
    }

    SmallVector<LaunchOp> launchOps;
    rootOp->walk([&](cnm::LaunchOp launch) { launchOps.push_back(launch); });

    SymbolTable rootSymTable(parentModule);

    IRRewriter rewriter(&getContext());
    for (auto launch : launchOps) {
      if (failed(convertCnmLaunchToUpmem(launch, rewriter, opts, rootSymTable,
                                         dpuKernelModule))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<Pass> createConvertCnmToUPMEMPass() {
  return std::make_unique<ConvertCnmToUPMEMPass>();
}
std::unique_ptr<Pass>
createConvertCnmToUPMEMPass(ConvertCnmToUPMEMPassOptions options) {
  return std::make_unique<ConvertCnmToUPMEMPass>(std::move(options));
}

} // namespace mlir::cnm
