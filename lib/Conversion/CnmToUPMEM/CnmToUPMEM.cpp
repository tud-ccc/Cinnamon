#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmScatterMap.h"
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

// A CNM map addresses (rank, dpu, tasklet) and, when a leaf receives more than
// one block, the buffer dimensions that index those blocks. An UPMEM map
// addresses whatever the chosen transfer form iterates -- (rank, dpu), or
// (rank, dpu, block) -- and always has one result per host dimension, giving
// the start of a transfer. `substitutions` says what to put in place of each
// CNM dimension, and `numDims` how many the result has.
static AffineMap rewriteMapForUpmem(AffineMap map, cnm::BufferType bufTy,
                                    ArrayRef<AffineExpr> substitutions,
                                    unsigned numDims) {
  assert(map.getNumDims() == substitutions.size());
  auto cst0 = getAffineConstantExpr(0, map.getContext());
  SmallVector<AffineExpr> exprs;
  for (AffineExpr e : map.getResults())
    exprs.push_back(e.replaceDims(substitutions));
  // The host dimensions the CNM map leaves implicit are the block's own shape,
  // so the transfer starts at their origin.
  for (int64_t i = 0, e = cnm::getNumImplicitHostDims(map, bufTy); i < e; ++i)
    exprs.push_back(cst0);
  return AffineMap::get(numDims, 0, std::move(exprs), map.getContext());
}

// The (rank, dpu, block) form, whose block dimension the UPMEM SDK's scatter
// transfer API evaluates once per block. A leaf's blocks vary fastest, so
// block index `b` is tasklet `b / blocksPerLeaf` at buffer position
// `b % blocksPerLeaf` inflated over the retained buffer dimensions.
static AffineMap keepTaskletDimAffineMapCnmToUpmem(AffineMap map,
                                                   cnm::BufferType bufTy) {
  MLIRContext *ctx = map.getContext();
  int64_t blocksPerLeaf = cnm::getScatterBlocksPerLeaf(map, bufTy);
  AffineExpr block = getAffineDimExpr(2, ctx);

  SmallVector<AffineExpr> substitutions{getAffineDimExpr(0, ctx),
                                        getAffineDimExpr(1, ctx),
                                        block.floorDiv(blocksPerLeaf)};
  ArrayRef<int64_t> retained = bufTy.getShape().take_front(
      cnm::getNumRetainedBufferDims(map, bufTy));
  if (!retained.empty()) {
    SmallVector<AffineExpr> coords;
    structureIndex(block % blocksPerLeaf, retained, coords);
    llvm::append_range(substitutions, coords);
  }
  return rewriteMapForUpmem(map, bufTy, substitutions, 3);
}

// How many blocks of the CNM map's implicit shape reach one DPU. A leaf may
// take several, and a DPU has `numTasklets` leaves -- unless the MRAM buffer
// is shared across them (isMramBroadcastOverThreads), in which case one
// leaf's worth is all that is stored.
static int64_t blocksPerDpu(AffineMap map, cnm::BufferType bufferTy,
                            size_t numTasklets, bool sharedAcrossTasklets) {
  int64_t perLeaf = cnm::getScatterBlocksPerLeaf(map, bufferTy);
  return sharedAcrossTasklets ? perLeaf
                              : static_cast<int64_t>(numTasklets) * perLeaf;
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
  const cnm::BufferType bufferTy = op.getBuffer().getType();
  // A DMA moves whole blocks, and the canonical map carries none, so derive
  // the widest one it allows. Anything the host layout leaves non-contiguous
  // stays a separate block.
  const AffineMap map =
      cnm::deflateScatterMap(op.getGatherMap(), bufferTy, op.getHostType());
  const int64_t perLeaf = cnm::getScatterBlocksPerLeaf(map, bufferTy);

  // A gathered buffer always has a per-tasklet dimension: results the
  // tasklets could not tell apart would race.
  upmem::GatherBlocksOp::create(
      rewriter, op->getLoc(), outputBuf, refToBuffer,
      op.getTransferCountInItems() / perLeaf,
      keepTaskletDimAffineMapCnmToUpmem(map, bufferTy),
      upmemWgAlloc.getResult(),
      blocksPerDpu(map, bufferTy, numTasklets,
                   /*sharedAcrossTasklets=*/false));

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
                                              bool sharedAcrossTasklets,
                                              upmem::AllocDPUsOp upmemWgAlloc,
                                              StringAttr refToBuffer) {

  rewriter.setInsertionPoint(op);
  const Value tensor = op.getInput();
  const ShapedType inputTy = op.getInput().getType();
  const MemRefType hostBufferTy = convertTensorToMemref(inputTy);

  const Value inputAsMemref = createOrFoldUnrealizedConversionCast(
      op.getLoc(), rewriter, hostBufferTy, tensor);

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const cnm::BufferType bufferTy = op.getBuffer().getType();
  // What the map leaves implicit is one contiguous run; a leaf may receive
  // several of them. The canonical map leaves nothing implicit, so derive the
  // widest block the map and the host layout allow.
  const AffineMap map =
      cnm::deflateScatterMap(op.getScatterMap(), bufferTy, inputTy);
  const int64_t perLeaf = cnm::getScatterBlocksPerLeaf(map, bufferTy);

  upmem::ScatterBlocksOp::create(
      rewriter, op->getLoc(), inputAsMemref, refToBuffer,
      op.getTransferCountInItems() / perLeaf,
      keepTaskletDimAffineMapCnmToUpmem(map, bufferTy),
      upmemWgAlloc.getResult(),
      blocksPerDpu(map, bufferTy, numTasklets, sharedAcrossTasklets));

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

// The slice of `mramBuf` belonging to the calling tasklet, shaped like
// `tileTy`. When the MRAM buffer carries a leading tasklet dimension (i.e. it
// is not broadcast over threads, see isMramBroadcastOverThreads) that is a
// subview indexed by the tasklet id; otherwise every tasklet sees the whole
// buffer and there is nothing to slice.
static Value getTaskletSlice(RewriterBase &rewriter, Location loc,
                             upmem::StaticAllocOp mramBuf, MemRefType tileTy) {
  auto mramBufTy = mramBuf.getBuffer().getType();
  if (mramBufTy.getRank() != tileTy.getRank() + 1)
    return mramBuf.getBuffer();

  auto taskletId = upmem::TaskletDimOp::create(rewriter, loc);

  SmallVector<OpFoldResult, 4> offsets(mramBufTy.getRank(),
                                       rewriter.getIndexAttr(0));
  offsets[0] = taskletId.getResult();

  SmallVector<OpFoldResult, 4> sizes;
  sizes.push_back(rewriter.getIndexAttr(1));
  for (auto size : tileTy.getShape())
    sizes.push_back(rewriter.getIndexAttr(size));

  llvm::SmallVector<OpFoldResult, 4> strides(mramBufTy.getRank(),
                                             rewriter.getIndexAttr(1));

  auto [baseStrides, baseOffset] = mramBufTy.getStridesAndOffset();

  // this is the type of the tile. We cannot let it be inferred as it may be
  // rank-reduced.
  MemRefType viewType = MemRefType::get(
      tileTy.getShape(), tileTy.getElementType(),
      rewriter.getAttr<StridedLayoutAttr>(
          ShapedType::kDynamic, ArrayRef<long>(baseStrides).drop_front()),
      mramBufTy.getMemorySpace());

  return memref::SubViewOp::create(rewriter, loc, viewType,
                                   mramBuf.getBuffer(), offsets, sizes,
                                   strides);
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

  Operation *insertionPointReset = nullptr;
  if (mramHasTaskletDim) {
    mramBufToScatter = getTaskletSlice(rewriter, loc, mramBuf, wramBufTy);
  } else if (isWramShared(wramBuffer)) {
    auto taskletId = upmem::TaskletDimOp::create(rewriter, loc);
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
      // The map addresses (rank, dpu, tasklet) and may go on to address the
      // buffer dimensions a leaf receives one block each of; only the tasklet
      // dimension matters here.
      if (map.getNumDims() < 3)
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

// Give the ops a launch body uses to stage its own buffers their UPMEM
// equivalents. Runs over the whole DPU program after the body has been cloned
// in, because the staging sits inside the body's loop nests and cloning copies
// regions wholesale.
static LogicalResult lowerBodyStagingOps(RewriterBase &rewriter,
                                         upmem::DpuProgramOp dpuProgram,
                                         Attribute wramMemspace) {
  SmallVector<Operation *> toRewrite;
  dpuProgram->walk([&](Operation *op) {
    if (isa<cnm::LocalTransferOp, memref::AllocOp, memref::DeallocOp>(op))
      toRewrite.push_back(op);
  });

  for (Operation *op : toRewrite) {
    rewriter.setInsertionPoint(op);

    if (auto transfer = dyn_cast<cnm::LocalTransferOp>(op)) {
      rewriter.replaceOpWithNewOp<upmem::LocalTransferOp>(
          transfer, transfer.getSource(), transfer.getTarget());
      continue;
    }

    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      auto type = alloc.getType();
      if (type.getMemorySpace() != wramMemspace)
        return alloc->emitOpError("cannot be lowered to UPMEM: only WRAM "
                                  "allocations are supported inside a launch "
                                  "body");
      // WRAM scratch is a per-tasklet allocation carved out of the WRAM
      // partition, which is what upmem.pwram_alloc denotes.
      rewriter.replaceOpWithNewOp<memref::AllocaOp>(alloc, type);
      continue;
    }

    // The WRAM partition is reclaimed when the kernel returns, so a matching
    // deallocation has nothing to do.
    auto dealloc = cast<memref::DeallocOp>(op);
    if (cast<MemRefType>(dealloc.getMemref().getType()).getMemorySpace() ==
        wramMemspace)
      rewriter.eraseOp(dealloc);
  }
  return success();
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
  // Buffers the launch body computes on in MRAM directly. It has already been
  // given its own WRAM staging (--upmem-tile-mram-buffers), so this pass must
  // not add a second one around it: the body binds straight to MRAM.
  llvm::DenseSet<Value> mramLevelBuffers;

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
      bool stagedInBody = bufferType.getLevel() == mramMemspaceAttr;

      if (stagedInBody) {
        // No WRAM buffer and no transfers here: the body already stages what
        // it needs. It binds to the tasklet's slice of the MRAM buffer, which
        // is created below once its shape is known.
        mramLevelBuffers.insert(alloc.getResult());
      } else if (wramIsShared) {
        // If all threads see the same buffer (broadcast), then we only
        // create one static buffer in WRAM.
        auto wrambuf =
            upmem::StaticAllocOp::create(rewriter, alloc->getLoc(), memrefTy,
                                         upmem::DpuMemSpace::WRAM, "buf", true);
        dpuProgramSymTable.insert(wrambuf); // this renames it to a unique name
        buffersToWramBufValue[alloc.getResult()] = wrambuf.getBuffer();
      } else {
        // WRAM is private - each tasklet gets its own buffer.
        auto pwramBuf = memref::AllocaOp::create(
            rewriter, alloc.getLoc(), memrefTy);

        buffersToWramBufValue[alloc.getResult()] = pwramBuf.getResult();
      }

      if (!mramIsBroadcast) {
        // the mram buffer type has tasklet dimension prepended - unless the
        // buffer is broadcasted.
        bufShape.insert(bufShape.begin(), upmemTy.getNumTaskletsPerDpu());
      }
      (void)memrefTy;

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
      // A DPU only stores one leaf's worth when the MRAM buffer has no
      // per-tasklet leading dimension (isMramBroadcastOverThreads). This is
      // independent of whether WRAM ends up shared.
      bool sharedAcrossTasklets =
          alloc && alloc.getBuffer().getType().getRank() ==
                       static_cast<int64_t>(
                           scatter.getBuffer().getType().getShape().size());

      if (!alloc || failed(convertCnmScatterToUpmem(
                        rewriter, scatter, sharedAcrossTasklets, upmemWgAlloc,
                        alloc.getSymNameAttr()))) {
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
  rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  for (auto [cnmBuf, memref] : llvm::zip_equal(
           llvm::concat<Value>(launch.getInputs(), launch.getOutBuffers()),
           launch.getBody().getArguments())) {

    if (mramLevelBuffers.contains(cnmBuf)) {
      // The body computes on MRAM: bind it to this tasklet's slice.
      mapping.map(memref, getTaskletSlice(rewriter, cnmBuf.getLoc(),
                                          buffersToMramBuf[cnmBuf],
                                          cast<MemRefType>(memref.getType())));
      continue;
    }
    auto wrambuf = buffersToWramBufValue.lookup(cnmBuf);
    mapping.map(memref, wrambuf);
  }

  // todo support moving tiles of the mram buffer into pwram
  rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  for (auto [buf, mramBuf] : buffersToMramBuf) {
    if (mramLevelBuffers.contains(buf))
      continue;
    auto wramBuf = buffersToWramBufValue[buf];
    createTransfer(rewriter, true, buf.getLoc(), mramBuf, wramBuf);
    rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  }

  // copy the old ops
  for (auto &op : launch.getBody().front().without_terminator())
    rewriter.clone(op, mapping);

  // transfer buffers back to mram
  for (auto buf : launch.getOutBuffers()) {
    if (mramLevelBuffers.contains(buf))
      continue;
    auto wramBuf = buffersToWramBufValue[buf];
    auto mramBuf = buffersToMramBuf[buf];

    createTransfer(rewriter, false, buf.getLoc(), mramBuf, wramBuf);
    rewriter.setInsertionPointToEnd(&dpuProgram.getBody().front());
  }

  upmem::ReturnOp::create(rewriter, launch->getLoc());

  if (failed(lowerBodyStagingOps(rewriter, dpuProgram, wramMemspaceAttr)))
    return failure();

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
    Opts opts{.cinm1codegen = cinm1Codegen, .useMramNoInit = !cinm1Codegen};

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
