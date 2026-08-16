/// Implements the UPMEM dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Utils/CinmUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "upmem-ops"

using namespace mlir;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.cpp.inc"

//===----------------------------------------------------------------------===//
// UPMEMDialect
//===----------------------------------------------------------------------===//

void upmem::UPMEMDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.cpp.inc"
      >();
}

// ===----------------------------------------------------------------------===//
// getDpuProgram helpers
// ===----------------------------------------------------------------------===//

// LoadProgramOp owns the symbol reference, so it does the real lookup.
upmem::DpuProgramOp upmem::LoadProgramOp::getDpuProgram() {
  auto *sym =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getDpuProgramRef());
  return dyn_cast_or_null<upmem::DpuProgramOp>(sym);
}

/// The program resident on `hierarchy` when `at` executes. Which program a
/// set holds is a property of the point in the schedule, not of the value:
/// the nearest upmem.load_program preceding `at` in its own block decides.
/// When no load precedes it there (the load can sit in an ancestor region,
/// e.g. at the top of the container function while `at` is inside a compute
/// block), a unique load anywhere on the value is unambiguous and is used;
/// several loads none of which precedes `at` locally cannot be told apart
/// without dominance analysis, and this returns null rather than guessing.
static upmem::DpuProgramOp programLoadedOn(Value hierarchy, Operation *at) {
  for (Operation *prev = at->getPrevNode(); prev; prev = prev->getPrevNode())
    if (auto load = dyn_cast<upmem::LoadProgramOp>(prev))
      if (load.getHierarchy() == hierarchy)
        return load.getDpuProgram();

  upmem::LoadProgramOp unique;
  for (Operation *user : hierarchy.getUsers())
    if (auto load = dyn_cast<upmem::LoadProgramOp>(user)) {
      if (unique)
        return {};
      unique = load;
    }
  return unique ? unique.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::ScatterOnArrayOp::getDpuProgram() {
  return programLoadedOn(getHierarchy(), getOperation());
}

upmem::DpuProgramOp upmem::GatherFromArrayOp::getDpuProgram() {
  return programLoadedOn(getHierarchy(), getOperation());
}

upmem::DpuProgramOp upmem::ScatterBlocksOp::getDpuProgram() {
  return programLoadedOn(getHierarchy(), getOperation());
}

upmem::DpuProgramOp upmem::GatherBlocksOp::getDpuProgram() {
  return programLoadedOn(getHierarchy(), getOperation());
}

upmem::DpuProgramOp upmem::BroadcastOp::getDpuProgram() {
  return programLoadedOn(getHierarchy(), getOperation());
}

upmem::DpuProgramOp upmem::WaitForOp::getDpuProgram() {
  return programLoadedOn(getDpuSet(), getOperation());
}

MemRefType upmem::detail::flatMemRefType(Type ty) {
  MemRefType structured = llvm::cast<MemRefType>(ty);

  auto numBytes =
      structured.getNumElements() * structured.getElementTypeBitWidth() / 8;
  return MemRefType::get(
      {numBytes}, IntegerType::get(structured.getContext(), 8),
      MemRefLayoutAttrInterface{}, structured.getMemorySpace());
}
// parsers/printers

LogicalResult upmem::UPMEMDialect::verifyOperationAttribute(Operation *,
                                                            NamedAttribute) {
  return success();
}

void upmem::StaticAllocOp::build(OpBuilder &builder, OperationState &result,
                                 MemRefType ty, DpuMemSpace memSpace,
                                 StringRef name, bool noinit, bool zeroinit) {
  result.addAttribute(getMemSpaceAttrName(result.name),
                      builder.getAttr<DpuMemSpaceAttr>(memSpace));
  if (noinit)
    result.addAttribute(getNoinitAttrName(result.name), builder.getUnitAttr());
  if (zeroinit)
    result.addAttribute(getZeroinitAttrName(result.name),
                        builder.getUnitAttr());

  if (!name.empty()) {
    result.addAttribute(getSymNameAttrName(result.name),
                        builder.getStringAttr(name));
  }
  result.addTypes(ty);
}

/// Every transfer op moves `blockSize` elements at a time, starting at the
/// host index `map` computes for each point of `box` -- the DPU array, plus
/// the block index for the `_blocks` forms. Both the flat memcpy
/// (do_dpu_transfer) and the SDK's scatter-gather API (dpu_push_sg_xfer) take
/// one *address* and a length, so a block that runs over a gap in the host
/// memref's layout silently reads or writes unrelated data. This is the one
/// contract all five ops share, checked the same way for each.
///
/// A strided memref is a regular grid of contiguous runs of
/// `getContiguousSuffixSize` elements. Linearizing the map's trailing results
/// against that run gives where in it each block starts; the block fits iff
/// that offset plus its length still lies inside. The check declines (rather
/// than rejects) whenever a bound cannot be computed exactly.
static LogicalResult verifyTransferBlocks(Operation *op, MemRefType hostTy,
                                          AffineMap map, int64_t blockSize,
                                          ArrayRef<int64_t> box) {
  int64_t runSize = getContiguousSuffixSize(hostTy);
  if (runSize < 0)
    return success(); // dynamic or unsupported layout; nothing to check

  if (blockSize > runSize)
    return op->emitOpError("the number of transferred elements (")
           << blockSize << ") exceeds the largest contiguous run of elements ("
           << runSize << ") in host buffer " << hostTy
           << "; each transferred block must be contiguous in memory";

  ArrayRef<int64_t> shape = hostTy.getShape();
  MLIRContext *ctx = op->getContext();
  int64_t runRank = getContiguousSuffixRank(hostTy);
  if (runRank > 0) {
    AffineMap runLayout = AffineMap::get(
        runRank, 0, linearizeIndices(ctx, shape.take_back(runRank)), ctx);
    AffineMap trailing =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                       map.getResults().take_back(runRank), ctx);
    std::optional<int64_t> start =
        getAffineUpperBound(runLayout.compose(trailing).getResult(0), box);
    if (start && *start + blockSize > runSize)
      return op->emitOpError("a transferred block starts at offset ")
             << *start << " of a contiguous run of " << runSize
             << " elements in host buffer " << hostTy << " and is " << blockSize
             << " elements long, so it runs past the end of the run";
  }

  // The transfer must also stay inside the memref at all. The largest linear
  // offset any index reaches is the last element's.
  FailureOr<AffineExpr> offset = linearizeToElementOffset(map, hostTy);
  if (succeeded(offset)) {
    if (std::optional<int64_t> highest = getAffineUpperBound(*offset, box)) {
      SmallVector<int64_t> last(
          llvm::map_range(shape, [](int64_t extent) { return extent - 1; }));
      AffineMap lastIndex = AffineMap::get(
          0, 0,
          llvm::to_vector(llvm::map_range(
              last, [&](int64_t i) { return getAffineConstantExpr(i, ctx); })),
          ctx);
      FailureOr<AffineExpr> extent =
          linearizeToElementOffset(lastIndex, hostTy);
      if (succeeded(extent))
        if (auto constant = dyn_cast<AffineConstantExpr>(*extent))
          if (*highest + blockSize > constant.getValue() + 1)
            return op->emitOpError("a transferred block reaches element ")
                   << (*highest + blockSize - 1) << " of a host buffer "
                   << hostTy << " that only addresses "
                   << (constant.getValue() + 1);
    }
  }
  return success();
}

/// The (dpu) box of `hierarchy`. The tasklet dimension is deliberately
/// absent: a transfer targets a DPU's MRAM, which its tasklets share.
static SmallVector<int64_t> arrayBox(upmem::DeviceHierarchyType hierarchy) {
  return {hierarchy.getNumDpus()};
}

LogicalResult upmem::GatherFromArrayOp::verify() {
  if (getScatterMap().getNumResults() !=
          getHostBuffer().getType().getShape().size() ||
      getScatterMap().getNumDims() != 1)
    return emitOpError("Scatter map should map (dpu) to a start index in "
                       "the host buffer");
  return verifyTransferBlocks(*this, getHostBuffer().getType(), getScatterMap(),
                              getTransferCount(),
                              arrayBox(getHierarchy().getType()));
}

LogicalResult upmem::ScatterOnArrayOp::verify() {
  if (getScatterMap().getNumResults() !=
          getHostBuffer().getType().getShape().size() ||
      getScatterMap().getNumDims() != 1)
    return emitOpError("Scatter map should map (dpu) to a start index in "
                       "the host buffer");
  return verifyTransferBlocks(*this, getHostBuffer().getType(), getScatterMap(),
                              getTransferCount(),
                              arrayBox(getHierarchy().getType()));
}

/// Shared by both `_blocks` ops: same map arity, same box, and
/// `transferCount` is one block of the `numBlocksPerDpu` a DPU receives.
template <class Op> static LogicalResult verifyBlockTransfer(Op op) {
  if (op.getScatterMap().getNumResults() !=
          op.getHostBuffer().getType().getShape().size() ||
      op.getScatterMap().getNumDims() != 2)
    return op.emitOpError("Scatter map should map (dpu, block) to a "
                          "start index in the host buffer");
  if (op.getNumBlocksPerDpu() < 1)
    return op.emitOpError("must transfer at least one block per DPU");

  SmallVector<int64_t> box = arrayBox(op.getHierarchy().getType());
  box.push_back(op.getNumBlocksPerDpu());
  return verifyTransferBlocks(op, op.getHostBuffer().getType(),
                              op.getScatterMap(), op.getTransferCount(), box);
}

LogicalResult upmem::ScatterBlocksOp::verify() {
  return verifyBlockTransfer(*this);
}

LogicalResult upmem::GatherBlocksOp::verify() {
  return verifyBlockTransfer(*this);
}

LogicalResult upmem::BroadcastOp::verify() {
  MemRefType hostTy = getHostBuffer().getType();
  if (!hostTy.hasStaticShape())
    return emitOpError("host buffer must have a static shape");

  // One block, the whole buffer, at the origin -- so the map is the constant
  // zero index and the box is a single point.
  MLIRContext *ctx = getContext();
  SmallVector<AffineExpr> origin(hostTy.getRank(),
                                 getAffineConstantExpr(0, ctx));
  return verifyTransferBlocks(*this, hostTy, AffineMap::get(1, 0, origin, ctx),
                              hostTy.getNumElements(), /*box=*/{1});
}

/// Resolves `dpuBufRef` to the upmem.static_alloc it must name, in the
/// dpu_program loaded onto `hierarchy`. Returns a null StaticAllocOp (not a
/// failure) if `hierarchy` is a block argument and can't be resolved
/// statically; emits an error and returns failure if it resolves to
/// something else.
static FailureOr<upmem::StaticAllocOp>
resolveDpuBuffer(Operation *op, Value hierarchy, FlatSymbolRefAttr dpuBufRef,
                 SymbolTableCollection & /*symbolTable*/) {
  // Which program the set holds is flow-sensitive since the alloc/load
  // split (see programLoadedOn): when it cannot be resolved statically --
  // forwarded hierarchy with no local load, or several candidate loads --
  // skip the static check rather than guess.
  upmem::DpuProgramOp program = programLoadedOn(hierarchy, op);
  if (!program)
    return upmem::StaticAllocOp{};

  Operation *bufOp = SymbolTable::lookupSymbolIn(program, dpuBufRef);
  if (!bufOp)
    return op->emitOpError("buffer reference ")
           << dpuBufRef << " does not refer to any symbol in @"
           << program.getSymName();

  auto staticAlloc = dyn_cast<upmem::StaticAllocOp>(bufOp);
  if (!staticAlloc)
    return op->emitOpError("buffer reference ")
           << dpuBufRef << " must refer to a named upmem.static_alloc op";

  return staticAlloc;
}

static LogicalResult
verifyScatterGatherSymbolUses(Operation *op, Value hierarchy,
                              FlatSymbolRefAttr dpuBufRef,
                              SymbolTableCollection &symbolTable) {
  return resolveDpuBuffer(op, hierarchy, dpuBufRef, symbolTable);
}

/// Returns true if `a` and `b` describe the same dense buffer: the same
/// elements in the same order, so that a verbatim copy between them is right.
///
/// Not shape equality, because the two sides group the same run of elements
/// differently. A broadcast operand is one the workgroup shares along some
/// distributed dimension, and the host side keeps that dimension separate --
/// a 16-tasklet buffer whose operand is shared along an iteration dimension
/// split two ways arrives shaped (2, 8, ...) where the device buffer is
/// (16, ...). Row-major over (2, 8) and over (16) is the same sequence.
///
/// Consecutive dimensions may therefore be grouped on either side, and unit
/// dimensions ignored, but nothing may be reordered: (32, 16) and (16, 32)
/// hold the same elements in different orders and a verbatim copy between
/// them would move the right bytes to the wrong places.
static bool shapesCompatibleUpToUnitDims(ArrayRef<int64_t> a,
                                         ArrayRef<int64_t> b) {
  auto significantDims = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t> result;
    llvm::copy_if(shape, std::back_inserter(result),
                  [](int64_t d) { return d != 1; });
    return result;
  };
  SmallVector<int64_t> lhs = significantDims(a), rhs = significantDims(b);
  if (llvm::any_of(lhs, ShapedType::isDynamic) ||
      llvm::any_of(rhs, ShapedType::isDynamic))
    return true; // nothing to check against

  // One shape has to be reachable from the other by collapsing consecutive
  // dimensions -- one of them refines the other. Merely having a common
  // refinement is no test at all: collapsing both sides to a single dimension
  // always succeeds when the totals agree, which would accept a transpose.
  auto refines = [](ArrayRef<int64_t> fine, ArrayRef<int64_t> coarse) {
    size_t i = 0;
    for (int64_t group : coarse) {
      int64_t covered = 1;
      while (covered < group && i < fine.size())
        covered *= fine[i++];
      if (covered != group)
        return false;
    }
    return i == fine.size();
  };
  return refines(lhs, rhs) || refines(rhs, lhs);
}

LogicalResult
upmem::ScatterOnArrayOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyScatterGatherSymbolUses(*this, getHierarchy(),
                                       getDpuBufRefAttr(), symbolTable);
}

LogicalResult
upmem::GatherFromArrayOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyScatterGatherSymbolUses(*this, getHierarchy(),
                                       getDpuBufRefAttr(), symbolTable);
}

LogicalResult
upmem::ScatterBlocksOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyScatterGatherSymbolUses(*this, getHierarchy(),
                                       getDpuBufRefAttr(), symbolTable);
}

LogicalResult
upmem::GatherBlocksOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyScatterGatherSymbolUses(*this, getHierarchy(),
                                       getDpuBufRefAttr(), symbolTable);
}

LogicalResult
upmem::BroadcastOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto staticAllocOrFailure =
      resolveDpuBuffer(*this, getHierarchy(), getDpuBufRefAttr(), symbolTable);
  if (failed(staticAllocOrFailure))
    return failure();
  upmem::StaticAllocOp staticAlloc = *staticAllocOrFailure;
  if (!staticAlloc)
    return success(); // hierarchy is a block argument; can't verify statically

  if (!shapesCompatibleUpToUnitDims(getHostBuffer().getType().getShape(),
                                    staticAlloc.getType().getShape()))
    return emitOpError("host buffer shape ")
           << getHostBuffer().getType()
           << " is not compatible with target buffer " << staticAlloc.getType()
           << " (a broadcast copies the buffer verbatim, so one shape must be "
              "reachable from the other by collapsing consecutive dimensions "
              "and ignoring extent-1 ones)";

  return success();
}

::mlir::LogicalResult upmem::LoadProgramOp::verifySymbolUses(
    ::mlir::SymbolTableCollection &symbolTable) {

  upmem::DpuProgramOp program =
      symbolTable.lookupNearestSymbolFrom<upmem::DpuProgramOp>(
          *this, getDpuProgramRefAttr());

  if (!program)
    return emitOpError("requires ")
           << getDpuProgramRefAttr() << " to refer to an upmem.dpu_program op";
  if (program.getNumTasklets() !=
      getHierarchy().getType().getNumTaskletsPerDpu())
    return emitOpError("loads a program compiled for ")
           << program.getNumTasklets() << " tasklet(s) onto a hierarchy of "
           << getHierarchy().getType().getNumTaskletsPerDpu()
           << " tasklet(s) per DPU";
  return success();
}

void upmem::StaticAllocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn fn) {
  fn(getBuffer(), isWram() ? "wram_buf" : "mram_buf");
}

void upmem::StaticAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0),
                       SideEffects::DefaultResource::get());
  if (!getSymName())
    return;

  // A named allocation is the one thing the host can address on the device: it
  // scatters into and gathers from the symbol. That reference is a
  // SymbolRefAttr in another module, not an SSA use, so to DCE the buffer looks
  // like an allocation nobody reads -- exactly the shape it removes. The write
  // effect below states what is true of it and stops that.
  //
  // It deliberately names no value: `wouldOpBeTriviallyDead` drops any effect
  // that lands on a result the same op allocates, so a write attached to the
  // buffer would count for nothing. Unattached is also the more honest
  // reading -- the writer is the host, not this op.
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}
namespace {

struct FoldCastForLocalTransfer
    : public OpRewritePattern<upmem::LocalTransferOp> {
public:
  using OpRewritePattern<upmem::LocalTransferOp>::OpRewritePattern;

  static bool foldOperand(OpOperand &opnd, PatternRewriter &rewriter) {
    auto cast = opnd.get().getDefiningOp<memref::CastOp>();
    if (!cast)
      return false;

    if (!memref::CastOp::canFoldIntoConsumerOp(cast))
      return false;

    rewriter.modifyOpInPlace(opnd.getOwner(),
                             [&]() { opnd.set(cast.getSource()); });
    return true;
  }

  LogicalResult matchAndRewrite(upmem::LocalTransferOp op,
                                PatternRewriter &rewriter) const override {

    auto foldSource = foldOperand(op.getSourceMutable(), rewriter);
    auto foldDest = foldOperand(op.getTargetMutable(), rewriter);
    return success(foldSource || foldDest);
  }
};

} // namespace

LogicalResult upmem::LocalTransferOp::verify() {
  for (auto [side, ty] :
       {std::pair<StringRef, MemRefType>{"source", getSource().getType()},
        std::pair<StringRef, MemRefType>{"target", getTarget().getType()}}) {
    if (memrefIsContiguous(ty))
      continue;
    InFlightDiagnostic diag = emitOpError(side)
                              << " is not contiguous: " << ty
                              << ". A local transfer is a DMA of one run of "
                                 "memory; a strided region would move the "
                                 "right number of bytes to or from the wrong "
                                 "addresses";
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (succeeded(ty.getStridesAndOffset(strides, offset)))
      diag << " (strides " << strides << ")";
    return diag;
  }
  return success();
}

void upmem::LocalTransferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldCastForLocalTransfer>(context);
}

namespace {

// A transfer map is evaluated over the DPU index alone, or, for the block
// forms, over (dpu, block). Those extents are what may be assumed while
// simplifying it. The block extent is `numBlocksPerDpu`, which is a property
// of the transfer and not of the hierarchy: one tasklet's data may arrive as
// several blocks, so the tasklet count would be both wrong and, being
// smaller, wrong in the direction that silently discards the high bits of the
// block index.
template <class Op, bool HasBlockDim>
class SimplifyScatterMap : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = op.getScatterMap();
    SmallVector<int64_t> domain{op.getHierarchy().getType().getNumDpus()};
    if constexpr (HasBlockDim)
      domain.push_back(op.getNumBlocksPerDpu());
    if (map.getNumDims() != domain.size())
      return failure();

    AffineMap simplified = simplifyAffineMapWithBounds(map, domain);
    if (simplified == map)
      return failure();

    rewriter.modifyOpInPlace(op, [&] { op.setScatterMap(simplified); });
    return success();
  }
};
} // namespace

void upmem::ScatterBlocksOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<SimplifyScatterMap<ScatterBlocksOp, true>>(context);
}
void upmem::ScatterOnArrayOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<SimplifyScatterMap<ScatterOnArrayOp, false>>(context);
}
void upmem::GatherBlocksOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<SimplifyScatterMap<GatherBlocksOp, true>>(context);
}
void upmem::GatherFromArrayOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<SimplifyScatterMap<GatherFromArrayOp, false>>(context);
}
