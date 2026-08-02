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

// AllocDPUsOp owns the symbol reference, so it does the real lookup.
upmem::DpuProgramOp upmem::AllocDPUsOp::getDpuProgram() {
  auto *sym =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getDpuProgramRef());
  return dyn_cast_or_null<upmem::DpuProgramOp>(sym);
}

// Every transfer op carries the hierarchy value produced by AllocDPUsOp.
upmem::DpuProgramOp upmem::ScatterOnArrayOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getHierarchy().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::GatherFromArrayOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getHierarchy().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::ScatterBlocksOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getHierarchy().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::GatherBlocksOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getHierarchy().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::BroadcastOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getHierarchy().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::WaitForOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getDpuSet().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
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
    AffineMap runLayout =
        AffineMap::get(runRank, 0,
                       linearizeIndices(ctx, shape.take_back(runRank)), ctx);
    AffineMap trailing =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                       map.getResults().take_back(runRank), ctx);
    std::optional<int64_t> start =
        getAffineUpperBound(runLayout.compose(trailing).getResult(0), box);
    if (start && *start + blockSize > runSize)
      return op->emitOpError("a transferred block starts at offset ")
             << *start << " of a contiguous run of " << runSize
             << " elements in host buffer " << hostTy << " and is "
             << blockSize
             << " elements long, so it runs past the end of the run";
  }

  // The transfer must also stay inside the memref at all. The largest linear
  // offset any index reaches is the last element's.
  FailureOr<AffineExpr> offset = linearizeToElementOffset(map, hostTy);
  if (succeeded(offset)) {
    if (std::optional<int64_t> highest = getAffineUpperBound(*offset, box)) {
      SmallVector<int64_t> last(llvm::map_range(
          shape, [](int64_t extent) { return extent - 1; }));
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

/// The (rank, dpu) box of `hierarchy`. The tasklet dimension is deliberately
/// absent: a transfer targets a DPU's MRAM, which its tasklets share.
static SmallVector<int64_t> arrayBox(upmem::DeviceHierarchyType hierarchy) {
  return {hierarchy.getNumRanks(), hierarchy.getNumDpusPerRank()};
}

LogicalResult upmem::GatherFromArrayOp::verify() {
  if (getScatterMap().getNumResults() !=
          getHostBuffer().getType().getShape().size() ||
      getScatterMap().getNumDims() != 2)
    return emitOpError("Scatter map should map (rank, dpu) to a start index in "
                       "the host buffer");
  return verifyTransferBlocks(*this, getHostBuffer().getType(), getScatterMap(),
                              getTransferCount(),
                              arrayBox(getHierarchy().getType()));
}

LogicalResult upmem::ScatterOnArrayOp::verify() {
  if (getScatterMap().getNumResults() !=
          getHostBuffer().getType().getShape().size() ||
      getScatterMap().getNumDims() != 2)
    return emitOpError("Scatter map should map (rank, dpu) to a start index in "
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
      op.getScatterMap().getNumDims() != 3)
    return op.emitOpError("Scatter map should map (rank, dpu, block) to a "
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
  return verifyTransferBlocks(*this, hostTy,
                              AffineMap::get(1, 0, origin, ctx),
                              hostTy.getNumElements(), /*box=*/{1});
}

/// Resolves `dpuBufRef` to the upmem.static_alloc it must name, in the
/// dpu_program loaded onto `hierarchy`. Returns a null StaticAllocOp (not a
/// failure) if `hierarchy` is a block argument and can't be resolved
/// statically; emits an error and returns failure if it resolves to
/// something else.
static FailureOr<upmem::StaticAllocOp>
resolveDpuBuffer(Operation *op, Value hierarchy, FlatSymbolRefAttr dpuBufRef,
                 SymbolTableCollection &symbolTable) {
  auto allocOp = hierarchy.getDefiningOp<upmem::AllocDPUsOp>();
  if (!allocOp)
    return upmem::StaticAllocOp{}; // hierarchy is a block argument; can't
                                   // verify statically

  auto program = symbolTable.lookupNearestSymbolFrom<upmem::DpuProgramOp>(
      op, allocOp.getDpuProgramRefAttr());
  if (!program)
    return op->emitOpError("cannot resolve dpu_program for the hierarchy");

  Operation *bufOp = SymbolTable::lookupSymbolIn(program, dpuBufRef);
  if (!bufOp)
    return op->emitOpError("buffer reference ")
           << dpuBufRef << " does not refer to any symbol in "
           << allocOp.getDpuProgramRefAttr();

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

/// Returns true if `a` and `b` are equal once dimensions of extent 1 are
/// dropped from each -- i.e. one is reachable from the other by only
/// inserting/removing unit dims (as memref.expand_shape/collapse_shape would).
static bool shapesCompatibleUpToUnitDims(ArrayRef<int64_t> a,
                                         ArrayRef<int64_t> b) {
  auto dropUnitDims = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t> result;
    llvm::copy_if(shape, std::back_inserter(result),
                 [](int64_t d) { return d != 1; });
    return result;
  };
  return dropUnitDims(a) == dropUnitDims(b);
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
           << " is not compatible with target buffer "
           << staticAlloc.getType()
           << " (shapes must be equal up to extent-1 dimensions)";

  return success();
}

::mlir::LogicalResult upmem::AllocDPUsOp::verifySymbolUses(
    ::mlir::SymbolTableCollection &symbolTable) {

  if (getDpuProgramRefAttr()) {
    upmem::DpuProgramOp program =
        symbolTable.lookupNearestSymbolFrom<upmem::DpuProgramOp>(
            *this, getDpuProgramRefAttr());

    if (!program)
      return emitOpError("requires ") << getDpuProgramRefAttr()
                                      << " to refer to an upmem.dpu_program op";
  }
  // TODO verify that tasklet count of the dpu_program matches the last item of
  // the hierarchy (result type)
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
void upmem::LocalTransferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldCastForLocalTransfer>(context);
}