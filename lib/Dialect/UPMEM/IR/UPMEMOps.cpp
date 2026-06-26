/// Implements the UPMEM dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
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

// ScatterOp/GatherOp both carry the hierarchy value produced by AllocDPUsOp.
upmem::DpuProgramOp upmem::ScatterOp::getDpuProgram() {
  auto alloc =
      dyn_cast_or_null<upmem::AllocDPUsOp>(getHierarchy().getDefiningOp());
  return alloc ? alloc.getDpuProgram() : upmem::DpuProgramOp{};
}

upmem::DpuProgramOp upmem::GatherOp::getDpuProgram() {
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

LogicalResult upmem::GatherOp::verify() {
  if (getScatterMap().getNumResults() !=
          getHostBuffer().getType().getShape().size() ||
      getScatterMap().getNumDims() != 2)
    return emitOpError("Scatter map should map (rank, dpu) to a start index in "
                       "the host buffer");
  // auto count = getDpuMemOffset();
  // if ((count % 8) != 0)
  //   return emitOpError("has unaligned DPU memory offset ")
  //          << count << ", needs to be 8-byte-aligned.";
  return success();
}

LogicalResult upmem::ScatterOp::verify() {
  if (getScatterMap().getNumResults() !=
          getHostBuffer().getType().getShape().size() ||
      getScatterMap().getNumDims() != 2)
    return emitOpError("Scatter map should map (rank, dpu) to a start index in "
                       "the host buffer");

  // auto count = getDpuMemOffset();
  // if ((count % 8) != 0)
  // return emitOpError("has unaligned DPU memory offset ")
  //        << count << ", needs to be 8-byte-aligned.";
  return success();
}

static LogicalResult
verifyScatterGatherSymbolUses(Operation *op, Value hierarchy,
                              FlatSymbolRefAttr dpuBufRef,
                              SymbolTableCollection &symbolTable) {
  auto allocOp = hierarchy.getDefiningOp<upmem::AllocDPUsOp>();
  if (!allocOp)
    return success(); // hierarchy is a block argument; can't verify statically

  auto program = symbolTable.lookupNearestSymbolFrom<upmem::DpuProgramOp>(
      op, allocOp.getDpuProgramRefAttr());
  if (!program)
    return op->emitOpError("cannot resolve dpu_program for the hierarchy");

  Operation *bufOp = SymbolTable::lookupSymbolIn(program, dpuBufRef);
  if (!bufOp)
    return op->emitOpError("buffer reference ")
           << dpuBufRef << " does not refer to any symbol in "
           << allocOp.getDpuProgramRefAttr();

  if (!isa<upmem::StaticAllocOp>(bufOp))
    return op->emitOpError("buffer reference ")
           << dpuBufRef << " must refer to a named upmem.static_alloc op";

  return success();
}

LogicalResult
upmem::ScatterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyScatterGatherSymbolUses(*this, getHierarchy(),
                                       getDpuBufRefAttr(), symbolTable);
}

LogicalResult
upmem::GatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyScatterGatherSymbolUses(*this, getHierarchy(),
                                       getDpuBufRefAttr(), symbolTable);
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
void upmem::PrivateWRAMAllocOp::getAsmResultNames(
    ::mlir::OpAsmSetValueNameFn fn) {
  fn(getBuffer(), "pwram_buf");
}

void upmem::StaticAllocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn fn) {
  fn(getBuffer(), isWram() ? "wram_buf" : "mram_buf");
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