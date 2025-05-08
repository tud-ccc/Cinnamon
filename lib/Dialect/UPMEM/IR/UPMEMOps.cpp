/// Implements the UPMEM dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"

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

MemRefType upmem::detail::flatMemRefType(Type ty) {
  MemRefType structured = llvm::cast<MemRefType>(ty);

  auto numBytes =
      structured.getNumElements() * structured.getElementTypeBitWidth() / 8;
  return MemRefType::get(
      {numBytes}, IntegerType::get(structured.getContext(), 8),
      MemRefLayoutAttrInterface{}, structured.getMemorySpace());
}
// parsers/printers

LogicalResult
upmem::UPMEMDialect::verifyOperationAttribute(Operation *op,
                                              NamedAttribute attr) {
  if (!llvm::isa<UnitAttr>(attr.getValue()) ||
      attr.getName() != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';
  return success();
}

void upmem::StaticAllocOp::build(OpBuilder &builder, OperationState &result,
                                 MemRefType ty, bool isWram, StringRef name,
                                 bool noinit) {
  if (isWram)
    result.addAttribute(getIsWramAttrName(result.name), builder.getUnitAttr());
  if (noinit)
    result.addAttribute(getNoinitAttrName(result.name), builder.getUnitAttr());

  if (!name.empty()) {
    result.addAttribute(getSymNameAttrName(result.name),
                        builder.getStringAttr(name));
  }
  result.addTypes({ty, detail::flatMemRefType(ty)});
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
  return success();
}

void upmem::PrivateWRAMAllocOp::getAsmResultNames(
    ::mlir::OpAsmSetValueNameFn fn) {
  fn(getBuffer(), "pwram_buf");
  fn(getBytes(), "bytes");
}

void upmem::StaticAllocOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn fn) {
  fn(getBuffer(), getIsWram() ? "wram_buf" : "mram_buf");
  fn(getBytes(), "bytes");
}