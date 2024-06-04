/// Implements the Cnm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cnm-ops"

using namespace mlir;
using namespace mlir::cnm;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CnmDialect
//===----------------------------------------------------------------------===//

void CnmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.cpp.inc"
      >();
}
// parsers/printers

LogicalResult LaunchOp::verify() {
  auto bodyArgs = getBody().getArguments();
  auto operands = getParams();
  if (bodyArgs.size() != operands.size())
    return emitOpError("expected ")
           << operands.size() << " arguments, got " << bodyArgs.size();

  for (auto [arg, operand] : llvm::zip(bodyArgs, operands)) {
    if (auto bufTy = operand.getType().dyn_cast<cnm::BufferType>()) {
      auto memrefTy = MemRefType::get(bufTy.getShape(), bufTy.getElementType());
      if (arg.getType() != memrefTy)
        return emitError("Mismatched type for launch argument, expected ")
               << memrefTy << ", got " << arg.getType();
    } else if (operand.getType().isIntOrIndexOrFloat()) {
      if (arg.getType() != operand.getType())
        return emitError("Mismatched type for launch argument, expected ")
               << arg.getType();
    } else {
      return emitError("Invalid type for argument ")
             << operand << ", expecting !cnm.buffer or scalar type";
    }
  }
  return success();
}

LogicalResult ScatterOp::verify() {
  auto tensorTy = getInput().getType();
  auto bufferTy = getBuffer().getType();
  auto map = getScatterMap();
  // The affine map maps every index in the input tensor to
  // a cnm index which is WG-element index, and element in the buffer.

  if (map.getNumInputs() != tensorTy.getShape().size()) {
    return emitError() << "Affine map inputs (" << map.getNumInputs()
                       << ") do not correspond to scattered tensor dimensions ("
                       << tensorTy.getShape().size() << ")";
  }

  if (map.getNumResults() !=
      bufferTy.getWorkgroupShape().size() + bufferTy.getShape().size())
    return emitError() << "Affine map results (" << map.getNumInputs()
                       << ") do not correspond to workgroup + buffer dims ("
                       << bufferTy.getWorkgroupShape().size() << " + "
                       << bufferTy.getShape().size() << ")";
  return success();
}

LogicalResult GatherOp::verify() {
  auto tensorTy = getOutput().getType();
  auto bufferTy = getBuffer().getType();
  auto map = getScatterMap();
  // The affine map maps every index in the input tensor to
  // a cnm index which is WG-element index, and element in the buffer.

  if (map.getNumResults() != tensorTy.getShape().size()) {
    return emitError() << "Affine map results (" << map.getNumInputs()
                       << ") do not correspond to scattered tensor dimensions ("
                       << tensorTy.getShape().size() << ")";
  }

  if (map.getNumInputs() !=
      bufferTy.getWorkgroupShape().size() + bufferTy.getShape().size())
    return emitError() << "Affine map inputs (" << map.getNumInputs()
                       << ") do not correspond to workgroup + buffer dims ("
                       << bufferTy.getWorkgroupShape().size() << " + "
                       << bufferTy.getShape().size() << ")";
  return success();
}