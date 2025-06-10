/// Implements the Cim dialect ops.
///
/// @file

#include <cinm-mlir/Dialect/Cim/IR/CimOps.h>
#include <cinm-mlir/Dialect/Cim/IR/CimTypes.h>
#include <cinm-mlir/Utils/CinmUtils.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdio>

#define DEBUG_TYPE "cim-ops"

using namespace mlir;
using namespace mlir::cim;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cim/IR/CimOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CimDialect
//===----------------------------------------------------------------------===//

void CimDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cim/IR/CimOps.cpp.inc"
      >();
}

void AcquireDeviceOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "cim_dev");
}

void AcquireCrossbarOp::getAsmResultNames(
    ::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "cim_cbr");
}

::mlir::LogicalResult GemmOp::verify() {
  auto lhs = cast<ShapedType>(getLhs().getType());
  auto rhs = cast<ShapedType>(getRhs().getType());
  auto result = cast<ShapedType>(getResult().getType());

  if (lhs.getElementType() != rhs.getElementType())
    return emitOpError("lhs and rhs must have the same element type");

  if (lhs.getElementType() != result.getElementType())
    return emitOpError("operands and result must have the same element type");

  if (lhs.getRank() != 2 || rhs.getRank() != 2)
    return emitOpError("lhs and rhs must be matrices (rank 2)");

  if (result.getRank() != 2)
    return emitOpError("result must be a matrix (rank 2)");

  if (lhs.getShape()[1] != rhs.getShape()[0] ||
      lhs.getShape()[0] != result.getShape()[0] ||
      rhs.getShape()[1] != result.getShape()[1])
    return emitOpError("operands and result must have compatible shapes");

  return success();
}

::mlir::LogicalResult GemvOp::verify() {
  auto lhs = cast<ShapedType>(getLhs().getType());
  auto rhs = cast<ShapedType>(getRhs().getType());
  auto result = cast<ShapedType>(getResult().getType());

  if (lhs.getElementType() != rhs.getElementType())
    return emitOpError("lhs and rhs must have the same element type");

  if (lhs.getElementType() != result.getElementType())
    return emitOpError("operands and result must have the same element type");

  if (lhs.getRank() != 2 || rhs.getRank() != 1)
    return emitOpError(
        "lhs must be a matrix (rank 2) and rhs must be a vector (rank 1)");

  if (result.getRank() != 1)
    return emitOpError("result must be a vector (rank 1)");

  if (lhs.getShape()[1] != rhs.getShape()[0] ||
      lhs.getShape()[0] != result.getShape()[0])
    return emitOpError("operands and result must have compatible shapes");

  return success();
}