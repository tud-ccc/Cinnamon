/// Implements the Cim dialect ops.
///
/// @file

#include <cinm-mlir/Dialect/Cim/IR/CimOps.h>
#include <cinm-mlir/Dialect/Cim/IR/CimTypes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static inline bool dimsEqualOrDynamic(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

static LogicalResult verifyFutureMatchesMemRef(Operation *op, MemRefType memTy,
                                               FutureType futureTy,
                                               StringRef memName,
                                               StringRef futureName) {
  ArrayRef<int64_t> futureShape = futureTy.getShape();
  if (static_cast<int64_t>(futureShape.size()) != memTy.getRank())
    return op->emitOpError()
           << "expects " << futureName << " to have rank " << memTy.getRank()
           << " like " << memName << ", but got " << futureShape.size();

  for (auto [idx, futDim] : llvm::enumerate(futureShape)) {
    int64_t memDim = memTy.getDimSize(static_cast<unsigned>(idx));
    if (!dimsEqualOrDynamic(memDim, futDim))
      return op->emitOpError()
             << "expects dimension " << idx << " of " << futureName << " ("
             << futDim << ") to match " << memName << " (" << memDim << ")";
  }

  return success();
}

::mlir::LogicalResult GemmOp::verify() {
  // Operands must be memrefs of rank 2.
  auto lhsTy = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsTy = dyn_cast<MemRefType>(getRhs().getType());
  if (!lhsTy || lhsTy.getRank() != 2)
    return emitOpError("lhs must be memref of rank 2");
  if (!rhsTy || rhsTy.getRank() != 2)
    return emitOpError("rhs must be memref of rank 2");

  // Element types must match across operands.
  if (lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("lhs and rhs must have the same element type");

  // K compatibility: dim1(lhs) == dim0(rhs) (or dynamic).
  if (!dimsEqualOrDynamic(lhsTy.getDimSize(1), rhsTy.getDimSize(0)))
    return emitOpError("incompatible K: dim1(lhs) must equal dim0(rhs)");

  // Result must be a cim.future (payload not inspected here).
  if (!isa<FutureType>(getResult().getType()))
    return emitOpError("result must be a !cim.future");

  return success();
}

::mlir::LogicalResult GemvOp::verify() {
  auto lhsTy = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsTy = dyn_cast<MemRefType>(getRhs().getType());
  if (!lhsTy || lhsTy.getRank() != 2)
    return emitOpError("lhs must be memref of rank 2");
  if (!rhsTy || rhsTy.getRank() != 1)
    return emitOpError("rhs must be memref of rank 1");

  if (lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("lhs and rhs must have the same element type");

  if (!isa<FutureType>(getResult().getType()))
    return emitOpError("result must be a !cim.future");

  return success();
}

::mlir::LogicalResult AddOp::verify() {
  auto lhsTy = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsTy = dyn_cast<MemRefType>(getRhs().getType());
  if (!lhsTy)
    return emitOpError("lhs must be a memref");
  if (!rhsTy)
    return emitOpError("rhs must be a memref");

  // Element types must match.
  if (lhsTy.getElementType() != rhsTy.getElementType())
    return emitOpError("lhs and rhs must have the same element type");

  // Ranks must match.
  if (lhsTy.getRank() != rhsTy.getRank())
    return emitOpError("lhs and rhs must have the same rank");

  // Shapes must match (dim-wise), allowing dynamics on either side.
  for (int64_t d = 0; d < lhsTy.getRank(); ++d) {
    if (!dimsEqualOrDynamic(lhsTy.getDimSize(d), rhsTy.getDimSize(d)))
      return emitOpError() << "incompatible shapes at dimension " << d
                           << ": got " << lhsTy.getDimSize(d) << " vs "
                           << rhsTy.getDimSize(d);
  }

  // Result must be a future (payload verification omitted in this build).
  if (!isa<FutureType>(getResult().getType()))
    return emitOpError("result must be a !cim.future");

  return success();
}

::llvm::LogicalResult QuantizeOp::verify() {
  auto srcTy = dyn_cast<MemRefType>(getSrc().getType());
  if (!srcTy)
    return emitOpError("expects memref operand 'src'");

  if (!isa<FloatType>(srcTy.getElementType()))
    return emitOpError("expects element type of 'src' to be floating point");

  auto futureTy = dyn_cast<FutureType>(getResult().getType());
  if (!futureTy)
    return emitOpError("result must be a !cim.future");

  auto futureElemTy = dyn_cast<IntegerType>(futureTy.getElementType());
  if (!futureElemTy || futureElemTy.getWidth() != 8)
    return emitOpError("expects future element type to be i8");

  if (failed(
          verifyFutureMatchesMemRef(*this, srcTy, futureTy, "src", "result")))
    return failure();

  return success();
}

::llvm::LogicalResult DequantizeOp::verify() {
  auto srcTy = dyn_cast<MemRefType>(getSrc().getType());
  if (!srcTy)
    return emitOpError("expects memref operand 'src'");

  auto srcElemTy = dyn_cast<IntegerType>(srcTy.getElementType());
  if (!srcElemTy || srcElemTy.getWidth() != 8)
    return emitOpError("expects element type of 'src' to be i8");

  auto futureTy = dyn_cast<FutureType>(getResult().getType());
  if (!futureTy)
    return emitOpError("result must be a !cim.future");

  if (!isa<FloatType>(futureTy.getElementType()))
    return emitOpError("expects future element type to be floating point");

  if (failed(
          verifyFutureMatchesMemRef(*this, srcTy, futureTy, "src", "result")))
    return failure();

  return success();
}

static llvm::LogicalResult verifyFutureOfSameMemRef(mlir::Operation *op,
                                                    mlir::MemRefType inTy,
                                                    mlir::Type resTy) {
  if (!llvm::isa<mlir::cim::FutureType>(resTy))
    return op->emitOpError("result must be !cim.future<...>");

  if (!llvm::isa<mlir::FloatType>(inTy.getElementType()))
    return op->emitOpError("input memref element type must be a floating type");

  return mlir::success();
}

llvm::LogicalResult mlir::cim::ActivateOp::verify() {
  auto inMR = llvm::dyn_cast<mlir::MemRefType>(getInput().getType());
  if (!inMR)
    return emitOpError("expects memref input");
  return verifyFutureOfSameMemRef(*this, inMR, getResult().getType());
}
