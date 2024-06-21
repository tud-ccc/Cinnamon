/// Implements the Cnm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
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

::mlir::LogicalResult GatherOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    GatherOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  auto out = adaptor.getOutputBuf();
  if (out.getType().isa<MemRefType>()) {
    return success();
  } else if (out.getType().isa<RankedTensorType>()) {
    ShapedType ty = out.getType().cast<RankedTensorType>();

    inferredReturnShapes.push_back(ShapedTypeComponents(ty));
    return success();
  }

  return failure();
}

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

/// Check that the memref is contiguous in the dimensions corresponding to the bufShape, which
/// is a suffix of the shape of the input tensor/memref.
static bool scatteredMemrefIsContiguous(TypedValue<ShapedType> value, llvm::ArrayRef<int64_t> bufShape) {
  if (value.getType().isa<MemRefType>()) {
    auto type = value.getType().cast<MemRefType>();
    if (!type.hasStaticShape())
      return false;

    SmallVector<int64_t> strides;
    int64_t offset;// offset may be dynamic, we don't
    if (failed(getStridesAndOffset(type, strides, offset)))
      return false;

    // MemRef is contiguous if outer dimensions are size-1 and inner
    // dimensions have unit strides.
    int64_t runningStride = 1;
    int64_t curDim = strides.size() - 1;
    int64_t lastDimToCheck = strides.size() - bufShape.size();
    // Finds all inner dimensions with unit strides.
    while (curDim >= lastDimToCheck && strides[curDim] == runningStride) {
      runningStride *= type.getDimSize(curDim);
      --curDim;
    }

    // Check if other dimensions are size-1.
    while (curDim >= lastDimToCheck && type.getDimSize(curDim) == 1) {
      --curDim;
    }

    // All dims are unit-strided or size-1.
    return curDim < lastDimToCheck;
  }
  return true;
}

LogicalResult ScatterOp::verify() {
  auto tensorTy = getInput().getType();
  auto bufferTy = getBuffer().getType();
  auto map = getScatterMap();
  // The affine map maps every WG element to a prefix of the input tensor which
  // has buffer shape

  if (map.getNumInputs() != bufferTy.getWorkgroupShape().size()) {
    return emitError() << "Affine map inputs (" << map.getNumInputs()
                       << " dims) do not correspond to workgroup dimensions ("
                       << bufferTy.getWorkgroupShape().size() << " dims)";
  }

  auto truncatedDims = tensorTy.getShape().size() - bufferTy.getShape().size();
  if (map.getNumResults() != truncatedDims) {
    return emitError()
           << "Affine map results (" << map.getNumResults()
           << ") do not correspond to truncated scattered tensor dimensions ("
           << tensorTy.getShape().size() << " - " << bufferTy.getShape().size()
           << ")";
  }

  if (tensorTy.getShape().slice(truncatedDims) != bufferTy.getShape()) {
    return emitError()
           << "Scattered tensor shape should end with buffer shape, ("
           << tensorTy.getShape().slice(truncatedDims)
           << " != " << bufferTy.getShape() << ")";
  }

  if (!scatteredMemrefIsContiguous(getInput(), bufferTy.getShape())) {
    return emitOpError("should scatter a contiguous memref");
  }

  return success();
}

LogicalResult GatherOp::verify() {
  auto tensorTy = getOutputBuf().getType();
  auto bufferTy = getBuffer().getType();
  auto map = getGatherMap();
  // The affine map maps every WG-element index and buffer element index
  // to a result tensor index

  if (map.getNumInputs() != bufferTy.getWorkgroupShape().size()) {
    return emitError() << "Affine map inputs (" << map.getNumInputs()
                       << " dims) do not correspond to workgroup dimensions ("
                       << bufferTy.getWorkgroupShape().size() << " dims)";
  }

  auto truncatedDims = tensorTy.getShape().size() - bufferTy.getShape().size();
  if (map.getNumResults() != truncatedDims) {
    return emitError()
           << "Affine map results (" << map.getNumResults()
           << ") do not correspond to truncated scattered tensor dimensions ("
           << tensorTy.getShape().size() << " - " << bufferTy.getShape().size()
           << ")";
  }

  if (tensorTy.getShape().slice(truncatedDims) != bufferTy.getShape()) {
    return emitError()
           << "Scattered tensor shape should end with buffer shape, ("
           << tensorTy.getShape().slice(truncatedDims)
           << " != " << bufferTy.getShape() << ")";
  }

  if (!scatteredMemrefIsContiguous(getOutputBuf(), bufferTy.getShape())) {
    return emitOpError("should gather into a contiguous memref");
  }

  return success();
}
