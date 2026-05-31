/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;
using linalg::UnaryFn;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmEnums.cpp.inc"

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"
      >();
}

namespace mlir {
namespace cinm {

cinm::ComputeOp getEnclosingComputeBlock(Operation *op) {
  Operation *parent = op;
  while ((parent = parent->getParentOp())) {
    if (auto parentCompute = dyn_cast<cinm::ComputeOp>(parent))
      return parentCompute;
  }

  assert(false && "CINM operator is not inside a cinm.compute block");
}

static bool dimsCompatible(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

::mlir::LogicalResult GemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    GemmOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLeft().getType());
  ShapeAdaptor rhsShape(adaptor.getRight().getType());

  if (lhsShape.getRank() == 2 && rhsShape.getRank() == 2 &&
      lhsShape.getDimSize(1) == rhsShape.getDimSize(0) &&
      lhsShape.getElementType() == rhsShape.getElementType()) {

    SmallVector<int64_t, 2> outShape;
    outShape.push_back(lhsShape.getDimSize(0));
    outShape.push_back(rhsShape.getDimSize(1));

    inferredReturnShapes.push_back(
        ShapedTypeComponents(outShape, lhsShape.getElementType()));
    return success();
  }
  return failure();
  //  return context->emitError("operand types are not compatible");
}

::mlir::LogicalResult BatchGemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    BatchGemmOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLeft().getType());
  ShapeAdaptor rhsShape(adaptor.getRight().getType());

  if (lhsShape.getRank() != 3 || rhsShape.getRank() != 3)
    return failure();

  if (!dimsCompatible(lhsShape.getDimSize(0), rhsShape.getDimSize(0)) ||
      !dimsCompatible(lhsShape.getDimSize(2), rhsShape.getDimSize(1)))
    return failure();

  auto elementType = lhsShape.getElementType();
  if (rhsShape.getElementType() != elementType)
    return failure();

  SmallVector<int64_t, 3> outShape = {lhsShape.getDimSize(0),
                                      lhsShape.getDimSize(1),
                                      rhsShape.getDimSize(2)};

  if (Value bias = adaptor.getBias()) {
    ShapeAdaptor biasShape(bias.getType());
    if (biasShape.getRank() != 3 ||
        !dimsCompatible(biasShape.getDimSize(0), outShape[0]) ||
        !dimsCompatible(biasShape.getDimSize(1), outShape[1]) ||
        !dimsCompatible(biasShape.getDimSize(2), outShape[2]) ||
        biasShape.getElementType() != elementType)
      return failure();
  }

  inferredReturnShapes.push_back(
      ShapedTypeComponents(outShape, elementType));
  return success();
}

::mlir::LogicalResult BatchGemvOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    BatchGemvOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLeft().getType());
  ShapeAdaptor rhsShape(adaptor.getRight().getType());

  if (lhsShape.getRank() != 3 || rhsShape.getRank() != 2)
    return failure();

  if (!dimsCompatible(lhsShape.getDimSize(0), rhsShape.getDimSize(0)) ||
      !dimsCompatible(lhsShape.getDimSize(2), rhsShape.getDimSize(1)))
    return failure();

  auto elementType = lhsShape.getElementType();
  if (rhsShape.getElementType() != elementType)
    return failure();

  SmallVector<int64_t, 2> outShape = {lhsShape.getDimSize(0),
                                      lhsShape.getDimSize(1)};

  if (Value bias = adaptor.getBias()) {
    ShapeAdaptor biasShape(bias.getType());
    if (biasShape.getRank() != 2 ||
        !dimsCompatible(biasShape.getDimSize(0), outShape[0]) ||
        !dimsCompatible(biasShape.getDimSize(1), outShape[1]) ||
        biasShape.getElementType() != elementType)
      return failure();
  }

  inferredReturnShapes.push_back(
      ShapedTypeComponents(outShape, elementType));
  return success();
}

::mlir::LogicalResult SimSearchOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getLeft().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

::mlir::LogicalResult TopKOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

// Copied from the TOSA codebase.
::mlir::LogicalResult TransposeOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  ShapeAdaptor permsShape(adaptor.getPerms().getType());

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of the
  // input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(permsShape.getDimSize(0), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputShape.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputShape.getRank(), inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
  // If the permuations are a constant we can directly determine the output
  // shape.
  DenseIntElementsAttr attr;
  if (matchPattern(adaptor.getPerms(), m_Constant(&attr)) &&
      attr.getType().getRank() == 1) {
    ShapeAdaptor permShape = attr;
    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult cinm::YieldOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  auto asCompute = dyn_cast_or_null<cinm::ComputeOp>(parent);
  auto asComputeMR = dyn_cast_or_null<cinm::ComputeMemRefOp>(parent);

  if (!asCompute && !asComputeMR)
    return emitOpError()
           << "must be inside 'cinm.compute' or 'cinm.compute_memref'";

  TypeRange expected = asCompute ? TypeRange(asCompute.getResultTypes())
                                 : TypeRange(asComputeMR.getResultTypes());

  if (getNumOperands() != expected.size())
    return emitOpError() << "has " << getNumOperands()
                         << " operand(s) but parent expects "
                         << expected.size();

  for (auto it : llvm::enumerate(expected)) {
    Type got = getOperand(it.index()).getType();
    if (got != it.value())
      return emitOpError() << "operand #" << it.index()
                           << " type mismatch: expected " << it.value()
                           << " but got " << got;
  }
  return success();
}

LogicalResult ActivateMemRefOp::verify() {
  auto inTy = dyn_cast<MemRefType>(getInput().getType());
  auto outTy = dyn_cast<MemRefType>(getOut().getType());
  if (!inTy || !outTy)
    return emitOpError("expects memref types for input and out");

  if (inTy.getElementType() != outTy.getElementType())
    return emitOpError("element types must match: ") << inTy << " vs " << outTy;

  if (inTy.getRank() != outTy.getRank())
    return emitOpError("ranks must match: ")
           << inTy.getRank() << " vs " << outTy.getRank();

  for (int i = 0, e = inTy.getRank(); i < e; ++i) {
    int64_t a = inTy.getDimSize(i), b = outTy.getDimSize(i);
    if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b)
      return emitOpError("static dims must match at dim ")
             << i << ": " << a << " vs " << b;
  }
  return success();
}

} // namespace cinm
} // namespace mlir

// parsers/printers
