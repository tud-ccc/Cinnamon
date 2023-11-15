/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;

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

::mlir::ParseResult MinOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  return MaxOp::parse(parser, result);
}

::mlir::ParseResult MaxOp::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType;
  if (parser.parseOperand(input) || parser.parseColonType(inputType))
    return failure();

  SmallVector<Value> typedInput;
  if (parser.resolveOperand(input, inputType, typedInput))
    return failure();

  result.addOperands(typedInput);

  if (auto shaped = dyn_cast<ShapedType>(inputType)) {
    result.addTypes({shaped.getElementType()});
    return success();
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a tensor type as operand");
  }
}

void printMinMaxOp(Value v, ::mlir::OpAsmPrinter &p) {
  p.printOperand(v);
  p << ": ";
  p.printType(v.getType());
}
void MaxOp::print(::mlir::OpAsmPrinter &p) { printMinMaxOp(getOperand(), p); }

void MinOp::print(::mlir::OpAsmPrinter &p) { printMinMaxOp(getOperand(), p); }

::mlir::LogicalResult SimSearchOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
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
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
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
    ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
    Adaptor adaptor,
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
} // namespace cinm
} // namespace mlir

// parsers/printers
