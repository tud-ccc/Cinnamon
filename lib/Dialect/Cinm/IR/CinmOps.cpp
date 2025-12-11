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
#include <llvm/Support/Casting.h>
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
#include <mlir/IR/OpDefinition.h>
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

::mlir::ParseResult ElementwiseOp::parse(::mlir::OpAsmParser &parser,
                                         ::mlir::OperationState &result) {
  std::string kindKw;
  if (parser.parseKeywordOrString(&kindKw))
    return failure();
  auto kind = symbolizeElementwiseKind(kindKw);
  if (!kind)
    return failure();
  result.addAttribute(getKindAttrName(result.name),
                      parser.getBuilder().getAttr<ElementwiseKindAttr>(*kind));

  bool hasOut = false, hasRhs = false;
  OpAsmParser::UnresolvedOperand lhs, rhs, out;
  if (parser.parseOperand(lhs))
    return failure();

  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseOperand(rhs))
      return failure();
    hasRhs = true;
  }
  if (parser.parseOptionalKeyword("into").succeeded()) {
    if (parser.parseOperand(out))
      return failure();
    hasOut = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  Type lhsAndRhsTy;
  Type outType;
  if (parser.parseType(lhsAndRhsTy))
    return failure();
  if (hasOut) {
    if (parser.parseKeyword("into") || parser.parseType(outType))
      return failure();
  }

  if (parser.resolveOperand(lhs, lhsAndRhsTy, result.operands))
    return failure();
  if (hasRhs && parser.resolveOperand(rhs, lhsAndRhsTy, result.operands))
    return failure();
  if (hasOut && parser.resolveOperand(out, outType, result.operands))
    return failure();

  if (!hasOut) {
    result.addTypes(lhsAndRhsTy);
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(hasRhs), static_cast<int32_t>(hasOut)}));

  return success();
}

void ElementwiseOp::print(::mlir::OpAsmPrinter &out) {
  out << " ";
  out.printKeywordOrString(stringifyElementwiseKind(getKind()));
  out << getLhs();
  if (getRhs()) {
    out << ", " << getRhs();
  }
  if (getOut()) {
    out << " into " << getOut();
  }
  out.printOptionalAttrDict((*this)->getAttrs(), {getKindAttrName(), getOperandSegmentSizesAttrName()});
  out << " : " << getLhs().getType();
  if (getOut()) {
    out << " into " << getOut().getType();
  }
}

::mlir::ParseResult parseGemmOp(::mlir::OpAsmParser &parser,
                                ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand lhs, rhs, bias, out;
  bool hasBias = false, hasOut = false;
  Type lhsType, rhsType, outType;

  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs))
    return failure();

  if (parser.parseOptionalKeyword("plus").succeeded()) {
    if (parser.parseOperand(bias))
      return failure();
    hasBias = true;
  }

  if (parser.parseOptionalKeyword("into").succeeded()) {
    if (parser.parseOperand(out))
      return failure();
    hasOut = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  if (parser.parseColon() || parser.parseType(lhsType) || parser.parseComma() ||
      parser.parseType(rhsType))
    return failure();

  if (hasOut) {
    if (parser.parseKeyword("into") || parser.parseType(outType))
      return failure();
  } else {
    if (parser.parseArrow() || parser.parseType(outType))
      return failure();
  }

  if (parser.resolveOperand(lhs, lhsType, result.operands).failed())
    return failure();
  if (parser.resolveOperand(rhs, rhsType, result.operands).failed())
    return failure();
  if (hasBias && parser.resolveOperand(bias, outType, result.operands).failed())
    return failure();
  if (hasOut && parser.resolveOperand(out, outType, result.operands).failed())
    return failure();

  if (dyn_cast<RankedTensorType>(outType)) {
    result.addTypes(outType);
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, 1, static_cast<int32_t>(hasBias), static_cast<int32_t>(hasOut)}));

  return success();
}

::mlir::ParseResult GemmOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  return parseGemmOp(parser, result);
}

template <typename Op> void printGemmLikeOp(OpAsmPrinter &out, Op op) {
  out << " " << op.getLhs() << ", " << op.getRhs();
  if (auto bias = op.getBias())
    out << " plus " << bias;
  Type outTy;
  bool useIntoKw;
  if (auto outBuf = op.getOut()) {
    outTy = outBuf.getType();
    out << " into " << outBuf;
    useIntoKw = true;
  } else {
    outTy = op.getResult().getType();
    useIntoKw = false;
  }
  out << " : " << op.getLhs().getType() << ", " << op.getRhs().getType();
  if (useIntoKw) {
    out << " into " << outTy;
  } else {
    out << " -> " << outTy;
  }
}

void GemmOp::print(::mlir::OpAsmPrinter &printer) {
  printGemmLikeOp<GemmOp>(printer, *this);
}

::mlir::ParseResult GemvOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  return parseGemmOp(parser, result);
}

void GemvOp::print(::mlir::OpAsmPrinter &printer) {
  printGemmLikeOp<GemvOp>(printer, *this);
}

::mlir::ParseResult parseUnaryOp(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand input, output;
  Type operandType;
  bool hasOutput = false;

  if (parser.parseOperand(input).failed()) {
    return failure();
  }
  if (parser.parseOptionalKeyword("into").succeeded()) {
    hasOutput = true;
    if (parser.parseOperand(output).failed()) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return failure();

  if (parser.parseColonType(operandType).failed())
    return failure();

  if (parser.resolveOperand(input, operandType, result.operands).failed())
    return failure();

  if (hasOutput) {
    if (parser.resolveOperand(output, operandType, result.operands).failed())
      return failure();
  } else {
    result.addTypes(operandType);
  }

  return success();
}

::mlir::ParseResult ActivateOp::parse(::mlir::OpAsmParser &parser,
                                      ::mlir::OperationState &result) {
  ActivationKindAttr kind;
  if (parser.parseAttribute(kind, "kind", result.attributes).failed())
    return failure();
  return parseUnaryOp(parser, result);
}

void ActivateOp::print(::mlir::OpAsmPrinter &printer) {}

::mlir::ParseResult QuantizeOp::parse(::mlir::OpAsmParser &parser,
                                      ::mlir::OperationState &result) {
  return parseUnaryOp(parser, result);
}

void QuantizeOp::print(::mlir::OpAsmPrinter &printer) {}

::mlir::ParseResult DequantizeOp::parse(::mlir::OpAsmParser &parser,
                                        ::mlir::OperationState &result) {
  return parseUnaryOp(parser, result);
}

void DequantizeOp::print(::mlir::OpAsmPrinter &printer) {}

::mlir::LogicalResult GemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> loc,
    GemmOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLhs().getType());
  ShapeAdaptor rhsShape(adaptor.getRhs().getType());

  if (lhsShape.getRank() == 2 && rhsShape.getRank() == 2 &&
      lhsShape.getDimSize(1) == rhsShape.getDimSize(0) &&
      lhsShape.getElementType() == rhsShape.getElementType()) {

    if (adaptor.getOut() && llvm::isa<MemRefType>(adaptor.getOut().getType())) {
      // This is the out buffer. Don't add any results.
      return success();
    }
    SmallVector<int64_t, 2> outShape;
    outShape.push_back(lhsShape.getDimSize(0));
    outShape.push_back(rhsShape.getDimSize(1));

    inferredReturnShapes.push_back(
        ShapedTypeComponents(outShape, lhsShape.getElementType()));
    return success();
  }
  return mlir::emitError(*loc, "operand types are not compatible: ")
         << adaptor.getLhs().getType() << " and " << adaptor.getRhs().getType();
}

::mlir::LogicalResult GemvOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> loc,
    GemvOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLhs().getType());
  ShapeAdaptor rhsShape(adaptor.getRhs().getType());

  if (lhsShape.getRank() == 2 && rhsShape.getRank() == 1 &&
      lhsShape.getDimSize(1) == rhsShape.getDimSize(0) &&
      lhsShape.getElementType() == rhsShape.getElementType()) {
    if (adaptor.getOut() && llvm::isa<MemRefType>(adaptor.getOut().getType())) {
      // This is the out buffer. Don't add any results.
      return success();
    }

    SmallVector<int64_t, 2> outShape;
    outShape.push_back(lhsShape.getDimSize(0));

    inferredReturnShapes.push_back(
        ShapedTypeComponents(outShape, lhsShape.getElementType()));
    return success();
  }
  return mlir::emitError(*loc, "operand types are not compatible: ")
         << adaptor.getLhs().getType() << " and " << adaptor.getRhs().getType();
}

::mlir::LogicalResult BatchGemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    BatchGemmOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getOperands()[0].getType());
  ShapeAdaptor rhsShape(adaptor.getOperands()[1].getType());

  if (lhsShape.getRank() != 3 || rhsShape.getRank() != 3)
    return failure();

  if (!dimsCompatible(lhsShape.getDimSize(0), rhsShape.getDimSize(0)) ||
      !dimsCompatible(lhsShape.getDimSize(2), rhsShape.getDimSize(1)))
    return failure();

  auto elementType = lhsShape.getElementType();
  if (rhsShape.getElementType() != elementType)
    return failure();

  SmallVector<int64_t, 3> outShape = {
      lhsShape.getDimSize(0), lhsShape.getDimSize(1), rhsShape.getDimSize(2)};

  if (Value bias = adaptor.getBias()) {
    ShapeAdaptor biasShape(bias.getType());
    if (biasShape.getRank() != 3 ||
        !dimsCompatible(biasShape.getDimSize(0), outShape[0]) ||
        !dimsCompatible(biasShape.getDimSize(1), outShape[1]) ||
        !dimsCompatible(biasShape.getDimSize(2), outShape[2]) ||
        biasShape.getElementType() != elementType)
      return failure();
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape, elementType));
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

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape, elementType));
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

  // This would imply the number of permutations does not match the rank of
  // the input which is illegal.
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

  if (!asCompute)
    return emitOpError() << "must be inside 'cinm.compute'";

  TypeRange expected = TypeRange(asCompute.getResultTypes());

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

} // namespace cinm
} // namespace mlir

// parsers/printers
