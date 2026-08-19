/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
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
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;
using linalg::UnaryFn;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmEnums.cpp.inc"
#include "cinm-mlir/Dialect/Cinm/IR/CinmGemmlikeOpInterface.cpp.inc"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.cpp.inc"

template <typename Self>
static void buildGemmLikeOp(OpBuilder &, OperationState &result, Value lhs,
                            Value rhs, Value bias, Value out) {
  result.addOperands({lhs, rhs});
  int biasInt = 0, outInt = 0;
  if (bias) {
    result.addOperands(bias);
    biasInt = 1;
  }
  if (out) {
    result.addOperands(out);
    outInt = 1;
  }

  typename Self::Properties &properties =
      result.getOrAddProperties<typename Self::Properties>();
  properties.setOperandSegmentSizes({1, 1, biasInt, outInt});

  if (!out || isa<RankedTensorType>(out.getType())) {
    ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
    if (::mlir::succeeded(Self::inferReturnTypes(
            result.getContext(), result.location, result.operands,
            result.attributes.getDictionary(result.getContext()),
            result.getRawProperties(), result.regions, inferredReturnTypes))) {
      assert(inferredReturnTypes.size() == 1u &&
             "mismatched number of return types");
      result.addTypes(inferredReturnTypes);
    } else {
      ::llvm::report_fatal_error("Failed to infer result type(s).");
    }
  }
}

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

cinm::ComputeOpInterface getEnclosingComputeBlock(Operation *op) {
  Operation *parent = op;
  while ((parent = parent->getParentOp())) {
    if (auto parentCompute = dyn_cast<cinm::ComputeOpInterface>(parent))
      return parentCompute;
  }

  return {};
}

static bool dimsCompatible(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

::mlir::ParseResult ElementwiseOp::parse(::mlir::OpAsmParser &parser,
                                         ::mlir::OperationState &result) {
  std::string kindKw;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeywordOrString(&kindKw))
    return failure();
  auto kind = symbolizeElementwiseKind(kindKw);
  if (!kind)
    return parser.emitError(loc, "Unknown operator kind");

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

  if (!hasOut || isa<TensorType>(outType)) {
    result.addTypes(lhsAndRhsTy);
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(hasRhs), static_cast<int32_t>(hasOut)}));

  return success();
}

static ParseResult parsePlatformOrAccelerator(OpAsmParser &parser,
                                              OperationState &result,
                                              StringRef platformAttrName,
                                              StringRef acceleratorAttrName) {
  if (parser.parseOptionalKeyword("on").succeeded()) {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword("platform").succeeded()) {
      CinmPlatformAttrInterface platform;
      if (parser.parseAttribute(platform))
        return failure();
      result.addAttribute(platformAttrName, platform);
      return success();
    } else if (parser.parseOptionalKeyword("accelerator").succeeded()) {
      CinmAcceleratorAttrInterface accelerator;
      if (parser.parseAttribute(accelerator))
        return failure();
      result.addAttribute(acceleratorAttrName, accelerator);
      return success();
    }
    return parser.emitError(loc,
                            "Expected `platform` or `accelerator` keyword");
  }
  return success();
}

ParseResult ComputeBlockOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  if (parsePlatformOrAccelerator(parser, result,
                                 getPlatformAttrName(result.name),
                                 getAcceleratorAttrName(result.name)))
    return failure();

  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
        OpAsmParser::UnresolvedOperand op;
        auto &arg = regionArgs.emplace_back();
        if (parser.parseArgument(arg) || parser.parseEqual() ||
            parser.parseOperand(op) || parser.parseColonType(arg.type) ||
            parser.resolveOperand(op, arg.type, result.operands)) {
          return failure();
        }
        return success();
      })) {
    return failure();
  }

  if (parser.parseOptionalArrow().succeeded()) {
    if (parser.parseTypeList(result.types))
      return failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // if (parser.parseOptionalArrowTypeList(result.types))
  //   return failure();
  auto *region = result.addRegion();
  if (parser.parseRegion(*region, regionArgs, true))
    return failure();

  return success();
}

void ComputeBlockOp::print(OpAsmPrinter &out) {
  if (auto platform = getPlatform()) {
    out << " on platform " << platform;
  } else if (auto accelerator = getAccelerator()) {
    out << " on accelerator " << accelerator;
  }
  out << " (";
  llvm::interleaveComma(zipArgsWithOperands(), out, [&](auto pair) {
    auto [arg, value] = pair;
    out.printRegionArgument(arg, {}, true);
    out << " = " << value << " : " << value.getType();
  });
  out << ")";
  if (!getResults().empty()) {
    out << " -> ";
    llvm::interleaveComma(getResultTypes(), out);
  }
  out.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getPlatformAttrName(), getAcceleratorAttrName()});
  out << ' ';
  out.printRegion(getRegion(), false);
}

ParseResult ComputeOp::parse(::mlir::OpAsmParser &parser,
                             ::mlir::OperationState &result) {
  if (parsePlatformOrAccelerator(parser, result,
                                 getPlatformAttrName(result.name),
                                 getAcceleratorAttrName(result.name)))
    return failure();

  if (parser.parseOptionalArrow().succeeded()) {
    if (parser.parseTypeList(result.types))
      return failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // if (parser.parseOptionalArrowTypeList(result.types))
  //   return failure();
  auto *region = result.addRegion();
  if (parser.parseRegion(*region, {}))
    return failure();

  return success();
}

void ComputeOp::print(OpAsmPrinter &out) {
  if (auto platform = getPlatform()) {
    out << " on platform " << platform;
  } else if (auto accelerator = getAccelerator()) {
    out << " on accelerator " << accelerator;
  }
  if (!getResults().empty()) {
    out << " -> ";
    llvm::interleaveComma(getResultTypes(), out);
  }
  out.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getPlatformAttrName(), getAcceleratorAttrName()});
  out << ' ';
  out.printRegion(getRegion(), false);
}

LogicalResult ComputeOp::verify() {
  if (getPlatform() && getAccelerator())
    return emitOpError("Cannot specify both platform and accelerator");
  return success();
}
LogicalResult ComputeBlockOp::verify() {
  if (getPlatform() && getAccelerator())
    return emitOpError("Cannot specify both platform and accelerator");
  return success();
}
LogicalResult ReduceOp::verify() {
  uint64_t maxDim = getInput().getType().getRank();
  if (getDimension() >= maxDim)
    return emitOpError("Reduce op dimension should be within [0, ")
           << maxDim << ")";

  if (!getOut())
    return success();

  // Memref (destination-passing) mode.
  auto inputTy = getInput().getType();
  auto outTy = cast<ShapedType>(getOut().getType());
  if (!isa<MemRefType>(inputTy) || !isa<MemRefType>(outTy))
    return emitOpError("`into` output buffer is only supported in memref mode, "
                       "where the input is a memref too");
  if (getResult())
    return emitOpError("memref mode does not produce a result");
  if (outTy.getElementType() != inputTy.getElementType())
    return emitOpError("output buffer element type ")
           << outTy.getElementType() << " does not match input element type "
           << inputTy.getElementType();

  SmallVector<int64_t> expected(inputTy.getShape());
  expected.erase(expected.begin() + getDimension());
  if (outTy.getShape() != ArrayRef<int64_t>(expected))
    return emitOpError("output buffer shape ")
           << outTy.getShape() << " does not match the shape obtained by "
           << "reducing dimension " << getDimension() << " of the input";
  return success();
}

::mlir::ParseResult ReduceOp::parse(::mlir::OpAsmParser &parser,
                                    ::mlir::OperationState &result) {
  // $method `(` $input `)` (`dim` $dimension^ )? attr-dict
  //     `:` type($input) `->` type($result)
  std::string methodKw;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeywordOrString(&methodKw))
    return failure();
  auto method = symbolizeReduceMethod(methodKw);
  if (!method)
    return parser.emitError(loc, "Unknown reduce method");

  OpAsmParser::UnresolvedOperand input;
  if (parser.parseLParen() || parser.parseOperand(input) ||
      parser.parseRParen())
    return failure();

  // default to last dim
  int64_t dimension = -1;
  if (parser.parseOptionalKeyword("dim").succeeded()) {
    if (parser.parseInteger(dimension))
      return failure();
  }

  // Memref mode: `into $out ... : type($input) into type($out)`.
  OpAsmParser::UnresolvedOperand outBuf;
  bool hasOut = parser.parseOptionalKeyword("into").succeeded();
  if (hasOut && parser.parseOperand(outBuf))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  Type inputType, otherType;
  if (parser.parseType(inputType))
    return failure();
  if (hasOut) {
    if (parser.parseKeyword("into") || parser.parseType(otherType))
      return failure();
  } else if (parser.parseArrow() || parser.parseType(otherType)) {
    return failure();
  }

  SmallVector<Value, 2> resolved;
  if (parser.resolveOperand(input, inputType, resolved))
    return failure();

  OpBuilder b(parser.getContext());
  if (hasOut) {
    if (parser.resolveOperand(outBuf, otherType, resolved))
      return failure();
    build(b, result, *method, resolved[0], resolved[1], dimension);
  } else {
    build(b, result, otherType, *method, resolved[0], dimension);
  }
  return success();
}

void ReduceOp::print(::mlir::OpAsmPrinter &out) {
  out << ' ' << stringifyReduceMethod(getMethod());
  out << '(' << getInput() << ')';
  auto dim = getDimension();
  if (dim != getInput().getType().getShape().size() - 1)
    out << " dim " << dim;
  if (getOut())
    out << " into " << getOut();

  out.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{getMethodAttrName(),
                                             getDimensionAttrName(),
                                             getRankReduceAttrName()});
  out << " : " << getInput().getType();
  if (getOut())
    out << " into " << getOut().getType();
  else
    out << " -> " << getResult().getType();
}

void ReduceOp::build(OpBuilder &builder, OperationState &state, Type resultTy,
                     ReduceMethod kind, Value input, int64_t dimension) {
  state.addTypes(resultTy);
  state.addOperands(input);
  state.addAttribute(getMethodAttrName(state.name),
                     builder.getAttr<ReduceMethodAttr>(kind));
  bool rankReduce = true;
  if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(input.getType());
      dimension < 0) {
    auto newDim = dimension + shaped.getRank();
    if (newDim >= 0 && newDim < shaped.getRank())
      dimension = newDim;
    if (shaped.getRank() == 1 && isa<ShapedType>(resultTy))
      rankReduce = false;
  }

  state.addAttribute(getRankReduceAttrName(state.name),
                     builder.getBoolAttr(rankReduce));
  state.addAttribute(getDimensionAttrName(state.name),
                     builder.getI64IntegerAttr(dimension));
}

void ReduceOp::build(OpBuilder &builder, OperationState &state,
                     ReduceMethod kind, Value input, Value out,
                     int64_t dimension) {
  state.addOperands({input, out});
  state.addAttribute(getMethodAttrName(state.name),
                     builder.getAttr<ReduceMethodAttr>(kind));
  if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(input.getType());
      shaped && dimension < 0)
    dimension += shaped.getRank();

  // Irrelevant in memref mode: the result shape is `out`'s.
  state.addAttribute(getRankReduceAttrName(state.name),
                     builder.getBoolAttr(true));
  state.addAttribute(getDimensionAttrName(state.name),
                     builder.getI64IntegerAttr(dimension));
}

::llvm::LogicalResult ReduceOp::inferReturnTypes(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr,
    ::mlir::PropertyRef properties, ::mlir::RegionRange,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {

  // Memref mode accumulates into its `out` operand and yields nothing.
  if (operands.size() > 1)
    return success();

  auto inputTy = cast<ShapedType>(operands[0].getType());
  const Properties *props = properties.as<Properties *>();
  auto dimension = props->dimension.getInt();
  if (dimension < 0)
    dimension += inputTy.getRank();
  if (dimension < 0 || dimension >= inputTy.getRank())
    return failure();

  SmallVector<int64_t> resultShape(inputTy.getShape());
  resultShape.erase(resultShape.begin() + dimension);
  if (resultShape.size() > 0 || !props->rankReduce)
    inferredReturnTypes.push_back(
        inputTy.cloneWith(resultShape, inputTy.getElementType()));
  else
    inferredReturnTypes.push_back(inputTy.getElementType());
  return success();
}

void ElementwiseOp::print(::mlir::OpAsmPrinter &out) {
  out << " ";
  out.printKeywordOrString(stringifyElementwiseKind(getKind()));
  out << " ";
  out << getLhs();
  if (getRhs()) {
    out << ", " << getRhs();
  }
  if (getOut()) {
    out << " into " << getOut();
  }
  out.printOptionalAttrDict(
      (*this)->getAttrs(),
      {getKindAttrName(), getOperandSegmentSizesAttrName()});
  out << " : " << getLhs().getType();
  if (getOut()) {
    out << " into " << getOut().getType();
  }
}

void ElementwiseOp::build(OpBuilder &builder, OperationState &state,
                          ElementwiseKind kind, Value a, Value b, Value out) {
  state.addOperands(a);
  int bInt = 0, outInt = 0;
  if (b) {
    state.addOperands(b);
    bInt = 1;
  }
  if (out) {
    state.addOperands(out);
    outInt = 1;
    if (isa<TensorType>(out.getType())) {
      state.addTypes(out.getType());
    }
  } else {
    state.addTypes(a.getType());
  }

  state.addAttribute(getKindAttrName(state.name),
                     builder.getAttr<ElementwiseKindAttr>(kind));
  state.addAttribute(getOperandSegmentSizesAttrName(state.name),
                     builder.getDenseI32ArrayAttr({1, bInt, outInt}));
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

::mlir::LogicalResult GemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location> loc,
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
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location> loc,
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
  ShapeAdaptor lhsShape(adaptor.getLhs().getType());
  ShapeAdaptor rhsShape(adaptor.getRhs().getType());

  if (lhsShape.getRank() != 3 || rhsShape.getRank() != 3)
    return failure();

  if (!dimsCompatible(lhsShape.getDimSize(0), rhsShape.getDimSize(0)) ||
      !dimsCompatible(lhsShape.getDimSize(2), rhsShape.getDimSize(1)))
    return failure();

  auto elementType = lhsShape.getElementType();
  if (rhsShape.getElementType() != elementType)
    return failure();

  if (adaptor.getOut() && llvm::isa<MemRefType>(adaptor.getOut().getType())) {
    // This is the out buffer. Don't add any results.
    return success();
  }

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
  ShapeAdaptor lhsShape(adaptor.getLhs().getType());
  ShapeAdaptor rhsShape(adaptor.getRhs().getType());

  if (lhsShape.getRank() != 3 || rhsShape.getRank() != 2)
    return failure();

  if (!dimsCompatible(lhsShape.getDimSize(0), rhsShape.getDimSize(0)) ||
      !dimsCompatible(lhsShape.getDimSize(2), rhsShape.getDimSize(1)))
    return failure();

  auto elementType = lhsShape.getElementType();
  if (rhsShape.getElementType() != elementType)
    return failure();

  if (adaptor.getOut() && llvm::isa<MemRefType>(adaptor.getOut().getType())) {
    // This is the out buffer. Don't add any results.
    return success();
  }

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
  auto perms = adaptor.getPermutation();

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of
  // the input which is illegal.
  if (static_cast<int64_t>(perms.size()) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(perms.size(), ShapedType::kDynamic);
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

  // Since the permuations are a constant we can directly determine the output
  // shape.
  outputShape.reserve(inputShape.getRank());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    outputShape[i] = inputShape.getDimSize(perms[i]);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult cinm::YieldOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  auto asCompute = dyn_cast_or_null<cinm::ComputeBlockOp>(parent);
  auto asFlexCompute = dyn_cast_or_null<cinm::ComputeOp>(parent);
  auto asSelect = dyn_cast_or_null<cinm::SelectOp>(parent);

  if (!asCompute && !asSelect && !asFlexCompute)
    return emitOpError() << "must be inside 'cinm.compute_block', "
                            "'cinm.compute' or 'cinm.select'";

  TypeRange expected = TypeRange(parent->getResultTypes());

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

LogicalResult AcceleratorOp::verify() {
  // verify that they are all at the start of a block
  // auto *prevOp = (*this)->getPrevNode();
  // if (prevOp && !llvm::isa<AcceleratorOp>(prevOp)) {
  //   return emitOpError("should be declared at the start of a block");
  // }
  // if (!prevOp) {
  //   if (!llvm::isa<TfSchedulableBlockOp>((*this)->getParentOp()))
  //     return emitOpError("should be declared at the start of a block");
  // }
  return llvm::success();
}

template <class RW>
static void addEffect(
    OpOperand &operand,
    ::llvm::SmallVectorImpl<
        SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(RW::get(), &operand, 0, true,
                       SideEffects::DefaultResource::get());
}

template <class GemmLikeOp>
static void getGemmLikeEffects(
    GemmLikeOp op,
    ::llvm::SmallVectorImpl<
        SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

  if (op.getResult()) {
    // tensor variant, no effect at all
    return;
  }
  for (auto &opoperand : op->getOpOperands()) {
    // read all operands (even out buf)
    addEffect<MemoryEffects::Read>(opoperand, effects);
  }

  // write out buf
  auto &out = op.getOutMutable()[0];
  addEffect<MemoryEffects::Write>(out, effects);
}

void GemmOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGemmLikeEffects(*this, effects);
}
void GemvOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGemmLikeEffects(*this, effects);
}
void BatchGemmOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGemmLikeEffects(*this, effects);
}
void BatchGemvOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGemmLikeEffects(*this, effects);
}

void ReduceOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!getOut()) {
    // tensor variant, no effects
    return;
  }
  addEffect<MemoryEffects::Read>(getInputMutable(), effects);
  // The reduction accumulates into the out buffer, so it reads it too.
  addEffect<MemoryEffects::Read>(getOutMutable()[0], effects);
  addEffect<MemoryEffects::Write>(getOutMutable()[0], effects);
}

void ElementwiseOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getResult()) {
    // tensor variant, no effects
    return;
  }
  addEffect<MemoryEffects::Read>(getLhsMutable(), effects);
  if (getRhs())
    addEffect<MemoryEffects::Read>(getRhsMutable()[0], effects);

  addEffect<MemoryEffects::Write>(getOutMutable()[0], effects);
}

void ComputeBlockOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<mlir::InvocationBounds> &result) {

  result.push_back(::mlir::InvocationBounds(1, 1));
}

::mlir::OperandRange
ComputeBlockOp::getEntrySuccessorOperands(::mlir::RegionSuccessor) {
  return getOperands();
}

void ComputeBlockOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
  } else {
    // region is body
    regions.push_back(RegionSuccessor::parent());
  }
}
void ComputeBlockOp::getSuccessorRegions(
    ::mlir::Region &,
    ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  regions.push_back(RegionSuccessor::parent());
}

ValueRange ComputeOp::getSuccessorInputs(::mlir::RegionSuccessor succ) {
  if (succ.isParent()) {
    return getResults();
  }
  return {};
}
ValueRange ComputeBlockOp::getSuccessorInputs(::mlir::RegionSuccessor succ) {
  if (succ.isParent()) {
    return getResults();
  }
  return getBodyArguments();
}

void ComputeOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<mlir::InvocationBounds> &result) {

  result.push_back(::mlir::InvocationBounds(1, 1));
}

void ComputeOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  if (point == RegionBranchPoint::parent()) {
    regions.emplace_back(&getBody());
  } else {
    // region is body
    regions.push_back(RegionSuccessor::parent());
  }
}
void ComputeOp::getSuccessorRegions(
    ::mlir::Region &,
    ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  regions.push_back(RegionSuccessor::parent());
}

namespace {

static bool isZeroSplatAttr(Attribute attr) {
  auto dense = dyn_cast_or_null<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return false;
  auto val = dense.getSplatValue<Attribute>();
  if (auto ia = dyn_cast<IntegerAttr>(val))
    return ia.getValue().isZero();
  if (auto fa = dyn_cast<FloatAttr>(val))
    return fa.getValue().isZero();
  return false;
}

template <typename Op, typename Adaptor>
static LogicalResult foldGemmlike(Op op, Adaptor adaptor,
                                  SmallVectorImpl<OpFoldResult> &) {
  bool changed = false;
  if (op.getBias() && isZeroSplatAttr(adaptor.getBias())) {
    op.getBiasMutable().clear();
    changed = true;
  }
  if (op.getOut() && isZeroSplatAttr(adaptor.getOut())) {
    op.getOutMutable().clear();
    changed = true;
  }
  return changed ? success() : failure();
}

} // namespace

LogicalResult GemmOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return foldGemmlike(*this, adaptor, results);
}
LogicalResult GemvOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return foldGemmlike(*this, adaptor, results);
}
LogicalResult BatchGemmOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  return foldGemmlike(*this, adaptor, results);
}
LogicalResult BatchGemvOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  return foldGemmlike(*this, adaptor, results);
}

arith::AtomicRMWKind cinm::getArithConstant(ReduceMethod r, Type ty) {
  switch (r) {
  case mlir::cinm::ReduceMethod::ADD:
    if (ty.isFloat()) {
      return mlir::arith::AtomicRMWKind::addf;
    } else {
      return mlir::arith::AtomicRMWKind::addi;
    }
  case mlir::cinm::ReduceMethod::MUL:
    if (ty.isFloat()) {
      return mlir::arith::AtomicRMWKind::mulf;
    } else {
      return mlir::arith::AtomicRMWKind::muli;
    }
  case mlir::cinm::ReduceMethod::MAXSI:
    return mlir::arith::AtomicRMWKind::maxs;
  case mlir::cinm::ReduceMethod::MAXUI:
    return mlir::arith::AtomicRMWKind::maxu;
  case mlir::cinm::ReduceMethod::MAXIMUMF:
    return mlir::arith::AtomicRMWKind::maximumf;
  case mlir::cinm::ReduceMethod::MAXNUMF:
    return mlir::arith::AtomicRMWKind::maxnumf;

  case mlir::cinm::ReduceMethod::MINSI:
    return mlir::arith::AtomicRMWKind::mins;
  case mlir::cinm::ReduceMethod::MINUI:
    return mlir::arith::AtomicRMWKind::minu;
  case mlir::cinm::ReduceMethod::MINIMUMF:
    return mlir::arith::AtomicRMWKind::minimumf;
  case mlir::cinm::ReduceMethod::MINNUMF:
    return mlir::arith::AtomicRMWKind::minnumf;
  }
}
