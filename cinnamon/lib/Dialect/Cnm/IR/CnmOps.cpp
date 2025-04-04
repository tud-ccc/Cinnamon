/// Implements the Cnm dialect ops.
///
/// @file

#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>

#include <cinm-mlir/Dialect/Cnm/IR/CnmTypes.h>
#include <cinm-mlir/Utils/CinmUtils.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
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
  if (isa<MemRefType>(out.getType())) {
    return success();
  } else if (isa<RankedTensorType>(out.getType())) {
    ShapedType ty = cast<RankedTensorType>(out.getType());

    inferredReturnShapes.push_back(ShapedTypeComponents(ty));
    return success();
  }

  return failure();
}

static ParseResult parseAffineMapInlineOrNot(OpAsmParser &parser,
                                             Attribute &affineMapAttr) {

  if (failed(parser.parseCustomAttributeWithFallback(
          affineMapAttr, Type(), [&](Attribute &result, Type) {
            AffineMap inlineMap;
            if (parser.parseAffineMap(inlineMap))
              return failure();
            result = AffineMapAttr::get(inlineMap);
            return success();
          })))
    return failure();
  if (!isa<AffineMapAttr>(affineMapAttr))
    return parser.emitError(
        parser.getCurrentLocation(),
        "invalid kind of attribute specified, expected affine map");
  return success();
}
static ParseResult parseComputeOperand(OpAsmParser &parser,
                                       Attribute &affineMapAttr,
                                       OperationState &result,
                                       bool canBeResult) {
  OpAsmParser::UnresolvedOperand operand;
  Type type;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parseAffineMapInlineOrNot(parser, affineMapAttr) ||
      parser.parseRSquare() || parser.parseColonType(type) ||
      parser.resolveOperand(operand, type, result.operands)) {
    return failure();
  }
  // a tensor result
  if (canBeResult && isa<TensorType>(type)) {
    result.addTypes(type);
  }

  return success();
}

static ParseResult
parseComputeOperandList(OpAsmParser &parser, StringRef kw,
                        llvm::SmallVectorImpl<Attribute> &affineMaps,
                        OperationState &result, bool canBeResult = false) {

  if (parser.parseKeyword(kw) || parser.parseLParen() ||
      parser.parseCommaSeparatedList([&]() -> ParseResult {
        return parseComputeOperand(parser, affineMaps.emplace_back(), result,
                                   canBeResult);
      }) ||
      parser.parseRParen())
    return failure();
  return success();
}

ParseResult ComputeOp::parse(OpAsmParser &parser, OperationState &result) {
  /*
  cnm.launch
  (symbols [%O1, %O2])?
  ins(%as[(i)->(i)]: memref<2x512xi32>)
  outs(%os[(i)->(i)]: memref<2x512xi32>)
  on hierarchy<2>
  do (%a1: memref<512xi32>,
      %o1: memref<512xi32>)  {
    affine.parallel (%i) = (0) to (512) {
      %x = memref.load %a1[%i]
      %t2 = arith.muli %x, 2
      memref.store %t2, %o1[%i]
    }
  }
  */
  int numSymbols = 0;
  if (succeeded(parser.parseOptionalKeyword("symbols"))) {
    SmallVector<OpAsmParser::UnresolvedOperand> symbolBindings;
    if (parser.parseOperandList(symbolBindings,
                                OpAsmParser::Delimiter::Square) ||
        parser.resolveOperands(symbolBindings,
                               parser.getBuilder().getIndexType(),
                               result.operands))
      return failure();

    numSymbols = symbolBindings.size();
  }

  SmallVector<Attribute> affineMaps;
  if (parseComputeOperandList(parser, "ins", affineMaps, result))
    return failure();
  const int64_t numInputs = affineMaps.size();
  if (parseComputeOperandList(parser, "outs", affineMaps, result,
                              /*canBeResult=*/true))
    return failure();
  const int numBuffers = affineMaps.size();
  result.addAttribute(
      getNumInputsAttrName(result.name),
      IntegerAttr::get(IntegerType::get(result.getContext(), 64), numInputs));
  result.addAttribute(getAffineMapsAttrName(result.name),
                      ArrayAttr::get(result.getContext(), affineMaps));

  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      parser.getBuilder().getDenseI32ArrayAttr({numSymbols, numBuffers}));

  SmallVector<int64_t> workgroupDimensions;
  if (parser.parseKeyword("on") || parser.parseKeyword("hierarchy") ||
      parser.parseLess() ||
      parser.parseDimensionList(workgroupDimensions, false, false) ||
      parser.parseGreater())
    return failure();

  result.addAttribute(
      getWorkgroupShapeAttrName(result.name),
      DenseI64ArrayAttr::get(result.getContext(), workgroupDimensions));

  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseKeyword("do") ||
      parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true)) {
    return failure();
  }
  auto &region = *result.addRegion();
  if (parser.parseRegion(region, args)) {
    return failure();
  }
  ComputeOp::ensureTerminator(region, parser.getBuilder(), result.location);

  // todo results, bufferization

  return success();
}

void ComputeOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                      ArrayRef<int64_t> workgroupShape, ValueRange allBuffers,
                      uint64_t numInputs, ArrayRef<AffineMap> affineMaps,
                      ValueRange symbolBindings) {
  if (numInputs > allBuffers.size()) {
    mlir::emitError(state.location, "Invalid number of inputs ")
        << numInputs << " > " << allBuffers.size();
    return;
  }

  build(builder, state, workgroupShape, allBuffers.slice(0, numInputs),
        allBuffers.slice(numInputs, allBuffers.size() - numInputs), affineMaps,
        symbolBindings);
}

void ComputeOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                      ArrayRef<int64_t> workgroupShape, ValueRange inputs,
                      ValueRange inits, ArrayRef<AffineMap> affineMaps,
                      ValueRange symbolBindings) {

  state.addOperands(symbolBindings);
  state.addOperands(inputs);
  state.addOperands(inits);

  state.addAttribute(getOperandSegmentSizesAttrName(state.name),
                     builder.getDenseI32ArrayAttr(
                         {static_cast<int>(symbolBindings.size()),
                          static_cast<int>(inputs.size() + inits.size())}));

  state.addAttribute(getNumInputsAttrName(state.name),
                     builder.getI64IntegerAttr(inputs.size()));
  state.addAttribute(getWorkgroupShapeAttrName(state.name),
                     builder.getDenseI64ArrayAttr(workgroupShape));
  state.addAttribute(getAffineMapsAttrName(state.name),
                     builder.getAffineMapArrayAttr(affineMaps));

  auto &entry = state.addRegion()->emplaceBlock();
  for (auto [i, buf] : llvm::enumerate(
           llvm::drop_begin(state.operands, symbolBindings.size()))) {

    if (auto bufTy = llvm::cast_or_null<ShapedType>(buf.getType())) {
      auto bufShape = bufTy.getShape();
      if (i < affineMaps.size()) {
        auto map = affineMaps[i];
        auto argRank = bufShape.size() - map.getNumResults();
        if (argRank >= 0) {
          auto argShape = bufShape.slice(map.getNumResults());
          auto argTy = MemRefType::get(argShape, bufTy.getElementType());
          entry.addArgument(argTy, state.location);
        } else {
          mlir::emitError(state.location, "Buffer of type ")
              << buf.getType() << " cannot be addressed by map "
              << AffineMapAttr::get(map);
        }
      }
      // tensor result
      if (i >= inputs.size() && isa<TensorType>(buf.getType()))
        state.addTypes(buf.getType());
    }
  }
  ComputeOp::ensureTerminator(*state.regions[0], builder, state.location);
}

void ComputeOp::print(OpAsmPrinter &out) {
  bool first = true;
  out.increaseIndent();
  out.printNewline();
  auto syms = getSymbolBindings();
  if (!syms.empty()) {
    out << "symbols [";
    out.printOperands(syms);
    out << "]";
    out.printNewline();
  }
  out << "ins(";
  for (auto [buf, map, i] :
       llvm::zip(getBuffers(), getAffineMaps(), llvm::seq(0UL, 100000UL))) {
    if (i == getNumInputs()) {
      out << ")";
      out.printNewline();
      out << "outs(";
      first = true;
    }
    if (!first) {
      out << ", ";
    }
    first = false;

    out.printOperand(buf);
    out << "[";
    llvm::cast<AffineMapAttr>(map).getValue().print(out.getStream());
    // out.printAttributeWithoutType(map);
    out << "] : ";
    out.printType(buf.getType());
  }
  out << ") ";
  out.printNewline();
  out << "on hierarchy<";
  out.printDimensionList(getWorkgroupShape());
  out << ">";
  out.printNewline();
  out << "do (";
  llvm::interleaveComma(getBody().getArguments(), out,
                        [&](auto arg) { out.printRegionArgument(arg); });
  out << ") ";
  out.printRegion(getBody(), false, false);
  out.decreaseIndent();
}

InFlightDiagnostic emitNiceError(Operation *op, Location loc,
                                 const Twine &message) {
  InFlightDiagnostic diag = mlir::emitError(loc, message);
  if (op->getContext()->shouldPrintOpOnDiagnostic()) {
    diag.attachNote(op->getLoc())
        .append("see current operation: ")
        .appendOp(*op, OpPrintingFlags().printGenericOpForm());
  }
  return diag;
}

LogicalResult ComputeOp::verify() {
  if (getWorkgroupShape().empty()) {
    return emitOpError("has empty workgroup shape");
  }
  if (getAffineMaps().size() != getBuffers().size()) {
    return emitOpError("affine map count does not match in/out buffer count (")
           << getAffineMaps().size() << " != " << getBuffers().size() << ")";
  }
  auto args = getBody().getArguments();
  if (args.size() != getBuffers().size()) {
    return emitOpError(
               "kernel argument count does not match in/out buffer count (")
           << args.size() << " != " << getBuffers().size() << ")";
  }

  // compute op may be partially bufferized
  SmallVector<Type> tensorArgs;
  for (auto [i, buf] : llvm::enumerate(getOutBuffers())) {
    if (isa<RankedTensorType>(buf.getType())) {
      tensorArgs.push_back(buf.getType());
    } else if (!isa<MemRefType>(buf.getType())) {
      return emitOpError("out argument #")
             << i << " should be a tensor or memref";
    }
  }
  if (tensorArgs != getResultTypes()) {
    return emitOpError("tensor results do not match tensor arguments");
  }

  const auto symbolCount = getSymbolBindings().size();

  for (auto [arg, buf, map, i] : llvm::zip(
           args, getBuffers(), getAffineMaps().getAsValueRange<AffineMapAttr>(),
           llvm::seq(0UL, 10000000UL))) {
    if (!isa<MemRefType>(arg.getType()))
      return emitNiceError(*this, arg.getLoc(), "kernel argument #")
             << i << " should be a memref";

    if (map.getNumDims() != getWorkgroupShape().size())
      return emitOpError("map for argument #")
             << i << " should have " << getWorkgroupShape().size()
             << " input dimensions, got " << map.getNumDims();

    if (map.getNumSymbols() != symbolCount)
      return emitOpError("map for argument #")
             << i << " should have " << symbolCount << " input symbols, got "
             << map.getNumSymbols();

    auto argTy = llvm::cast<ShapedType>(arg.getType());
    auto inputTy = llvm::cast<ShapedType>(buf.getType());
    if (argTy.getElementType() != inputTy.getElementType())
      return emitNiceError(*this, arg.getLoc(), "Kernel argument #")
             << i << " should have element type " << inputTy.getElementType();

    auto argShape = argTy.getShape();
    auto bufShape = inputTy.getShape();

    if (argShape.size() > bufShape.size())
      return emitNiceError(*this, arg.getLoc(), "Kernel argument #")
             << i << " should have fewer than " << bufShape.size()
             << " dimensions, got " << argShape.size();

    if (map.getNumResults() + argShape.size() != bufShape.size())
      return emitNiceError(*this, arg.getLoc(),
                           "Buffer, map, and kernel argument #")
             << i << " are incompatible";

    if (bufShape.slice(map.getNumResults()) != argShape) {
      return emitNiceError(*this, arg.getLoc(), "Kernel argument #")
             << i << "shape should be suffix of corresponding buffer shape, ("
             << bufShape.slice(map.getNumResults()) << " != " << argShape
             << ")";
    }
  }

  return success();
}

LogicalResult LaunchOp::verify() {
  auto bodyArgs = getBody().getArguments();
  auto operands = getParams();
  if (bodyArgs.size() != operands.size())
    return emitOpError("expected ")
           << operands.size() << " arguments, got " << bodyArgs.size();

  for (auto [arg, operand] : llvm::zip(bodyArgs, operands)) {
    if (auto bufTy = dyn_cast<cnm::BufferType>(operand.getType())) {
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

  if (!mlir::scatteredMemrefIsContiguous(getInput(), bufferTy.getShape())) {
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

  if (!mlir::scatteredMemrefIsContiguous(getOutputBuf(), bufferTy.getShape())) {
    return emitOpError("should gather into a contiguous memref");
  }

  return success();
}
