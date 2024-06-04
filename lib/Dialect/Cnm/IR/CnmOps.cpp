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

ParseResult WorkgroupOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  if (parser.parseLSquare().failed()) {
    return ParseResult::failure();
  }

  int64_t current = 0;
  llvm::SmallVector<int64_t, 2> shape;
  NamedAttrList attributes;

  OptionalParseResult dimension = parser.parseOptionalInteger(current);
  while (dimension.has_value()) {
    if (dimension.value().failed()) {
      return ParseResult::failure();
    }

    shape.push_back(current);
    dimension = parser.parseOptionalInteger(current);
  }

  if (parser.parseRSquare() || parser.parseOptionalAttrDict(attributes)) {
    return ParseResult::failure();
  }

  result.addAttributes(attributes);
  result.addTypes(WorkgroupType::get(result.getContext(), shape));

  return ParseResult::success();
}

void WorkgroupOp::print(mlir::OpAsmPrinter &printer) {
  printer << " [";
  const auto shape = getType().getShape();
  for (uint64_t i = 0; i < shape.size(); i++) {
    if (i > 0) {
      printer << " ";
    }
    printer << shape[i];
  }
  printer << "]";
  printer.printOptionalAttrDict(this->getOperation()->getAttrs());
}

ParseResult AllocOp::parse(mlir::OpAsmParser &parser,
                           mlir::OperationState &result) {

  OpAsmParser::UnresolvedOperand wg;
  NamedAttrList attributes;
  Type bufferType;
  Type workgroupType;

  if (parser.parseLParen() || parser.parseRParen() ||
      parser.parseKeyword("for") || parser.parseOperand(wg) ||
      parser.parseOptionalAttrDict(attributes) ||
      parser.parseColonType(bufferType) || parser.parseKeyword("for") ||
      parser.parseType(workgroupType)) {
    return ParseResult::failure();
  }

  llvm::SmallVector<Value, 1> operands;
  if (parser.resolveOperand(wg, workgroupType, operands).failed()) {
    return ParseResult::failure();
  }

  result.addOperands(operands);
  result.addAttributes(attributes);
  result.addTypes(bufferType);

  return ParseResult::success();
}

void AllocOp::print(mlir::OpAsmPrinter &printer) {
  printer << " () for " << getOperand();
  printer.printOptionalAttrDict(this->getOperation()->getAttrs());
  printer << " : " << getType() << " for " << getOperand().getType();
}

ParseResult SetZeroOp::parse(mlir::OpAsmParser &parser,
                             mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand buffer;
  Type bufferType;
  llvm::SmallVector<Value, 1> operands;

  if (parser.parseOperand(buffer) || parser.parseColonType(bufferType) ||
      parser.resolveOperand(buffer, bufferType, operands)) {
    return ParseResult::failure();
  }

  result.addOperands(operands);
  result.addTypes(bufferType);

  return ParseResult::success();
}

void SetZeroOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getOperand() << " : " << getOperand().getType();
}

ParseResult ScatterOp::parse(mlir::OpAsmParser &parser,
                             mlir::OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> unresolvedOperands;
  AffineMapAttr mapAttr;
  SmallVector<Type, 3> operandTypes;

  if (parser.parseOperand(unresolvedOperands.emplace_back()) ||
      parser.parseKeyword("into") ||
      parser.parseOperand(unresolvedOperands.emplace_back()) ||
      parser.parseLSquare() ||
      parser.parseAttribute(mapAttr, "scatterMap", result.attributes) ||
      parser.parseRSquare() || parser.parseKeyword("of") ||
      parser.parseOperand(unresolvedOperands.emplace_back()) ||
      parser.parseColonType(operandTypes.emplace_back()) ||
      parser.parseKeyword("into") ||
      parser.parseType(operandTypes.emplace_back()) ||
      parser.parseKeyword("of") ||
      parser.parseType(operandTypes.emplace_back())) {
    return ParseResult::failure();
  }

  llvm::SmallVector<Value, 3> operands;
  for (size_t i = 0; i < unresolvedOperands.size(); i++) {
    if (parser.resolveOperand(unresolvedOperands[i], operandTypes[i],
                              operands)) {
      return ParseResult::failure();
    }
  }

  result.addOperands(operands);
  result.addTypes(ScatterTokenType::get(parser.getContext()));

  return ParseResult::success();
}

void ScatterOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getOperand(0) << " into " << getOperand(1) << "["
          << getScatterMapAttr() << "]"
          << " of " << getOperand(2);
  printer << " : " << getOperand(0).getType() << " into "
          << getOperand(1).getType() << " of " << getOperand(2).getType();
}

ParseResult GatherOp::parse(mlir::OpAsmParser &parser,
                            mlir::OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> unresolvedOperands;
  Type result_type;
  SmallVector<Type, 2> operandTypes;
  AffineMapAttr mapAttr;

  if (parser.parseOperand(unresolvedOperands.emplace_back()) ||
      parser.parseLSquare() ||
      parser.parseAttribute(mapAttr, "scatterMap", result.attributes) ||
      parser.parseRSquare() || parser.parseKeyword("of") ||
      parser.parseOperand(unresolvedOperands.emplace_back()) ||
      parser.parseColonType(operandTypes.emplace_back()) ||
      parser.parseKeyword("of") ||
      parser.parseType(operandTypes.emplace_back()) ||
      parser.parseKeyword("into") || parser.parseType(result_type)) {
    return ParseResult::failure();
  }

  llvm::SmallVector<Value, 2> operands;
  for (size_t i = 0; i < unresolvedOperands.size(); i++) {
    if (parser.resolveOperand(unresolvedOperands[i], operandTypes[i],
                              operands)) {
      return ParseResult::failure();
    }
  }

  result.addOperands(operands);
  result.addTypes(result_type);
  result.addTypes(GatherTokenType::get(parser.getContext()));

  return ParseResult::success();
}

void GatherOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getOperand(0) << "[" << getScatterMapAttr() << "]"
          << " of " << getOperand(1);
  printer << " : " << getOperand(0).getType() << " of "
          << getOperand(1).getType() << " into " << getResultTypes()[0];
}

ParseResult LaunchOp::parse(mlir::OpAsmParser &parser,
                            mlir::OperationState &result) {
  result.addTypes(LaunchTokenType::get(parser.getContext()));

  SmallVector<OpAsmParser::UnresolvedOperand> unresolvedOperands;
  SmallVector<Type> operandTypes;
  operandTypes.push_back(Type()); // workgroup type

  if (parser.parseOperand(unresolvedOperands.emplace_back()) ||
      parser.parseLParen()) {
    return ParseResult::failure();
  }

  if (parser.parseOptionalRParen().failed()) { // 1 or more parameters
    if (parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
      return ParseResult::failure();
    }

    while (parser.parseOptionalColon().failed()) {
      if (parser.parseComma().failed() ||
          parser.parseOperand(unresolvedOperands.emplace_back()).failed()) {
        return ParseResult::failure();
      }
    }

    if (parser.parseType(operandTypes.emplace_back()).failed()) {
      return ParseResult::failure();
    }

    while (parser.parseOptionalRParen().failed()) {
      if (parser.parseComma().failed() ||
          parser.parseType(operandTypes.emplace_back()).failed()) {
        return ParseResult::failure();
      }
    }
  }

  if (parser.parseKeyword("on") || parser.parseType(operandTypes[0]) ||
      parser.parseRegion(*result.addRegion(), {}, false)) {
    return ParseResult::failure();
  }

  auto block = &result.regions.back()->back();
  if (!llvm::isa<cnm::TerminatorOp>(block->back())) {
    OpBuilder builder(result.getContext());
    builder.setInsertionPointToEnd(&result.regions.back()->back());
    builder.create<cnm::TerminatorOp>(result.location);
  }

  llvm::SmallVector<Value> operands;
  for (size_t i = 0; i < unresolvedOperands.size(); i++) {
    if (parser.resolveOperand(unresolvedOperands[i], operandTypes[i],
                              operands)) {
      return ParseResult::failure();
    }
  }

  result.addOperands(operands);

  return ParseResult::success();
}

void LaunchOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getOperand(0);
  printer << " (";
  for (uint64_t i = 1; i < getNumOperands(); i++) {
    if (i > 1) {
      printer << ", ";
    }
    printer << getOperand(i);
  }
  printer << ": ";
  for (uint64_t i = 1; i < getNumOperands(); i++) {
    if (i > 1) {
      printer << ", ";
    }
    printer << getOperand(i).getType();
  }
  printer << ") on " << getWg().getType();
  printer.printRegion(getRegion());
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