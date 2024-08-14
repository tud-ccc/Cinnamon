/// Implements the Hbmpim dialect ops.
///
/// @file

#include <cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.h>

#include <cinm-mlir/Dialect/Hbmpim/IR/HbmpimTypes.h>
#include <cinm-mlir/Utils/CinmUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

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

#define DEBUG_TYPE "hbmpimm-ops"

using namespace mlir;
using namespace mlir::hbmpim;


//===----------------------------------------------------------------------===//
// HbmpimDialect
//===----------------------------------------------------------------------===//

void HbmpimDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// HbmpimFuncOp
//===----------------------------------------------------------------------===//

ParseResult HbmpimFuncOp::parse(OpAsmParser &parser, OperationState &result) {

  // Parse the function name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parser.parseLParen() || parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  return parser.parseRegion(*body, {});
}

void HbmpimFuncOp::print(OpAsmPrinter &p) {
  ::mlir::Builder odsBuilder{getContext()};
  p << ' ';
  p.printSymbolName(getName());
  p << "()";
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

static ParseResult parseLaunchFuncOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &argNames,
    SmallVectorImpl<Type> &argTypes) {
  if (parser.parseOptionalKeyword("args"))
    return success();

  auto parseElement = [&]() -> ParseResult {
    return failure(parser.parseOperand(argNames.emplace_back()) ||
                   parser.parseColonType(argTypes.emplace_back()));
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElement, " in argument list");
}

static void printLaunchFuncOperands(OpAsmPrinter &printer, Operation *,
                                    OperandRange operands, TypeRange types) {
  if (operands.empty())
    return;
  printer << "args(";
  llvm::interleaveComma(llvm::zip(operands, types), printer,
                        [&](const auto &pair) {
                          printer.printOperand(std::get<0>(pair));
                          printer << " : ";
                          printer.printType(std::get<1>(pair));
                        });
  printer << ")";
}

void HbmpimLaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                         HbmpimFuncOp kernelFunc, 
                         ValueRange kernelOperands) {
  result.addOperands(kernelOperands);
  auto kernelModule = kernelFunc->getParentOfType<HbmpimModuleOp>();
  auto kernelSymbol =
      SymbolRefAttr::get(kernelModule.getNameAttr(),
                         {SymbolRefAttr::get(kernelFunc.getNameAttr())});

  Properties &prop = result.getOrAddProperties<Properties>();
  prop.kernel = kernelSymbol;
  // size_t segmentSizesLen = std::size(prop.operandSegmentSizes);
  // Initialize the segment sizes to 1.
  // for (auto &sz : prop.operandSegmentSizes)
  //   sz = 1;
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.cpp.inc"