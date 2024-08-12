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

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimOps.cpp.inc"

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
