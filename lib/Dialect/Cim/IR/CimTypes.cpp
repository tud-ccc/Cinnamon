/// Implements the Cim dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "cim-types"

using namespace mlir;
using namespace mlir::cim;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CimDialect
//===----------------------------------------------------------------------===//

void CimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.cpp.inc"
      >();
}

Type mlir::cim::FutureType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;

  if (parser.parseLess() ||                            //
      parser.parseDimensionList(shape, false, true) || //
      parser.parseType(elementType) ||                 //
      parser.parseGreater()) {
    return Type();
  }

  return cim::FutureType::get(parser.getContext(), shape, elementType);
}

void mlir::cim::FutureType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << (getShape().empty() ? "" : "x") << getElementType() << ">";
}