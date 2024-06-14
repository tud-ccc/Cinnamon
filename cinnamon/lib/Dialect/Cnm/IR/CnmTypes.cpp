/// Implements the Cnm dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "cnm-types"

using namespace mlir;
using namespace mlir::cnm;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CnmDialect
//===----------------------------------------------------------------------===//

void CnmDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.cpp.inc"
      >();
}

// parsers/printers

Type mlir::cnm::WorkgroupType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t, 2> shape;
  if (parser.parseLess() || parser.parseDimensionList(shape, false, false) ||
      parser.parseGreater()) {
    return Type();
  }

  return cnm::WorkgroupType::get(parser.getContext(), shape);
}

void mlir::cnm::WorkgroupType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << ">";
}

Type mlir::cnm::BufferType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape, workgroupShape;
  Type elementType;
  int64_t level;

  if (parser.parseLess() || parser.parseDimensionList(shape, false, true) ||
      parser.parseType(elementType) || parser.parseKeyword("on") ||
      parser.parseDimensionList(workgroupShape, false, false) ||
      parser.parseComma().failed() || parser.parseKeyword("level") ||
      parser.parseInteger(level) || parser.parseGreater()) {
    return Type();
  }

  return cnm::BufferType::get(parser.getContext(), shape, elementType,
                              workgroupShape, level);
}

void mlir::cnm::BufferType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  if (!getShape().empty()) {
    printer << "x";
  }
  printer << getElementType();
  printer << " on ";
  printer.printDimensionList(getWorkgroupShape());
  printer << ", level " << getLevel();
  printer << ">";
}
