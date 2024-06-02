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
  if (parser.parseLess().failed()) {
    return Type();
  }

  SmallVector<int64_t, 2> shape;
  if (parser.parseDimensionList(shape, false, false).failed()) {
    return Type();
  }

  if (parser.parseGreater().failed()) {
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
  if (parser.parseLess().failed()) {
    return Type();
  }

  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, false, true).failed()) {
    return Type();
  }

  Type element_type;
  if (parser.parseType(element_type).failed()) {
    return Type();
  }

  if (parser.parseComma().failed()) {
    return Type();
  }

  if (parser.parseKeyword("level").failed()) {
    return Type();
  }

  int64_t level;
  if (parser.parseInteger(level).failed()) {
    return Type();
  }

  if (parser.parseGreater().failed()) {
    return Type();
  }

  return cnm::BufferType::get(parser.getContext(), shape, element_type, level);
}

void mlir::cnm::BufferType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  if (!getShape().empty()) {
    printer << "x";
  }
  printer << getElementType();
  printer << ", level " << getLevel();
  printer << ">";
}
