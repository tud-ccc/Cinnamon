/// Implements the Cnm dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h"
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
  CnmAcceleratorAttrInterface ax;
  if (parser.parseLess() || parser.parseCustomAttributeWithFallback(ax) ||
      parser.parseGreater()) {
    return Type();
  }

  return cnm::WorkgroupType::get(parser.getContext(), ax);
}

void mlir::cnm::WorkgroupType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getAccelerator());
  printer << ">";
}

Type mlir::cnm::BufferType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  CnmAcceleratorAttrInterface ax;
  Attribute level;

  if (parser.parseLess() || parser.parseDimensionList(shape, false, true) ||
      parser.parseType(elementType) || parser.parseKeyword("on") ||
      parser.parseCustomAttributeWithFallback(ax))
    return Type();

  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseAttribute(level))
      return Type();
  }
  if (parser.parseGreater()) {
    return Type();
  }

  return cnm::BufferType::get(parser.getContext(), shape, elementType, ax,
                              level);
}

void mlir::cnm::BufferType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  if (!getShape().empty()) {
    printer << "x";
  }
  printer << getElementType();
  printer << " on ";
  if (failed(printer.printAlias(getAccelerator())))
    printer.printAttribute(getAccelerator());
  if (getLevel())
    printer << ", " << getLevel();
  printer << ">";
}
