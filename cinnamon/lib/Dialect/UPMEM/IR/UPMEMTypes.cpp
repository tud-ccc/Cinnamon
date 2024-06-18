/// Implements the UPMEM dialect types.
///
/// @file

#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "upmem-types"

using namespace mlir;
using namespace mlir::upmem;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// UPMEMDialect
//===----------------------------------------------------------------------===//

void UPMEMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DeviceHierarchyType
//===----------------------------------------------------------------------===//

Type mlir::upmem::DeviceHierarchyType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t, 3> shape;
  if (parser.parseLess() || parser.parseDimensionList(shape, false, false) ||
      parser.parseGreater()) {
    return Type();
  }

  return upmem::DeviceHierarchyType::get(parser.getContext(), shape);
}

void mlir::upmem::DeviceHierarchyType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << ">";
}

LogicalResult mlir::upmem ::DeviceHierarchyType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape) {
  if (shape.size() != 3)
    return emitError() << "upmem device hierarchy should have 3 dimensions: "
                       << shape;
  return success();
}
