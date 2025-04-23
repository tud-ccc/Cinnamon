/// Implements the Cinm dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cinm-base"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::initialize() {
  registerOps();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.cpp.inc"
      >();
}

::mlir::LogicalResult
CinmDialect::verifyOperationAttribute(::mlir::Operation *op,
                                      ::mlir::NamedAttribute attribute) {

  if (attribute.getName() == CinmDialect::NOTILE_NAME) {
    if (op->getDialect() == this) {
      return success();
    }
    return op->emitOpError()
           << CinmDialect::NOTILE_NAME
           << " attribute can only be used on cinm dialect operations";
  }
  return op->emitOpError("unknown attribute ") << attribute.getName();
}
