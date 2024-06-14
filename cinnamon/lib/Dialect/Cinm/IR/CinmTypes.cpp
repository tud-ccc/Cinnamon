/// Implements the Cinm dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "cinm-types"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.cpp.inc"
        >();
}
