/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"
        >();
}

// parsers/printers
