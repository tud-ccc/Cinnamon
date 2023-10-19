/// Implements the Cinm dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"

#define DEBUG_TYPE "cinm-base"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::initialize()
{
    registerOps();
    registerTypes();
}
