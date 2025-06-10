/// Implements the Cnm dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmDialect.h"

#define DEBUG_TYPE "cnm-base"

using namespace mlir;
using namespace mlir::cnm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CnmDialect
//===----------------------------------------------------------------------===//

void CnmDialect::initialize()
{
    registerOps();
    registerTypes();
}
