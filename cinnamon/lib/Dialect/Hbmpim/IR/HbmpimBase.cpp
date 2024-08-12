/// Implements the Hbmpim dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimBase.h"

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimDialect.h"

#define DEBUG_TYPE "hbmpim-base"

using namespace mlir;
using namespace mlir::hbmpim;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Hbmpim/IR/HbmpimBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HbmpimDialect
//===----------------------------------------------------------------------===//

void HbmpimDialect::initialize()
{
    registerOps();
    registerTypes();
}
