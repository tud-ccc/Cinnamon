/// Implements the Cim dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Cim/IR/CimBase.h"

#include "cinm-mlir/Dialect/Cim/IR/CimDialect.h"

#define DEBUG_TYPE "cim-base"

using namespace mlir;
using namespace mlir::cim;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cim/IR/CimBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CimDialect
//===----------------------------------------------------------------------===//

void CimDialect::initialize() {
  registerOps();
  registerTypes();
}
