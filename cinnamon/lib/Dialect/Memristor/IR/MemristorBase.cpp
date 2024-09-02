/// Implements the Memristor dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Memristor/IR/MemristorBase.h"

#include "cinm-mlir/Dialect/Memristor/IR/MemristorDialect.h"

#define DEBUG_TYPE "memristor-base"

using namespace mlir;
using namespace mlir::memristor;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Memristor/IR/MemristorBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MemristorDialect
//===----------------------------------------------------------------------===//

void MemristorDialect::initialize() { registerOps(); }
