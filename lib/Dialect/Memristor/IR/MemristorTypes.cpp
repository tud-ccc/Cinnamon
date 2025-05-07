/// Implements the Memristor dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Memristor/IR/MemristorTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define DEBUG_TYPE "memristor-types"

using namespace mlir;
using namespace mlir::memristor;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Memristor/IR/MemristorTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MemristorDialect
//===----------------------------------------------------------------------===//
