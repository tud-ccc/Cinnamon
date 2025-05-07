/// Declaration of the Memristor dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Memristor/IR/MemristorTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir::memristor {
std::string generateLibraryCallName(Operation *op);
void appendOperandPrecision(llvm::raw_string_ostream &ss, Type t);
} // namespace mlir::memristor

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Memristor/IR/MemristorOps.h.inc"

//===----------------------------------------------------------------------===//
