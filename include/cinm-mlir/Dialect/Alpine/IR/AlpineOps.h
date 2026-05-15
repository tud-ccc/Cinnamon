#pragma once

#include "cinm-mlir/Dialect/Alpine/IR/AlpineTypes.h"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>

namespace mlir::alpine {
std::string generateLibraryCallName(Operation *op);
void appendOperandPrecision(llvm::raw_string_ostream &ss, Type t);
} // namespace mlir::alpine

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.h.inc"

//===----------------------------------------------------------------------===//
