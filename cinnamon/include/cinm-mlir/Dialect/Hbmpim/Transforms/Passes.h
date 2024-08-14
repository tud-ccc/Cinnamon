/// Declaration of the transform pass within Hbmpim dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Hbmpim/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
