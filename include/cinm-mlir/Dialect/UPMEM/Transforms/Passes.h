/// Declaration of the transform pass within UPMEM dialect.
///
/// @file

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//
namespace upmem {

} // namespace upmem

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
