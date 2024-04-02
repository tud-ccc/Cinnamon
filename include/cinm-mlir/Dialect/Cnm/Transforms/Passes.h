/// Declaration of the transform pass within Cnm dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
