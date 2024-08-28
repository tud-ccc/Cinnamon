/// Declaration of the transform pass within Cim dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cim {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h.inc"
//===----------------------------------------------------------------------===//

} // namespace cim
} // namespace mlir
