/// Declaration of conversion passes for the Cinm dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace cinm {
class CinmDialect;
} // namespace cinm

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir