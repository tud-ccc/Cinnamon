/// Declaration of conversion passes for the ${dialectNameUpper} dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Forward declaration from Dialect.h
template<typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace ${dialectNs} {
class ${dialectNameUpper}Dialect;
} // namespace ${dialectNs}

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "${projectPrefix}/Conversion/${dialectNameUpper}Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir