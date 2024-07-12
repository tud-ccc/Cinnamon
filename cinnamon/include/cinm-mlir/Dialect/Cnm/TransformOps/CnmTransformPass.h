

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Pass/Pass.h>

namespace mlir::cnm {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Cnm/TransformOps/TransformPass.h.inc"

} // namespace mlir::cnm