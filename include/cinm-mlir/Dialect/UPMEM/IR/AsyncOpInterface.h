
#pragma once

#include "mlir/IR/OpDefinition.h"

//===- Generated includes -------------------------------------------------===//
namespace mlir {
namespace upmem {
// Adds a `upmem.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);
} // namespace upmem
} // namespace mlir

#include "cinm-mlir/Dialect/UPMEM/IR/AsyncOpInterface.h.inc"

//===----------------------------------------------------------------------===//
