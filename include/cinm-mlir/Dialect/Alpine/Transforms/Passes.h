#pragma once

#include "mlir/Pass/Pass.h"
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace alpine {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Alpine/Transforms/Passes.h.inc"
//===----------------------------------------------------------------------===//

bool isAlpineOp(Operation *op);
bool isAlpineFuture(Value v);
bool isLegalOp(Operation *op);
bool isLegalBarrier(Operation *op);

struct AlpineEraseRedundantBarriersPattern;

} // namespace alpine
} // namespace mlir
