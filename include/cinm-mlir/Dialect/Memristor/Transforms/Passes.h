/// Declaration of the transform pass within Memristor dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace memristor {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Memristor/Transforms/Passes.h.inc"
//===----------------------------------------------------------------------===//

void populateMemristorScheduleAsapPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
bool memristorScheduleAsapCheckDynamicallyLegal(Operation *op);
void populateMemristorScheduleAlapPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
bool memristorScheduleAlapCheckDynamicallyLegal(Operation *op);

void populateMemristorEraseRedundantBarriersPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx);

bool isMemristorOp(Operation *op);
bool isMemristorFuture(Value v);
bool isLegalOp(Operation *op);
bool isLegalBarrier(Operation *op);

struct MemristorEraseRedundantBarriersPattern;

} // namespace memristor
} // namespace mlir
