/// Declaration of the transform pass within Cim dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace cim {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h.inc"
//===----------------------------------------------------------------------===//

void populateCimScheduleAsapPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
bool cimScheduleAsapCheckDynamicallyLegal(Operation *op);
void populateCimScheduleAlapPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
bool cimScheduleAlapCheckDynamicallyLegal(Operation *op);

void populateCimEraseRedundantBarriersPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);

bool isCimOp(Operation *op);
bool isCimFuture(Value v);
bool isLegalOp(Operation *op);
bool isLegalBarrier(Operation *op);

struct CimEraseRedundantBarriersPattern;

} // namespace cim
} // namespace mlir
