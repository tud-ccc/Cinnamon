#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMDECOMPOSEACCUMULATIONPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

template <typename GemmLikeOp>
static LogicalResult rewriteGemmTensorBias(GemmLikeOp op, RewriterBase &b) {

  auto bias = op.getBias();
  if (!bias)
    return success();
  op.getBiasMutable().clear();

  Location loc = op.getLoc();

  if (op.getResult()) {
    // tensor variant
    b.setInsertionPointAfter(op);
    auto sum = cinm::ElementwiseOp::create(b, loc, ElementwiseKind::Add,
                                           op.getResult(), bias);
    b.replaceAllUsesWith(op.getResult(), sum.getResult());
  } else {
    // memref variant
    // todo insert toBuffer op for the bias?
    b.setInsertionPoint(op); // do it before
    cinm::ElementwiseOp::create(b, loc, ElementwiseKind::Add, op.getOut(), bias,
                                op.getOut());
  }
  return success();
}

struct DecomposeCinmAccumulationsPass
    : public impl::CinmDecomposeAccumulationPassBase<
          DecomposeCinmAccumulationsPass> {

  void runOnOperation() override {
    Operation *func = getOperation();
    IRRewriter b(func->getContext());

    auto res = func->walk([&](Operation *op) {
      if (auto gemv = dyn_cast<cinm::GemvOp>(op)) {
        if (failed(rewriteGemmTensorBias(gemv, b)))
          return WalkResult::interrupt();
      } else if (auto gemm = dyn_cast<cinm::GemmOp>(op)) {
        if (failed(rewriteGemmTensorBias(gemv, b)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

void registerDecomposeCinmAccumulationPass() {
  PassRegistration<DecomposeCinmAccumulationsPass>(
      []() -> std::unique_ptr<mlir::Pass> {
        return std::make_unique<DecomposeCinmAccumulationsPass>();
      });
}

} // namespace mlir::cinm
