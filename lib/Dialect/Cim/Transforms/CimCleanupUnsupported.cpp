#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h"

#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include "cinm-mlir/Dialect/Cim/IR/CimTypes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace mlir::cim {
#define GEN_PASS_DEF_CIMCLEANUPUNSUPPORTEDPASS
#include "cinm-mlir/Dialect/Cim/Transforms/Passes.h.inc"
}

namespace {

struct LowerCimAddBarrierCopyToLinalgAdd : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto bar = copy.getSource().getDefiningOp<cim::BarrierOp>();
    if (!bar)
      return rewriter.notifyMatchFailure(copy, "src is not cim.barrier");

    Value fut = bar->getOperand(0);
    auto add = llvm::dyn_cast_or_null<cim::AddOp>(fut.getDefiningOp());
    if (!add)
      return rewriter.notifyMatchFailure(copy, "barrier not from cim.op.add");

    Value lhs = add.getOperand(1);
    Value rhs = add.getOperand(2);
    auto lhsMR = llvm::dyn_cast<MemRefType>(lhs.getType());
    auto rhsMR = llvm::dyn_cast<MemRefType>(rhs.getType());
    auto dstMR = llvm::dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!lhsMR || !rhsMR || !dstMR)
      return rewriter.notifyMatchFailure(copy,
                                         "expected memref operands/results");

    if (lhsMR.getElementType() != rhsMR.getElementType() ||
        lhsMR.getElementType() != dstMR.getElementType())
      return rewriter.notifyMatchFailure(copy, "element type mismatch");

    Location loc = copy.getLoc();
    rewriter.setInsertionPoint(copy);
    (void)linalg::AddOp::create(rewriter, 
        loc, ValueRange{lhs, rhs},
        ValueRange{copy.getTarget()});

    rewriter.eraseOp(copy);
    if (bar->use_empty())
      rewriter.eraseOp(bar);
    if (add->use_empty())
      rewriter.eraseOp(add);

    return success();
  }
};

struct EraseDeadBarrier : OpRewritePattern<cim::BarrierOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cim::BarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "barrier still used");
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseReleaseDevice : OpRewritePattern<cim::ReleaseDeviceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cim::ReleaseDeviceOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseReleaseCrossbar : OpRewritePattern<cim::ReleaseCrossbarOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cim::ReleaseCrossbarOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseDeadAcquireDevice : OpRewritePattern<cim::AcquireDeviceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cim::AcquireDeviceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "acquire_device still used");
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseDeadAcquireCrossbar : OpRewritePattern<cim::AcquireCrossbarOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cim::AcquireCrossbarOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "acquire_crossbar still used");
    rewriter.eraseOp(op);
    return success();
  }
};

struct CimCleanupUnsupported
    : public mlir::cim::impl::CimCleanupUnsupportedPassBase<
          CimCleanupUnsupported> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect, func::FuncDialect,
                    arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<LowerCimAddBarrierCopyToLinalgAdd>(ctx);

    patterns.add<EraseDeadBarrier, EraseReleaseDevice, EraseReleaseCrossbar,
                 EraseDeadAcquireDevice, EraseDeadAcquireCrossbar>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}

std::unique_ptr<mlir::Pass> createCimCleanupUnsupportedPass() {
  return std::make_unique<CimCleanupUnsupported>();
}
