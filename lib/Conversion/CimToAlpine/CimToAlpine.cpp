#include "cinm-mlir/Conversion/CimToAlpine/CimToAlpine.h"
#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.h"
#include "cinm-mlir/Dialect/Cim/IR/CimOps.h"
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace mlir::cim {
#define GEN_PASS_DEF_CONVERTCIMTOALPINEPASS
#include "cinm-mlir/Conversion/CimPasses.h.inc"
}

namespace {

static inline bool isRelowerTrue(Operation *op) {
  if (auto b = op->getAttrOfType<BoolAttr>("relower"))
    return b.getValue();
  return false;
}

using XbarToTileMap = DenseMap<Value, Value>;

static void preAllocateTiles(Operation *root, XbarToTileMap &map) {
  root->walk([&](cim::AcquireCrossbarOp acq) {
    Value xb = acq.getResult();
    if (map.count(xb))
      return;
    OpBuilder b(acq);
    b.setInsertionPointAfter(acq);
    SmallVector<NamedAttribute> attrs;
    if (auto h = acq->getAttr("height"))
      attrs.emplace_back(b.getStringAttr("height"), h);
    if (auto w = acq->getAttr("width"))
      attrs.emplace_back(b.getStringAttr("width"), w);
    Value tile = alpine::AllocTileOp::create(b, 
        acq.getLoc(),  b.getI32Type(),  ValueRange{},
         attrs);
    map.try_emplace(xb, tile);
  });
}

struct LowerCimQuantizeToAlpine : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto bar = copy.getSource().getDefiningOp<cim::BarrierOp>();
    if (!bar)
      return rewriter.notifyMatchFailure(copy, "src is not cim.barrier");

    Value fut = bar->getOperand(0);
    auto q = dyn_cast_or_null<cim::QuantizeOp>(fut.getDefiningOp());
    if (!q)
      return rewriter.notifyMatchFailure(copy,
                                         "barrier is not of cim.op.quantize");

    if (isRelowerTrue(q))
      return rewriter.notifyMatchFailure(copy, "quantize relower=true");

    Value dst = copy.getTarget();
    auto dstMR = dyn_cast<MemRefType>(dst.getType());
    if (!dstMR)
      return rewriter.notifyMatchFailure(copy, "dst must be a memref");

    auto zpAttr = q.getZeroPointAttr();
    if (!zpAttr)
      return rewriter.notifyMatchFailure(copy, "missing zeroPoint attr");
    int64_t z64 = zpAttr.getInt();
    if (z64 < -128 || z64 > 127)
      return rewriter.notifyMatchFailure(
          copy, "zeroPoint out of i8 range [-128, 127]");

    Location loc = copy.getLoc();

    Value srcForOp = q.getSrc();
    MemRefType srcTy = dyn_cast<MemRefType>(srcForOp.getType());
    if (!srcTy)
      return rewriter.notifyMatchFailure(copy, "src must be memref");

    if (auto altSrcTy = dyn_cast<MemRefType>(copy.getSource().getType())) {
      if (altSrcTy.getRank() == srcTy.getRank() && altSrcTy != srcTy &&
          memref::CastOp::areCastCompatible(srcTy, altSrcTy)) {
        srcForOp = memref::CastOp::create(rewriter, loc, altSrcTy, srcForOp);
        srcTy = altSrcTy;
      }
    }

    Value dstForOp = dst;
    if (srcTy.getRank() == dstMR.getRank()) {
      SmallVector<int64_t> shape(srcTy.getShape().begin(), srcTy.getShape().end());
      auto expectedDstTy =
          MemRefType::get(shape, dstMR.getElementType(), dstMR.getLayout(),
                          dstMR.getMemorySpace());
      if (expectedDstTy != dstMR &&
          memref::CastOp::areCastCompatible(dstMR, expectedDstTy)) {
        dstForOp = memref::CastOp::create(rewriter, loc, expectedDstTy, dst);
      }
    }

    rewriter.setInsertionPoint(copy);
    alpine::QuantizeOp::create(rewriter, 
        loc, srcForOp, dstForOp, q.getScaleAttr(),
        rewriter.getI32IntegerAttr((int32_t)z64));

    rewriter.eraseOp(copy);
    if (bar->use_empty())
      rewriter.eraseOp(bar);
    if (q->use_empty())
      rewriter.eraseOp(q);

    return success();
  }
};

struct LowerCimDequantizeToAlpine : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto bar = copy.getSource().getDefiningOp<cim::BarrierOp>();
    if (!bar)
      return rewriter.notifyMatchFailure(copy, "src is not cim.barrier");

    Value fut = bar->getOperand(0);
    auto dq = dyn_cast_or_null<cim::DequantizeOp>(fut.getDefiningOp());
    if (!dq)
      return rewriter.notifyMatchFailure(copy,
                                         "barrier is not of cim.op.dequantize");

    if (isRelowerTrue(dq))
      return rewriter.notifyMatchFailure(copy, "dequantize relower=true");

    Value dst = copy.getTarget();
    auto dstMR = dyn_cast<MemRefType>(dst.getType());
    if (!dstMR)
      return rewriter.notifyMatchFailure(copy, "dst must be a memref");

    auto zpAttr = dq.getZeroPointAttr();
    if (!zpAttr)
      return rewriter.notifyMatchFailure(copy, "missing zeroPoint attr");
    int64_t z64 = zpAttr.getInt();
    if (z64 < -128 || z64 > 127)
      return rewriter.notifyMatchFailure(
          copy, "zeroPoint out of i8 range [-128, 127]");

    Location loc = copy.getLoc();

    Value srcForOp = dq.getSrc();
    MemRefType srcTy = dyn_cast<MemRefType>(srcForOp.getType());
    if (!srcTy)
      return rewriter.notifyMatchFailure(copy, "src must be memref");

    if (auto altSrcTy = dyn_cast<MemRefType>(copy.getSource().getType())) {
      if (altSrcTy.getRank() == srcTy.getRank() && altSrcTy != srcTy &&
          memref::CastOp::areCastCompatible(srcTy, altSrcTy)) {
        srcForOp = memref::CastOp::create(rewriter, loc, altSrcTy, srcForOp);
        srcTy = altSrcTy;
      }
    }

    Value dstForOp = dst;
    if (srcTy.getRank() == dstMR.getRank()) {
      SmallVector<int64_t> shape(srcTy.getShape().begin(), srcTy.getShape().end());
      auto expectedDstTy =
          MemRefType::get(shape, dstMR.getElementType(), dstMR.getLayout(),
                          dstMR.getMemorySpace());
      if (expectedDstTy != dstMR &&
          memref::CastOp::areCastCompatible(dstMR, expectedDstTy)) {
        dstForOp = memref::CastOp::create(rewriter, loc, expectedDstTy, dst);
      }
    }

    rewriter.setInsertionPoint(copy);
    alpine::DequantizeOp::create(rewriter, 
        loc, srcForOp, dstForOp, dq.getScaleAttr(),
        rewriter.getI32IntegerAttr((int32_t)z64));

    rewriter.eraseOp(copy);
    if (bar->use_empty())
      rewriter.eraseOp(bar);
    if (dq->use_empty())
      rewriter.eraseOp(dq);

    return success();
  }
};

struct LowerCimGemvChainToAlpine : OpRewritePattern<memref::CopyOp> {
  LowerCimGemvChainToAlpine(MLIRContext *ctx, XbarToTileMap &m)
      : OpRewritePattern(ctx), xbarToTile(m) {}

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto bar = copy.getSource().getDefiningOp<cim::BarrierOp>();
    if (!bar)
      return rewriter.notifyMatchFailure(copy, "src is not cim.barrier");

    Value fut = bar->getOperand(0);
    auto gemv = dyn_cast_or_null<cim::GemvOp>(fut.getDefiningOp());
    if (!gemv)
      return rewriter.notifyMatchFailure(copy,
                                         "barrier is not of cim.gemv future");

    if (isRelowerTrue(gemv))
      return rewriter.notifyMatchFailure(copy, "gemv relower=true");

    Value dst = copy.getTarget();
    auto dstMR = dyn_cast<MemRefType>(dst.getType());
    if (!dstMR || dstMR.getRank() != 1)
      return rewriter.notifyMatchFailure(copy, "dst must be rank-1 memref");

    Location loc = copy.getLoc();

    Value xb = gemv.getOperand(0);
    auto it = xbarToTile.find(xb);
    if (it == xbarToTile.end())
      return rewriter.notifyMatchFailure(copy,
                                         "no alpine.alloc_tile for crossbar");

    Value tile = it->second;

    Value a = gemv.getOperand(1);
    Value b = gemv.getOperand(2);
    auto aSh = cast<ShapedType>(a.getType());
    Value mat = (aSh.getRank() == 2) ? a : b;
    Value vec = (aSh.getRank() == 1) ? a : b;

    rewriter.setInsertionPoint(copy);

    alpine::WriteWeightsOp::create(rewriter, loc, tile, mat);

    alpine::EnqueueVecOp::create(rewriter, loc, tile, vec);

    (void)alpine::ProcessOp::create(rewriter, 
        loc,
        tile,
        StringAttr(),
        rewriter.getBoolAttr(false),
        rewriter.getI64IntegerAttr(1));

    alpine::DequeueVecOp::create(rewriter, loc, tile, dst);

    rewriter.eraseOp(copy);

    if (bar->use_empty())
      rewriter.eraseOp(bar);
    if (gemv->use_empty())
      rewriter.eraseOp(gemv);

    return success();
  }

private:
  XbarToTileMap &xbarToTile;
};

struct LowerCimReluToAlpine : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    auto bar = copy.getSource().getDefiningOp<cim::BarrierOp>();
    if (!bar)
      return rewriter.notifyMatchFailure(copy, "src is not cim.barrier");

    Value fut = bar->getOperand(0);
    auto act = dyn_cast_or_null<cim::ActivateOp>(fut.getDefiningOp());
    if (!act)
      return rewriter.notifyMatchFailure(copy,
                                         "barrier is not of cim.op.activate");

    if (isRelowerTrue(act))
      return rewriter.notifyMatchFailure(copy, "activate relower=true");

    if (act.getKind() != cim::ActivationKind::RELU)
      return rewriter.notifyMatchFailure(copy, "activation is not relu");

    auto inMR = dyn_cast<MemRefType>(act.getInput().getType());
    auto dstMR = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!inMR || !dstMR || !inMR.getElementType().isF32() ||
        !dstMR.getElementType().isF32())
      return rewriter.notifyMatchFailure(copy, "expects f32 memrefs");

    Value srcForOp = act.getInput();
    Value dst = copy.getTarget();
    Value dstForOp = dst;
    if (inMR.getRank() == dstMR.getRank()) {
      SmallVector<int64_t> shape(inMR.getShape().begin(), inMR.getShape().end());
      auto expectedDstTy = MemRefType::get(shape, dstMR.getElementType(),
                                           inMR.getLayout(),
                                           dstMR.getMemorySpace());
      if (expectedDstTy != dstMR &&
          memref::CastOp::areCastCompatible(dstMR, expectedDstTy))
        dstForOp = memref::CastOp::create(rewriter, copy.getLoc(), expectedDstTy,
                                                   dst);
    }

    rewriter.setInsertionPoint(copy);
    alpine::ReluOp::create(rewriter, copy.getLoc(), srcForOp, dstForOp);

    rewriter.eraseOp(copy);
    if (bar->use_empty())
      rewriter.eraseOp(bar);
    if (act->use_empty())
      rewriter.eraseOp(act);
    return success();
  }
};

}

struct ConvertCimToAlpine
    : public mlir::cim::impl::ConvertCimToAlpinePassBase<ConvertCimToAlpine> {
  using Base::Base;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                    mlir::alpine::AlpineDialect>();
  }

  void runOnOperation() override {
    XbarToTileMap xbarToTile;

    preAllocateTiles(getOperation(), xbarToTile);

    mlir::MLIRContext &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<LowerCimGemvChainToAlpine>(&ctx, xbarToTile);

    patterns.add<LowerCimQuantizeToAlpine>(&ctx);
    patterns.add<LowerCimDequantizeToAlpine>(&ctx);

    patterns.add<LowerCimReluToAlpine>(&ctx);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

namespace mlir::cim {
std::unique_ptr<Pass> createConvertCimToAlpinePass() {
  return std::make_unique<ConvertCimToAlpine>();
}
void registerCimToAlpinePipeline() {}
}
