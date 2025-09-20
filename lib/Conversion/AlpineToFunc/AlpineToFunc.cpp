//===- AlpineToFunc.cpp ----------------------------------------*- C++ -*-===//

#include "cinm-mlir/Conversion/AlpineToFunc/AlpineToFunc.h"

#include "cinm-mlir/Dialect/Alpine/IR/AlpineDialect.h"
#include "cinm-mlir/Dialect/Alpine/IR/AlpineOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/Support/Casting.h>

#define DEBUG_TYPE "mlir-alpine-to-func"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::alpine;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/AlpinePasses.h.inc"

namespace {

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

static Value castToFullyDynamicMemRef(Value v, PatternRewriter &rewriter) {
  auto mr = dyn_cast<MemRefType>(v.getType());
  if (!mr)
    return v;

  SmallVector<int64_t, 4> dynShape(mr.getRank(), ShapedType::kDynamic);
  auto dynMr = MemRefType::get(dynShape, mr.getElementType(), mr.getLayout(),
                               mr.getMemorySpace());
  return rewriter.create<memref::CastOp>(v.getLoc(), dynMr, v);
}

static void ensureCallee(StringRef fnName, ArrayRef<Type> paramTypes,
                         ArrayRef<Type> resultTypes, PatternRewriter &rewriter,
                         ModuleOp parentModule, Location loc) {
  if (parentModule.lookupSymbol<func::FuncOp>(fnName))
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  Operation *end = &parentModule.getBodyRegion().front().back();
  rewriter.setInsertionPoint(end);

  auto fTy = FunctionType::get(rewriter.getContext(), paramTypes, resultTypes);
  auto func = rewriter.create<func::FuncOp>(loc, fnName, fTy);
  func.setVisibility(func::FuncOp::Visibility::Nested);
}

static void collectCastedOperands(Operation *op, PatternRewriter &rewriter,
                                  SmallVectorImpl<Value> &args,
                                  SmallVectorImpl<Type> &argTys) {
  for (Value operand : op->getOperands()) {
    Value c = castToFullyDynamicMemRef(operand, rewriter);
    args.push_back(c);
    argTys.push_back(c.getType());
  }
}

static LogicalResult createPlainLibraryCall(Operation *op,
                                            std::string calleeStr,
                                            PatternRewriter &rewriter) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitOpError("expected parent module");

  StringAttr calleeAttr = rewriter.getStringAttr(calleeStr);

  SmallVector<Value> args;
  SmallVector<Type> argTys;
  collectCastedOperands(op, rewriter, args, argTys);

  SmallVector<Type> resTys(op->getResultTypes().begin(),
                           op->getResultTypes().end());

  ensureCallee(calleeAttr.getValue(), argTys, resTys, rewriter, module,
               op->getLoc());

  auto call = rewriter.create<func::CallOp>(op->getLoc(), calleeAttr.getValue(),
                                            resTys, args);

  if (resTys.empty()) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, call.getResults());
  }
  return success();
}

static std::string withRankSuffix(StringRef base, Value firstMemrefOperand) {
  unsigned rank = 0;
  if (auto mr = dyn_cast<MemRefType>(firstMemrefOperand.getType()))
    rank = mr.getRank();
  std::string s = base.str();
  s += "_r";
  s += std::to_string(rank);
  return s;
}

//------------------------------------------------------------------------------
// Patterns
//------------------------------------------------------------------------------

struct AllocTileOpConversion : OpRewritePattern<alpine::AllocTileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(alpine::AllocTileOp op,
                                PatternRewriter &rewriter) const override {
    constexpr llvm::StringLiteral kName("alpine_alloc_tile");

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");

    Location loc = op.getLoc();
    StringAttr calleeAttr = rewriter.getStringAttr(kName);

    int64_t h = 0, w = 0;
    if (auto harr = op->getAttrOfType<IntegerAttr>("height"))
      h = harr.getInt();
    if (auto warr = op->getAttrOfType<IntegerAttr>("width"))
      w = warr.getInt();

    Value hConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(h));
    Value wConst =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(w));

    SmallVector<Value> args{hConst, wConst};
    SmallVector<Type> argTys{hConst.getType(), wConst.getType()};
    SmallVector<Type> resTys{op.getType()}; // i32/i64 tile id

    ensureCallee(calleeAttr.getValue(), argTys, resTys, rewriter, module, loc);

    auto call =
        rewriter.create<func::CallOp>(loc, calleeAttr.getValue(), resTys, args);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct ReluOpConversion : OpRewritePattern<alpine::ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(alpine::ReluOp op,
                                PatternRewriter &rewriter) const override {
    std::string base = op.getLibraryCallName();
    if (base.empty())
      base = "alpine_relu";
    std::string callee = withRankSuffix(base, op->getOperand(0));
    return createPlainLibraryCall(op, std::move(callee), rewriter);
  }
};

struct QuantizeOpConversion : OpRewritePattern<alpine::QuantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(alpine::QuantizeOp op,
                                PatternRewriter &rewriter) const override {
    std::string base = op.getLibraryCallName();
    if (base.empty())
      base = "alpine_quantize";
    std::string callee = withRankSuffix(base, op->getOperand(0));

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");

    StringAttr calleeAttr = rewriter.getStringAttr(callee);

    SmallVector<Value> args;
    SmallVector<Type> argTys;

    for (Value v : op->getOperands().take_front(2)) {
      Value c = castToFullyDynamicMemRef(v, rewriter);
      args.push_back(c);
      argTys.push_back(c.getType());
    }

    Location loc = op.getLoc();
    Value scaleC = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), op.getScaleAttr());
    Value zeroC = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(),
                                                     op.getZeroAttr());

    args.push_back(scaleC);
    argTys.push_back(scaleC.getType());
    args.push_back(zeroC);
    argTys.push_back(zeroC.getType());

    ensureCallee(calleeAttr.getValue(), argTys, /*resTys=*/{}, rewriter, module,
                 loc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeAttr.getValue(),
                                              TypeRange{}, args);
    return success();
  }
};

struct DequantizeOpConversion : OpRewritePattern<alpine::DequantizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(alpine::DequantizeOp op,
                                PatternRewriter &rewriter) const override {
    std::string base = op.getLibraryCallName();
    if (base.empty())
      base = "alpine_dequantize";
    std::string callee = withRankSuffix(base, op->getOperand(0));

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");

    StringAttr calleeAttr = rewriter.getStringAttr(callee);

    SmallVector<Value> args;
    SmallVector<Type> argTys;

    for (Value v : op->getOperands().take_front(2)) {
      Value c = castToFullyDynamicMemRef(v, rewriter);
      args.push_back(c);
      argTys.push_back(c.getType());
    }

    Location loc = op.getLoc();
    Value scaleC = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), op.getScaleAttr());
    Value zeroC = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(),
                                                     op.getZeroAttr());

    args.push_back(scaleC);
    argTys.push_back(scaleC.getType());
    args.push_back(zeroC);
    argTys.push_back(zeroC.getType());

    ensureCallee(calleeAttr.getValue(), argTys, /*resTys=*/{}, rewriter, module,
                 loc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeAttr.getValue(),
                                              TypeRange{}, args);
    return success();
  }
};

static void extract1D(Value mr, PatternRewriter &rewriter, Value &base,
                      Value &offset, Value &len, Value &stride) {
  Location loc = mr.getLoc();
  auto em = rewriter.create<memref::ExtractStridedMetadataOp>(loc, mr);
  base = em.getBaseBuffer();
  offset = em.getOffset();
  auto sizes = em.getSizes();
  auto strides = em.getStrides();
  len = sizes.front();
  stride = strides.front();
}

static void extract2D(Value mr, PatternRewriter &rewriter, Value &base,
                      Value &offset, Value &rows, Value &cols, Value &rowStride,
                      Value &colStride) {
  Location loc = mr.getLoc();
  auto em = rewriter.create<memref::ExtractStridedMetadataOp>(loc, mr);
  base = em.getBaseBuffer();
  offset = em.getOffset();
  auto sizes = em.getSizes();
  auto strides = em.getStrides();
  rows = sizes[0];
  cols = sizes[1];
  rowStride = strides[0];
  colStride = strides[1];
}

static Value i64C(PatternRewriter &rw, Location loc, int64_t v) {
  return rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(v));
}

struct WriteWeightsOpConversion : OpRewritePattern<alpine::WriteWeightsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(alpine::WriteWeightsOp op,
                                PatternRewriter &rewriter) const override {
    std::string callee = op.getLibraryCallName();
    if (callee.empty())
      return op->emitOpError("No library call defined");

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");

    Location loc = op.getLoc();
    Value tile = op.getOperand(0);
    if (!tile.getType().isInteger(64))
      tile = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), tile);

    Value base, offset, rows, cols, rs, cs;
    extract2D(op.getOperand(1), rewriter, base, offset, rows, cols, rs, cs);

    SmallVector<Value> args{tile,
                            base,
                            rows,
                            cols,
                            i64C(rewriter, loc, 0),
                            i64C(rewriter, loc, 0),
                            i64C(rewriter, loc, 0)};
    SmallVector<Type> argTys;
    for (Value a : args)
      argTys.push_back(a.getType());

    StringAttr calleeAttr = rewriter.getStringAttr(callee);
    ensureCallee(calleeAttr.getValue(), argTys, /*resTys=*/{}, rewriter, module,
                 loc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeAttr.getValue(),
                                              TypeRange{}, args);
    return success();
  }
};

struct EnqueueVecOpConversion : OpRewritePattern<alpine::EnqueueVecOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(alpine::EnqueueVecOp op,
                                PatternRewriter &rewriter) const override {
    std::string callee = op.getLibraryCallName();
    if (callee.empty())
      return op->emitOpError("No library call defined");
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");
    Location loc = op.getLoc();

    Value tile = op.getOperand(0);
    if (!tile.getType().isInteger(64))
      tile = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), tile);

    Value base, offset, len, stride;
    extract1D(op.getOperand(1), rewriter, base, offset, len, stride);

    SmallVector<Value> args{tile, base, len, i64C(rewriter, loc, 0), offset};
    SmallVector<Type> argTys;
    for (Value a : args)
      argTys.push_back(a.getType());
    StringAttr calleeAttr = rewriter.getStringAttr(callee);
    ensureCallee(calleeAttr.getValue(), argTys, /*resTys=*/{}, rewriter, module,
                 loc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeAttr.getValue(),
                                              TypeRange{}, args);
    return success();
  }
};

struct DequeueVecOpConversion : OpRewritePattern<alpine::DequeueVecOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(alpine::DequeueVecOp op,
                                PatternRewriter &rewriter) const override {
    std::string callee = op.getLibraryCallName();
    if (callee.empty())
      return op->emitOpError("No library call defined");
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");
    Location loc = op.getLoc();

    Value tile = op.getOperand(0);
    if (!tile.getType().isInteger(64))
      tile = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), tile);

    Value base, offset, len, stride;
    extract1D(op.getOperand(1), rewriter, base, offset, len, stride);

    SmallVector<Value> args{tile, base, len, i64C(rewriter, loc, 0), offset};
    SmallVector<Type> argTys;
    for (Value a : args)
      argTys.push_back(a.getType());
    StringAttr calleeAttr = rewriter.getStringAttr(callee);
    ensureCallee(calleeAttr.getValue(), argTys, /*resTys=*/{}, rewriter, module,
                 loc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeAttr.getValue(),
                                              TypeRange{}, args);
    return success();
  }
};

struct MVMOpConversion : OpRewritePattern<alpine::MVMOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(alpine::MVMOp op,
                                PatternRewriter &rewriter) const override {
    std::string callee = op.getLibraryCallName();
    if (callee.empty())
      return op->emitOpError("No library call defined");
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return op->emitOpError("expected parent module");
    Location loc = op.getLoc();

    Value tile = op.getOperand(0);
    if (!tile.getType().isInteger(64))
      tile = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), tile);

    Value inBase, inOff, inLen, inStride;
    extract1D(op.getOperand(1), rewriter, inBase, inOff, inLen, inStride);
    Value outBase, outOff, outLen, outStride;
    extract1D(op.getOperand(2), rewriter, outBase, outOff, outLen, outStride);

    SmallVector<Value> args{tile,  inBase,  inLen,  i64C(rewriter, loc, 0),
                            inOff, outBase, outLen, i64C(rewriter, loc, 0),
                            outOff};
    SmallVector<Type> argTys;
    for (Value a : args)
      argTys.push_back(a.getType());
    StringAttr calleeAttr = rewriter.getStringAttr(callee);
    ensureCallee(calleeAttr.getValue(), argTys, /*resTys=*/{}, rewriter, module,
                 loc);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, calleeAttr.getValue(),
                                              TypeRange{}, args);
    return success();
  }
};

struct ProcessOpConversion : OpRewritePattern<alpine::ProcessOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(alpine::ProcessOp op,
                                PatternRewriter &rewriter) const override {
    std::string callee = op.getLibraryCallName();
    if (callee.empty())
      return op->emitOpError("No library call defined for op");
    return createPlainLibraryCall(op, std::move(callee), rewriter);
  }
};

//------------------------------------------------------------------------------
// Pass driver
//------------------------------------------------------------------------------

struct ConvertAlpineToFunc
    : public ConvertAlpineToFuncBase<ConvertAlpineToFunc> {
  void runOnOperation() final {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns{&ctx};
    populateAlpineToFuncConversionPatterns(patterns, &ctx);

    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalDialect<AlpineDialect>();

    target.addLegalDialect<FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    target.addLegalOp<FuncOp, CallOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//------------------------------------------------------------------------------
// Pattern population
//------------------------------------------------------------------------------

void mlir::alpine::populateAlpineToFuncConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ReluOpConversion, QuantizeOpConversion, DequantizeOpConversion,
               AllocTileOpConversion, WriteWeightsOpConversion,
               EnqueueVecOpConversion, ProcessOpConversion,
               DequeueVecOpConversion, MVMOpConversion>(ctx);
}

std::unique_ptr<Pass> mlir::alpine::createConvertAlpineToFuncPass() {
  return std::make_unique<ConvertAlpineToFunc>();
}
