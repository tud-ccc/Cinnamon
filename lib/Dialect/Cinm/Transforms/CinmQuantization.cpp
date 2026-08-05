#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::cinm {

#define GEN_PASS_DEF_INSERTCINMQUANTIZATION
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

// TODO: Fix Bias in rewriteGem*()

namespace {

static Type parseQuantElemType(StringRef s, MLIRContext &ctx) {
  std::string lowered = s.trim().lower();
  StringRef t(lowered);

  if (t.consume_front("int")) {
    unsigned bits = 0;
    if (t.getAsInteger(10, bits))
      return {};
    return IntegerType::get(&ctx, bits,
                            IntegerType::SignednessSemantics::Signless);
  }
  if (t.starts_with("i")) {
    unsigned bits = 0;
    if (t.drop_front().getAsInteger(10, bits))
      return {};
    return IntegerType::get(&ctx, bits,
                            IntegerType::SignednessSemantics::Signless);
  }
  if (t.starts_with("si")) {
    unsigned bits = 0;
    if (t.drop_front(2).getAsInteger(10, bits))
      return {};
    return IntegerType::get(&ctx, bits,
                            IntegerType::SignednessSemantics::Signed);
  }
  if (t.starts_with("ui")) {
    unsigned bits = 0;
    if (t.drop_front(2).getAsInteger(10, bits))
      return {};
    return IntegerType::get(&ctx, bits,
                            IntegerType::SignednessSemantics::Unsigned);
  }
  return {};
}

static cinm::RoundingMode parseRoundingOrDefault(StringRef s) {
  std::string lowered = s.trim().lower();
  StringRef t(lowered);
  if (t == "towards_zero" || t == "towardszero" || t == "tz")
    return cinm::RoundingMode::TowardsZero;
  return cinm::RoundingMode::Nearest;
}

static llvm::StringSet<> parseOpsList(StringRef s) {
  llvm::StringSet<> out;
  SmallVector<StringRef, 8> parts;
  s.split(parts, ',');
  for (StringRef p : parts) {
    std::string tok = p.trim().lower();
    if (!tok.empty())
      (void)out.insert(tok);
  }
  return out;
}

static bool isFloatTensor(Type t) {
  auto rtt = dyn_cast<RankedTensorType>(t);
  return rtt && isa<FloatType>(rtt.getElementType());
}
static bool isFloatMemRef(Type t) {
  auto mt = dyn_cast<MemRefType>(t);
  return mt && isa<FloatType>(mt.getElementType());
}

static Value buildQuantize(IRRewriter &rewriter, Location loc, Value src,
                           Type qElem, float scale, int64_t zp,
                           cinm::RoundingMode rounding, bool narrow) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto qTy = RankedTensorType::get(srcTy.getShape(), qElem);

  auto fScale = rewriter.getF32FloatAttr(scale);
  auto iZp = rewriter.getI64IntegerAttr(zp);
  auto rAttr = cinm::RoundingModeAttr::get(rewriter.getContext(), rounding);
  auto nAttr = rewriter.getBoolAttr(narrow);

  auto q = cinm::QuantizeOp::create(rewriter, loc, qTy, src, Value(), fScale,
                                    iZp, IntegerAttr{}, rAttr, nAttr);
  return q.getResult();
}

static Value buildDequantize(IRRewriter &rewriter, Location loc, Value srcQ,
                             Type resultFloatElem, float scale, int64_t zp) {
  auto qTy = cast<RankedTensorType>(srcQ.getType());
  auto outTy = RankedTensorType::get(qTy.getShape(), resultFloatElem);
  auto fScale = rewriter.getF32FloatAttr(scale);
  auto iZp = rewriter.getI64IntegerAttr(zp);

  auto dq = cinm::DequantizeOp::create(rewriter, loc, outTy, srcQ, Value(),
                                       fScale, iZp, IntegerAttr{});
  return dq.getResult();
}

static LogicalResult rewriteGemmTensor(GemmOp op, Type qElem, float scale,
                                       int64_t zp, cinm::RoundingMode rounding,
                                       bool narrow, IRRewriter &rewriter) {
  auto outTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!outTy || !isa<FloatType>(outTy.getElementType()))
    return failure();
  if (op.getBias())
    return failure();

  Value A = op.getLhs(), B = op.getRhs();
  if (!isFloatTensor(A.getType()) || !isFloatTensor(B.getType()))
    return failure();

  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Value Aq =
      buildQuantize(rewriter, loc, A, qElem, scale, zp, rounding, narrow);
  Value Bq =
      buildQuantize(rewriter, loc, B, qElem, scale, zp, rounding, narrow);

  auto qGemm = cinm::GemmOp::create(rewriter, loc, Aq, Bq, Value(), Value());
  Value dq = buildDequantize(rewriter, loc, qGemm.getResult(),
                             outTy.getElementType(), scale, zp);

  rewriter.replaceOp(op, dq);
  return success();
}

static LogicalResult rewriteGemvTensor(GemvOp op, Type qElem, float scale,
                                       int64_t zp, cinm::RoundingMode rounding,
                                       bool narrow, IRRewriter &rewriter) {
  auto outTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!outTy || !isa<FloatType>(outTy.getElementType()))
    return failure();
  if (op.getBias())
    return failure();

  Value A = op.getLhs(), x = op.getRhs();
  if (!isFloatTensor(A.getType()) || !isFloatTensor(x.getType()))
    return failure();

  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Value Aq =
      buildQuantize(rewriter, loc, A, qElem, scale, zp, rounding, narrow);
  Value xq =
      buildQuantize(rewriter, loc, x, qElem, scale, zp, rounding, narrow);

  auto qGemv = cinm::GemvOp::create(rewriter, loc, Aq, xq, Value(), Value());
  Value dq = buildDequantize(rewriter, loc, qGemv.getResult(),
                             outTy.getElementType(), scale, zp);

  rewriter.replaceOp(op, dq);
  return success();
}

static Value allocLikeWithElem(IRRewriter &rewriter, Location loc, Value like,
                               Type elemTy) {
  auto mt = cast<MemRefType>(like.getType());
  MemRefType newTy = MemRefType::Builder(mt).setElementType(elemTy);

  SmallVector<Value, 4> dynSizes;
  for (int64_t i = 0; i < newTy.getRank(); ++i)
    if (newTy.isDynamicDim(i))
      dynSizes.push_back(memref::DimOp::create(rewriter, loc, like, i));
  return memref::AllocOp::create(rewriter, loc, newTy, dynSizes).getResult();
}

static void emitQuantizeMemRef(IRRewriter &rewriter, Location loc, Value src,
                               Value dst, float scale, int64_t zp,
                               cinm::RoundingMode rounding, bool narrow) {
  auto fScale = rewriter.getF32FloatAttr(scale);
  auto iZp = rewriter.getI64IntegerAttr(zp);
  auto rAttr = cinm::RoundingModeAttr::get(rewriter.getContext(), rounding);
  auto nAttr = rewriter.getBoolAttr(narrow);
  cinm::QuantizeOp::create(rewriter, loc, Type(), src, dst, fScale, iZp,
                           IntegerAttr{}, rAttr, nAttr);
}

static void emitDequantizeMemRef(IRRewriter &rewriter, Location loc, Value src,
                                 Value dst, float scale, int64_t zp) {
  auto fScale = rewriter.getF32FloatAttr(scale);
  auto iZp = rewriter.getI64IntegerAttr(zp);
  cinm::DequantizeOp::create(rewriter, loc, Type(), src, dst, fScale, iZp,
                             IntegerAttr{});
}

static LogicalResult rewriteGemmMemRef(GemmOp op, Type qElem, float scale,
                                       int64_t zp, cinm::RoundingMode rounding,
                                       bool narrow, IRRewriter &rewriter) {
  Value A = op.getLhs(), B = op.getRhs(), C = op.getOut();
  if (!isFloatMemRef(A.getType()) || !isFloatMemRef(B.getType()) ||
      !isFloatMemRef(C.getType()))
    return failure();

  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Value qA = allocLikeWithElem(rewriter, loc, A, qElem);
  Value qB = allocLikeWithElem(rewriter, loc, B, qElem);
  Value qC = allocLikeWithElem(rewriter, loc, C, qElem);

  emitQuantizeMemRef(rewriter, loc, A, qA, scale, zp, rounding, narrow);
  emitQuantizeMemRef(rewriter, loc, B, qB, scale, zp, rounding, narrow);

  cinm::GemmOp::create(rewriter, loc, qA, qB, Value(), qC);
  emitDequantizeMemRef(rewriter, loc, qC, C, scale, zp);

  memref::DeallocOp::create(rewriter, loc, qA);
  memref::DeallocOp::create(rewriter, loc, qB);
  memref::DeallocOp::create(rewriter, loc, qC);
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult rewriteGemvMemRef(GemvOp op, Type qElem, float scale,
                                       int64_t zp, cinm::RoundingMode rounding,
                                       bool narrow, IRRewriter &rewriter) {
  Value A = op.getLhs(), x = op.getRhs(), y = op.getOut();
  if (!isFloatMemRef(A.getType()) || !isFloatMemRef(x.getType()) ||
      !isFloatMemRef(y.getType()))
    return failure();

  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Value qA = allocLikeWithElem(rewriter, loc, A, qElem);
  Value qx = allocLikeWithElem(rewriter, loc, x, qElem);
  Value qy = allocLikeWithElem(rewriter, loc, y, qElem);

  emitQuantizeMemRef(rewriter, loc, A, qA, scale, zp, rounding, narrow);
  emitQuantizeMemRef(rewriter, loc, x, qx, scale, zp, rounding, narrow);

  cinm::GemvOp::create(rewriter, loc, qA, qx, Value(), qy);
  emitDequantizeMemRef(rewriter, loc, qy, y, scale, zp);

  memref::DeallocOp::create(rewriter, loc, qA);
  memref::DeallocOp::create(rewriter, loc, qx);
  memref::DeallocOp::create(rewriter, loc, qy);
  rewriter.eraseOp(op);
  return success();
}

struct InsertCinmQuantization
    : public impl::InsertCinmQuantizationBase<InsertCinmQuantization> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto want = parseOpsList(ops);
    const bool doGemm = want.contains("gemm");
    const bool doGemv = want.contains("gemv");
    const bool doGemmMR = want.contains("gemm_memref");
    const bool doGemvMR = want.contains("gemv_memref");

    if (!doGemm && !doGemv && !doGemmMR && !doGemvMR)
      return;

    if (scale <= 0.0) {
      func.emitError() << "cinm-insert-quantization: --scale must be > 0";
      signalPassFailure();
      return;
    }

    Type qElem = parseQuantElemType(qElementTypeStr, getContext());
    if (!qElem || !isa<IntegerType>(qElem)) {
      func.emitError() << "cinm-insert-quantization: unsupported --qtype '"
                       << qElementTypeStr
                       << "'; try i8, i16, int8, ui8, si8...";
      signalPassFailure();
      return;
    }

    cinm::RoundingMode mode = parseRoundingOrDefault(roundingStr);
    IRRewriter rewriter(&getContext());

    SmallVector<Operation *, 32> wl;
    func.walk([&](Operation *op) {
      auto compute = op->getParentOfType<cinm::ComputeBlockOp>();
      if (!compute)
        return;
      if (doGemm && isa<cinm::GemmOp>(op))
        wl.push_back(op);
      if (doGemv && isa<cinm::GemvOp>(op))
        wl.push_back(op);
    });

    for (Operation *op : wl) {
      if (auto g = dyn_cast<cinm::GemmOp>(op)) {
        if (dyn_cast<MemRefType>(g.getLhs().getType())) {
          (void)rewriteGemmMemRef(g, qElem, (float)scale, (int64_t)zeroPoint,
                                  mode, (bool)narrowRange, rewriter);
        } else {
          (void)rewriteGemmTensor(g, qElem, (float)scale, (int64_t)zeroPoint,
                                  mode, (bool)narrowRange, rewriter);
        }
        continue;
      }
      if (auto v = dyn_cast<cinm::GemvOp>(op)) {
        if (dyn_cast<MemRefType>(v.getLhs().getType())) {
          (void)rewriteGemvMemRef(v, qElem, (float)scale, (int64_t)zeroPoint,
                                  mode, (bool)narrowRange, rewriter);
        } else {
          (void)rewriteGemvTensor(v, qElem, (float)scale, (int64_t)zeroPoint,
                                  mode, (bool)narrowRange, rewriter);
        }
        continue;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createInsertCinmQuantizationPass() {
  return std::make_unique<InsertCinmQuantization>();
}

} // namespace mlir::cinm
