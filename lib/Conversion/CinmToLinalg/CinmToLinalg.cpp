#include "cinm-mlir/Conversion/CinmToLinalg/CinmToLinalg.h"
#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace mlir::cinm;

namespace mlir::cinm {

#define GEN_PASS_DEF_CONVERTCINMOPSTOLINALG
#include "cinm-mlir/Conversion/CinmPasses.h.inc"
} // namespace mlir::cinm

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value buildEmpty(OpBuilder &b, Location loc, RankedTensorType type,
                        Value exemplar) {
  SmallVector<Value> dynDims;
  for (int64_t i = 0, e = type.getRank(); i < e; ++i)
    if (type.isDynamicDim(i))
      dynDims.push_back(tensor::DimOp::create(b, loc, exemplar, i));
  return tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType(),
                                 dynDims);
}

static Value buildZero(OpBuilder &b, Location loc, Type elemType) {
  if (isa<FloatType>(elemType))
    return arith::ConstantOp::create(
        b, loc,
        FloatAttr::get(
            elemType,
            APFloat::getZero(cast<FloatType>(elemType).getFloatSemantics())));
  return arith::ConstantIntOp::create(b, loc, elemType, 0);
}

static Value buildReduceIdentity(OpBuilder &b, Location loc,
                                 ReduceMethod method, Type elemType) {
  auto arithConst = cinm::getArithConstant(method, elemType);
  return arith::getIdentityValue(arithConst, elemType, b, loc);
}

static Value emitReduceCombine(OpBuilder &b, Location loc, ReduceMethod method,
                               Value elem, Value acc, Type elemType) {
  auto arithConst = cinm::getArithConstant(method, elemType);
  return arith::getReductionOp(arithConst, b, loc, elem, acc);
}

// Build the outs init for a gemm-like op, in priority order:
//   1. tensor out (DPS hint — result aliases the out buffer)
//   2. bias (accumulator init)
//   3. zero-filled fresh tensor
static Value buildGemmInit(OpBuilder &b, Location loc, Value out, Value bias,
                           RankedTensorType resultTy) {
  if (out && isa<RankedTensorType>(out.getType()))
    return out;
  if (bias)
    return bias;
  Value empty = tensor::EmptyOp::create(b, loc, resultTy.getShape(),
                                        resultTy.getElementType());
  return linalg::FillOp::create(
             b, loc, buildZero(b, loc, resultTy.getElementType()), empty)
      .getResult(0);
}

// Build a linalg.generic for an N-ary elementwise op.
// If `tensorOut` is non-null it is used as the outs init (DPS hint); otherwise
// a fresh empty tensor is allocated.  bodyBuilder receives the scalar input
// args (without the output arg) and must return one Value.
static Value buildElementwiseGeneric(
    OpBuilder &b, Location loc, RankedTensorType resultTy, ValueRange inputs,
    Value tensorOut,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  int64_t rank = resultTy.getRank();
  AffineMap id = b.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> maps(inputs.size() + 1, id);

  SmallVector<Attribute> iterAttrs(
      rank, linalg::IteratorTypeAttr::get(b.getContext(),
                                          utils::IteratorType::parallel));
  auto iterAttr = b.getArrayAttr(iterAttrs);
  auto mapsAttr = b.getAffineMapArrayAttr(maps);

  Value init = tensorOut ? tensorOut : buildEmpty(b, loc, resultTy, inputs[0]);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{resultTy}, inputs, ValueRange{init}, mapsAttr, iterAttr,
      StringAttr{}, StringAttr{},
      [&](OpBuilder &nb, Location nloc, ValueRange args) {
        Value out = bodyBuilder(nb, nloc, args.drop_back());
        linalg::YieldOp::create(nb, nloc, out);
      },
      ArrayRef<NamedAttribute>{});
  return generic.getResult(0);
}

/// Replace a `cinm` op with the linalg op carrying its computation, keeping
/// the discardable attributes.
///
/// Those attributes are how decisions travel down the pipeline: the inference
/// plugin stamps `cnm.tile_sizes` / `upmem.leaf_tile_sizes` on a cinm op, and
/// the passes that consume them run well after this conversion. Dropping them
/// here would silently lose the configuration, so they move onto the op that
/// inherits the same iteration space. Ops produced alongside it (an init
/// `linalg.fill`, a `tensor.empty`) deliberately do not get them.
static void replaceWithLinalgOp(ConversionPatternRewriter &rewriter,
                                Operation *op, Value result) {
  if (Operation *producer = result.getDefiningOp())
    if (isa<linalg::LinalgOp>(producer)) {
      producer->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      // Record which cinm op this came from. Consumers need to tell the op
      // carrying the computation apart from the ops produced around it -- an
      // init `linalg.fill` is a LinalgOp too, and giving it tile sizes or
      // search parameters would be nonsense.
      producer->setAttr(cinm::CinmDialect::DEBUG_TAG_NAME,
                        rewriter.getStringAttr(op->getName().getStringRef()));
    }
  rewriter.replaceOp(op, result);
}

//===----------------------------------------------------------------------===//
// cinm.op.elementwise
//===----------------------------------------------------------------------===//

static FailureOr<Value> emitElementwiseScalar(OpBuilder &b, Location loc,
                                              ElementwiseKind kind, Value lhs,
                                              Value rhs, Type elemTy) {
  bool isFloat = isa<FloatType>(elemTy);
  bool isInt = isa<IntegerType>(elemTy);
  bool isUnary = !rhs;

  // Binary
  if (!isUnary) {
    switch (kind) {
    case ElementwiseKind::Add:
      if (isFloat)
        return arith::AddFOp::create(b, loc, lhs, rhs).getResult();
      if (isInt)
        return arith::AddIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::Sub:
      if (isFloat)
        return arith::SubFOp::create(b, loc, lhs, rhs).getResult();
      if (isInt)
        return arith::SubIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::Mul:
      if (isFloat)
        return arith::MulFOp::create(b, loc, lhs, rhs).getResult();
      if (isInt)
        return arith::MulIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::Div:
      if (isFloat)
        return arith::DivFOp::create(b, loc, lhs, rhs).getResult();
      if (isInt)
        return arith::DivSIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::Mod:
      if (isFloat)
        return arith::RemFOp::create(b, loc, lhs, rhs).getResult();
      if (isInt)
        return arith::RemSIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::And:
      if (isInt)
        return arith::AndIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::Or:
      if (isInt)
        return arith::OrIOp::create(b, loc, lhs, rhs).getResult();
      break;
    case ElementwiseKind::Xor:
      if (isInt)
        return arith::XOrIOp::create(b, loc, lhs, rhs).getResult();
      break;
    default:
      break;
    }
    return failure();
  }

  // Unary
  switch (kind) {
  case ElementwiseKind::Neg:
    if (isFloat)
      return arith::NegFOp::create(b, loc, lhs).getResult();
    if (isInt) {
      Value zero = arith::ConstantIntOp::create(b, loc, elemTy, 0);
      return arith::SubIOp::create(b, loc, zero, lhs).getResult();
    }
    break;
  case ElementwiseKind::Abs:
    if (isFloat)
      return math::AbsFOp::create(b, loc, lhs).getResult();
    if (isInt)
      return math::AbsIOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Ceil:
    if (isFloat)
      return math::CeilOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Erf:
    if (isFloat)
      return math::ErfOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Exp:
    if (isFloat)
      return math::ExpOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Floor:
    if (isFloat)
      return math::FloorOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Log:
    if (isFloat)
      return math::LogOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Reciprocal: {
    if (isFloat) {
      Value one =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elemTy, 1.0));
      return arith::DivFOp::create(b, loc, one, lhs).getResult();
    }
    break;
  }
  case ElementwiseKind::Round:
    if (isFloat)
      return math::RoundOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Rsqrt:
    if (isFloat)
      return math::RsqrtOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Sqrt:
    if (isFloat)
      return math::SqrtOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Square:
    if (isFloat)
      return arith::MulFOp::create(b, loc, lhs, lhs).getResult();
    if (isInt)
      return arith::MulIOp::create(b, loc, lhs, lhs).getResult();
    break;
  case ElementwiseKind::Tanh:
    if (isFloat)
      return math::TanhOp::create(b, loc, lhs).getResult();
    break;
  case ElementwiseKind::Not:
    if (isInt) {
      Value allOnes = arith::ConstantIntOp::create(b, loc, elemTy, -1);
      return arith::XOrIOp::create(b, loc, lhs, allOnes).getResult();
    }
    break;
  case ElementwiseKind::Relu: {
    if (isFloat) {
      Value zero =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elemTy, 0.0));
      return arith::MaxNumFOp::create(b, loc, lhs, zero).getResult();
    }
    break;
  }
  case ElementwiseKind::Sigmoid: {
    if (isFloat) {
      // 1 / (1 + exp(-x))
      Value neg = arith::NegFOp::create(b, loc, lhs);
      Value e = math::ExpOp::create(b, loc, neg);
      Value one =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elemTy, 1.0));
      Value denom = arith::AddFOp::create(b, loc, one, e);
      return arith::DivFOp::create(b, loc, one, denom).getResult();
    }
    break;
  }
  case ElementwiseKind::Gelu: {
    if (isFloat) {
      // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
      Value half =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elemTy, 0.5));
      Value c =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elemTy, 0.044715));
      Value s2pi = arith::ConstantOp::create(
          b, loc, FloatAttr::get(elemTy, 0.7978845608));
      Value one =
          arith::ConstantOp::create(b, loc, FloatAttr::get(elemTy, 1.0));
      Value v2 = arith::MulFOp::create(b, loc, lhs, lhs);
      Value v3 = arith::MulFOp::create(b, loc, v2, lhs);
      Value inner = arith::AddFOp::create(b, loc, lhs,
                                          arith::MulFOp::create(b, loc, c, v3));
      Value t = math::TanhOp::create(
          b, loc, arith::MulFOp::create(b, loc, s2pi, inner));
      return arith::MulFOp::create(
                 b, loc, half,
                 arith::MulFOp::create(b, loc, lhs,
                                       arith::AddFOp::create(b, loc, one, t)))
          .getResult();
    }
    break;
  }
  default:
    break;
  }
  return failure();
}

struct ConvertElementwiseToLinalg
    : public OpConversionPattern<cinm::ElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle the tensor form (result present, no memref out).
    if (!op.getResult())
      return failure();

    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = resultTy.getElementType();
    auto kind = op.getKind();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = adaptor.getOut();
    Value tensorOut =
        (out && isa<RankedTensorType>(out.getType())) ? out : Value{};

    SmallVector<Value> inputs =
        rhs ? SmallVector<Value>{lhs, rhs} : SmallVector<Value>{lhs};

    bool failed = false;
    Value result = buildElementwiseGeneric(
        rewriter, loc, resultTy, inputs, tensorOut,
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          auto out = emitElementwiseScalar(b, loc, kind, args[0],
                                           args.size() > 1 ? args[1] : Value{},
                                           elemTy);
          if (::mlir::failed(out)) {
            failed = true;
            return args[0]; // placeholder
          }
          return *out;
        });

    if (failed)
      return op.emitError(
          "unsupported elementwise kind/element-type combination");

    replaceWithLinalgOp(rewriter, op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cinm.op.reduce
//===----------------------------------------------------------------------===//

struct ConvertReduceToLinalg : public OpConversionPattern<cinm::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode is not handled here, same as the gemm-like patterns below.
    if (!op.getResult())
      return failure();

    auto loc = op.getLoc();
    auto inputTy = cast<RankedTensorType>(adaptor.getInput().getType());
    Type elemTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();

    // Normalize negative dimension indices.
    SmallVector<int64_t> dims;
    for (int64_t d : {op.getDimension()})
      dims.push_back(d < 0 ? rank + d : d);

    // Output shape: input shape with reduced dims removed.
    SmallVector<int64_t> outputShape;
    for (int64_t i = 0; i < rank; ++i)
      if (!llvm::is_contained(dims, i))
        outputShape.push_back(inputTy.getDimSize(i));

    bool isScalarResult = !isa<ShapedType>(op.getResult().getType());

    Value identity = buildReduceIdentity(rewriter, loc, op.getMethod(), elemTy);
    Value initTensor =
        tensor::EmptyOp::create(rewriter, loc, outputShape, elemTy);
    Value filledInit =
        linalg::FillOp::create(rewriter, loc, identity, initTensor)
            .getResult(0);

    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{adaptor.getInput()}, ValueRange{filledInit},
        dims, [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = element, args[1] = accumulator
          Value combined = emitReduceCombine(b, loc, op.getMethod(), args[0],
                                             args[1], elemTy);
          linalg::YieldOp::create(b, loc, combined);
        });

    Value result = reduceOp.getResult(0);
    if (isScalarResult)
      result = tensor::ExtractOp::create(rewriter, loc, result, ValueRange{});

    replaceWithLinalgOp(rewriter, op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cinm.op.gemv
//===----------------------------------------------------------------------===//

struct ConvertGemvToLinalg : public OpConversionPattern<cinm::GemvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::GemvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult())
      return failure();

    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    Value init = buildGemmInit(rewriter, loc, adaptor.getOut(),
                               adaptor.getBias(), resultTy);

    Value result =
        linalg::MatvecOp::create(rewriter, loc, TypeRange{resultTy},
                                 ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                                 ValueRange{init})
            .getResult(0);
    replaceWithLinalgOp(rewriter, op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cinm.op.gemm
//===----------------------------------------------------------------------===//

struct ConvertGemmToLinalg : public OpConversionPattern<cinm::GemmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult())
      return failure();

    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    Value init = buildGemmInit(rewriter, loc, adaptor.getOut(),
                               adaptor.getBias(), resultTy);

    Value result =
        linalg::MatmulOp::create(rewriter, loc, TypeRange{resultTy},
                                 ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                                 ValueRange{init})
            .getResult(0);
    replaceWithLinalgOp(rewriter, op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cinm.op.batch_gemm
//===----------------------------------------------------------------------===//

struct ConvertBatchGemmToLinalg
    : public OpConversionPattern<cinm::BatchGemmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::BatchGemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult())
      return failure();

    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    Value init = buildGemmInit(rewriter, loc, adaptor.getOut(),
                               adaptor.getBias(), resultTy);

    Value result =
        linalg::BatchMatmulOp::create(
            rewriter, loc, TypeRange{resultTy},
            ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init})
            .getResult(0);
    replaceWithLinalgOp(rewriter, op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cinm.op.batch_gemv   (B x M x K) * (B x K) -> (B x M)
//===----------------------------------------------------------------------===//

struct ConvertBatchGemvToLinalg
    : public OpConversionPattern<cinm::BatchGemvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::BatchGemvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResult())
      return failure();

    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    Type elemTy = resultTy.getElementType();
    bool isFloat = isa<FloatType>(elemTy);

    auto *ctx = rewriter.getContext();
    AffineExpr b = getAffineDimExpr(0, ctx);
    AffineExpr m = getAffineDimExpr(1, ctx);
    AffineExpr k = getAffineDimExpr(2, ctx);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {b, m, k}, ctx), // lhs (B x M x K)
        AffineMap::get(3, 0, {b, k}, ctx),    // rhs (B x K)
        AffineMap::get(3, 0, {b, m}, ctx),    // out (B x M)
    };
    SmallVector<Attribute> iterAttrs = {
        linalg::IteratorTypeAttr::get(ctx, utils::IteratorType::parallel),
        linalg::IteratorTypeAttr::get(ctx, utils::IteratorType::parallel),
        linalg::IteratorTypeAttr::get(ctx, utils::IteratorType::reduction),
    };

    Value init = buildGemmInit(rewriter, loc, adaptor.getOut(),
                               adaptor.getBias(), resultTy);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultTy},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init},
        rewriter.getAffineMapArrayAttr(maps), rewriter.getArrayAttr(iterAttrs),
        StringAttr{}, StringAttr{},
        [&](OpBuilder &nb, Location nloc, ValueRange args) {
          Value mul = isFloat
                          ? arith::MulFOp::create(nb, nloc, args[0], args[1])
                                .getResult()
                          : arith::MulIOp::create(nb, nloc, args[0], args[1])
                                .getResult();
          Value acc =
              isFloat
                  ? arith::AddFOp::create(nb, nloc, mul, args[2]).getResult()
                  : arith::AddIOp::create(nb, nloc, mul, args[2]).getResult();
          linalg::YieldOp::create(nb, nloc, acc);
        },
        ArrayRef<NamedAttribute>{});

    replaceWithLinalgOp(rewriter, op, generic.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cinm.op.transpose
//===----------------------------------------------------------------------===//

struct ConvertTransposeToLinalg
    : public OpConversionPattern<cinm::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cinm::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto perms = op.getPermutation();

    auto inputTy = cast<RankedTensorType>(adaptor.getInput1().getType());
    SmallVector<int64_t> outputShape;
    for (int64_t p : perms)
      outputShape.push_back(inputTy.getDimSize(p));

    Value init = tensor::EmptyOp::create(rewriter, loc, outputShape,
                                         inputTy.getElementType());
    Value result = linalg::TransposeOp::create(rewriter, loc,
                                               adaptor.getInput1(), init, perms)
                       .getResults()[0];
    replaceWithLinalgOp(rewriter, op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertCinmOpsToLinalg
    : public mlir::cinm::impl::ConvertCinmOpsToLinalgBase<
          ConvertCinmOpsToLinalg> {
  using mlir::cinm::impl::ConvertCinmOpsToLinalgBase<
      ConvertCinmOpsToLinalg>::ConvertCinmOpsToLinalgBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCinmOpsToLinalgPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    // Anything not explicitly illegal is legal (covers scf, arith, tensor, …).
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // All cinm.op.* ops must be lowered.
    target.addIllegalOp<cinm::ElementwiseOp, cinm::ReduceOp, cinm::GemvOp,
                        cinm::GemmOp, cinm::BatchGemmOp, cinm::BatchGemvOp,
                        cinm::TransposeOp>();

    // The container ops and yield are left untouched.
    target.addLegalOp<cinm::ComputeBlockOp, cinm::ComputeOp, cinm::YieldOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

void mlir::cinm::populateCinmOpsToLinalgPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {
  patterns.insert<ConvertElementwiseToLinalg, ConvertReduceToLinalg,
                  ConvertGemvToLinalg, ConvertGemmToLinalg,
                  ConvertBatchGemmToLinalg, ConvertBatchGemvToLinalg,
                  ConvertTransposeToLinalg>(ctx);
}

std::unique_ptr<mlir::Pass> mlir::cinm::createConvertCinmOpsToLinalgPass() {
  return std::make_unique<ConvertCinmOpsToLinalg>();
}
