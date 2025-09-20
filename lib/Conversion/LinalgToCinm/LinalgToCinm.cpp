//===- LinalgToCinm.cpp ---------------------------------------*- C++ -*-===//

#include "cinm-mlir/Conversion/LinalgToCinm/LinalgToCinm.h"

#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CinmPasses.h.inc"

namespace {

static bool isTensor(Value v) { return isa<RankedTensorType>(v.getType()); }
static bool isMemRef(Value v) { return isa<MemRefType>(v.getType()); }

static bool isZeroConst(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    Attribute a = c.getValue();
    if (auto ia = dyn_cast<IntegerAttr>(a))
      return ia.getValue().isZero();
    if (auto fa = dyn_cast<FloatAttr>(a))
      return fa.getValue().isZero();
  }
  return false;
}

static bool isOneConst(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    Attribute a = c.getValue();
    if (auto ia = dyn_cast<IntegerAttr>(a))
      return ia.getValue().isOne();
    if (auto fa = dyn_cast<FloatAttr>(a))
      return fa.getValue().isExactlyValue(1.0);
  }
  return false;
}

static bool isNegOf(Value v, Value x) {
  if (auto negf = v.getDefiningOp<arith::NegFOp>())
    return negf.getOperand() == x;
  if (auto mulf = v.getDefiningOp<arith::MulFOp>()) {
    auto check = [&](Value a, Value b) {
      if (a != x)
        return false;
      auto c = dyn_cast_or_null<arith::ConstantOp>(b.getDefiningOp());
      if (!c)
        return false;
      if (auto fa = dyn_cast<FloatAttr>(c.getValue()))
        return fa.getValue().isExactlyValue(-1.0);
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
        return ia.getInt() == -1;
      return false;
    };
    return check(mulf.getLhs(), mulf.getRhs()) ||
           check(mulf.getRhs(), mulf.getLhs());
  }
  return false;
}

static bool matchRelu(Value yielded, Value x) {
  if (auto maxf = yielded.getDefiningOp<arith::MaximumFOp>()) {
    Value a = maxf.getLhs(), b = maxf.getRhs();
    if ((a == x && isZeroConst(b)) || (b == x && isZeroConst(a)))
      return true;
  }
  if (auto maxnum = yielded.getDefiningOp<arith::MaxNumFOp>()) {
    Value a = maxnum.getLhs(), b = maxnum.getRhs();
    if ((a == x && isZeroConst(b)) || (b == x && isZeroConst(a)))
      return true;
  }
  if (auto maxsi = yielded.getDefiningOp<arith::MaxSIOp>()) {
    Value a = maxsi.getLhs(), b = maxsi.getRhs();
    if ((a == x && isZeroConst(b)) || (b == x && isZeroConst(a)))
      return true;
  }

  auto sel = yielded.getDefiningOp<arith::SelectOp>();
  if (!sel)
    return false;

  auto cond = sel.getCondition();
  auto cmpf = cond.getDefiningOp<arith::CmpFOp>();
  auto cmpi = cond.getDefiningOp<arith::CmpIOp>();

  auto floatPos = [&](arith::CmpFPredicate p, Value lhs, Value rhs) {
    bool direct = lhs == x && isZeroConst(rhs) &&
                  (p == arith::CmpFPredicate::UGT || p == arith::CmpFPredicate::OGT ||
                   p == arith::CmpFPredicate::UGE || p == arith::CmpFPredicate::OGE);
    bool swapped = isZeroConst(lhs) && rhs == x &&
                   (p == arith::CmpFPredicate::ULT || p == arith::CmpFPredicate::OLT ||
                    p == arith::CmpFPredicate::ULE || p == arith::CmpFPredicate::OLE);
    return direct || swapped;
  };

  auto floatNeg = [&](arith::CmpFPredicate p, Value lhs, Value rhs) {
    bool direct = lhs == x && isZeroConst(rhs) &&
                  (p == arith::CmpFPredicate::ULT || p == arith::CmpFPredicate::OLT ||
                   p == arith::CmpFPredicate::ULE || p == arith::CmpFPredicate::OLE);
    bool swapped = isZeroConst(lhs) && rhs == x &&
                   (p == arith::CmpFPredicate::UGT || p == arith::CmpFPredicate::OGT ||
                    p == arith::CmpFPredicate::UGE || p == arith::CmpFPredicate::OGE);
    return direct || swapped;
  };

  auto intPos = [&](arith::CmpIPredicate p, Value lhs, Value rhs) {
    bool direct = lhs == x && isZeroConst(rhs) &&
                  (p == arith::CmpIPredicate::sgt || p == arith::CmpIPredicate::sge ||
                   p == arith::CmpIPredicate::ugt || p == arith::CmpIPredicate::uge);
    bool swapped = isZeroConst(lhs) && rhs == x &&
                   (p == arith::CmpIPredicate::slt || p == arith::CmpIPredicate::sle ||
                    p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::ule);
    return direct || swapped;
  };

  auto intNeg = [&](arith::CmpIPredicate p, Value lhs, Value rhs) {
    bool direct = lhs == x && isZeroConst(rhs) &&
                  (p == arith::CmpIPredicate::slt || p == arith::CmpIPredicate::sle ||
                   p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::ule);
    bool swapped = isZeroConst(lhs) && rhs == x &&
                   (p == arith::CmpIPredicate::sgt || p == arith::CmpIPredicate::sge ||
                    p == arith::CmpIPredicate::ugt || p == arith::CmpIPredicate::uge);
    return direct || swapped;
  };

  if (sel.getTrueValue() == x && isZeroConst(sel.getFalseValue())) {
    if (cmpf && floatPos(cmpf.getPredicate(), cmpf.getLhs(), cmpf.getRhs()))
      return true;
    if (cmpi && intPos(cmpi.getPredicate(), cmpi.getLhs(), cmpi.getRhs()))
      return true;
  }
  if (isZeroConst(sel.getTrueValue()) && sel.getFalseValue() == x) {
    if (cmpf && floatNeg(cmpf.getPredicate(), cmpf.getLhs(), cmpf.getRhs()))
      return true;
    if (cmpi && intNeg(cmpi.getPredicate(), cmpi.getLhs(), cmpi.getRhs()))
      return true;
  }

  return false;
}

static bool matchTanh(Value yielded, Value x) {
  if (auto t = yielded.getDefiningOp<math::TanhOp>())
    return t.getOperand() == x;
  return false;
}

static bool matchSigmoid(Value yielded, Value x) {
  auto div = yielded.getDefiningOp<arith::DivFOp>();
  if (!div || !isOneConst(div.getLhs()))
    return false;

  auto add = div.getRhs().getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;

  auto tryExpNeg = [&](Value v) {
    if (auto e = v.getDefiningOp<math::ExpOp>())
      return isNegOf(e.getOperand(), x);
    return false;
  };

  if (isOneConst(add.getLhs()) && tryExpNeg(add.getRhs()))
    return true;
  if (isOneConst(add.getRhs()) && tryExpNeg(add.getLhs()))
    return true;

  return false;
}

enum class BinaryElementwiseKind { Add, Sub };

static std::optional<BinaryElementwiseKind> matchBinaryAddSub(Value yielded,
                                                              Value a,
                                                              Value b) {
  if (auto addf = yielded.getDefiningOp<arith::AddFOp>()) {
    if ((addf.getLhs() == a && addf.getRhs() == b) ||
        (addf.getLhs() == b && addf.getRhs() == a))
      return BinaryElementwiseKind::Add;
  }
  if (auto addi = yielded.getDefiningOp<arith::AddIOp>()) {
    if ((addi.getLhs() == a && addi.getRhs() == b) ||
        (addi.getLhs() == b && addi.getRhs() == a))
      return BinaryElementwiseKind::Add;
  }
  if (auto subf = yielded.getDefiningOp<arith::SubFOp>()) {
    if (subf.getLhs() == a && subf.getRhs() == b)
      return BinaryElementwiseKind::Sub;
  }
  if (auto subi = yielded.getDefiningOp<arith::SubIOp>()) {
    if (subi.getLhs() == a && subi.getRhs() == b)
      return BinaryElementwiseKind::Sub;
  }
  return std::nullopt;
}

static std::optional<std::pair<BinaryElementwiseKind, arith::ConstantOp>>
matchUnaryAddSubWithScalar(Value yielded, Value operand) {
  auto check = [&](Value maybeConst)
      -> std::optional<arith::ConstantOp> {
    if (auto cst = maybeConst.getDefiningOp<arith::ConstantOp>())
      return cst;
    return std::nullopt;
  };

  if (auto addf = yielded.getDefiningOp<arith::AddFOp>()) {
    if (addf.getLhs() == operand)
      if (auto c = check(addf.getRhs()))
        return {{BinaryElementwiseKind::Add, *c}};
    if (addf.getRhs() == operand)
      if (auto c = check(addf.getLhs()))
        return {{BinaryElementwiseKind::Add, *c}};
  }
  if (auto addi = yielded.getDefiningOp<arith::AddIOp>()) {
    if (addi.getLhs() == operand)
      if (auto c = check(addi.getRhs()))
        return {{BinaryElementwiseKind::Add, *c}};
    if (addi.getRhs() == operand)
      if (auto c = check(addi.getLhs()))
        return {{BinaryElementwiseKind::Add, *c}};
  }
  if (auto subf = yielded.getDefiningOp<arith::SubFOp>()) {
    if (subf.getLhs() == operand)
      if (auto c = check(subf.getRhs()))
        return {{BinaryElementwiseKind::Sub, *c}};
  }
  if (auto subi = yielded.getDefiningOp<arith::SubIOp>()) {
    if (subi.getLhs() == operand)
      if (auto c = check(subi.getRhs()))
        return {{BinaryElementwiseKind::Sub, *c}};
  }
  return std::nullopt;
}

static LogicalResult rewriteActivationGeneric(linalg::GenericOp op,
                                              IRRewriter &rewriter) {
  if (!linalg::isElementwise(op) || op.getNumDpsInputs() != 1 ||
      op.getNumDpsInits() != 1)
    return failure();
  Block &body = op.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  Value xScalar = body.getArgument(0);
  Value yScalar = yield.getOperand(0);

  std::optional<cinm::ActivationKind> kind;
  if (matchRelu(yScalar, xScalar))
    kind = cinm::ActivationKind::RELU;
  else if (matchSigmoid(yScalar, xScalar))
    kind = cinm::ActivationKind::SIGMOID;
  else if (matchTanh(yScalar, xScalar))
    kind = cinm::ActivationKind::TANH;
  else
    return failure();

  auto kindAttr = cinm::ActivationKindAttr::get(rewriter.getContext(), *kind);
  Value inputVal = op.getDpsInputs()[0];
  Value outputVal = op.getDpsInits()[0];

  rewriter.setInsertionPoint(op);
  if (!op->getResults().empty()) {
    if (!isTensor(inputVal) || !isTensor(op->getResult(0)))
      return failure();
    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    auto act = rewriter.create<cinm::ActivateOp>(op.getLoc(), resultTy, kindAttr,
                                                 inputVal);
    rewriter.replaceOp(op, act.getResult());
    return success();
  }

  if (!isMemRef(inputVal) || !isMemRef(outputVal))
    return failure();
  rewriter.replaceOpWithNewOp<cinm::ActivateMemRefOp>(op, kindAttr, inputVal,
                                                      outputVal);
  return success();
}

static LogicalResult rewriteActivationMap(linalg::MapOp op, IRRewriter &rewriter) {
  if (op.getInputs().size() != 1)
    return failure();
  Block &body = op.getMapper().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  Value xScalar = body.getArgument(0);
  Value yScalar = yield.getOperand(0);

  std::optional<cinm::ActivationKind> kind;
  if (matchRelu(yScalar, xScalar))
    kind = cinm::ActivationKind::RELU;
  else if (matchSigmoid(yScalar, xScalar))
    kind = cinm::ActivationKind::SIGMOID;
  else if (matchTanh(yScalar, xScalar))
    kind = cinm::ActivationKind::TANH;
  else
    return failure();

  auto kindAttr = cinm::ActivationKindAttr::get(rewriter.getContext(), *kind);
  Value inputVal = op.getInputs()[0];
  Value initVal = op.getInit();

  rewriter.setInsertionPoint(op);
  if (op->getResults().empty()) {
    if (!isMemRef(inputVal) || !isMemRef(initVal))
      return failure();
    rewriter.replaceOpWithNewOp<cinm::ActivateMemRefOp>(op, kindAttr, inputVal,
                                                        initVal);
    return success();
  }

  if (!isTensor(inputVal) || !isTensor(op->getResult(0)))
    return failure();
  auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
  auto act = rewriter.create<cinm::ActivateOp>(op.getLoc(), resultTy, kindAttr,
                                               inputVal);
  rewriter.replaceOp(op, act.getResult());
  return success();
}

static LogicalResult rewriteAddSubGeneric(linalg::GenericOp op,
                                          IRRewriter &rewriter) {
  if (!linalg::isElementwise(op))
    return failure();
  Block &body = op.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  Value yielded = yield.getOperand(0);

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  if (op.getNumDpsInputs() == 2) {
    Value lhs = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];
    Value lhsScalar = body.getArgument(0);
    Value rhsScalar = body.getArgument(1);

    auto kind = matchBinaryAddSub(yielded, lhsScalar, rhsScalar);
    if (!kind)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy || lhs.getType() != resultTy || rhs.getType() != resultTy)
      return failure();

    switch (*kind) {
    case BinaryElementwiseKind::Add: {
      auto add = rewriter.create<cinm::AddOp>(loc, resultTy, lhs, rhs);
      rewriter.replaceOp(op, add.getResult());
      return success();
    }
    case BinaryElementwiseKind::Sub: {
      auto sub = rewriter.create<cinm::SubOp>(loc, resultTy, lhs, rhs);
      rewriter.replaceOp(op, sub.getResult());
      return success();
    }
    }
    llvm_unreachable("unhandled binary kind");
  }

  if (op.getNumDpsInputs() == 1) {
    Value tensor = op.getDpsInputs()[0];
    Value scalarArg = body.getArgument(0);
    auto match = matchUnaryAddSubWithScalar(yielded, scalarArg);
    if (!match)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy || tensor.getType() != resultTy)
      return failure();

    auto cloned = rewriter.create<arith::ConstantOp>(loc, match->second.getValue());
    switch (match->first) {
    case BinaryElementwiseKind::Add: {
      auto adds = rewriter.create<cinm::AddsOp>(loc, resultTy, tensor,
                                                cloned.getResult());
      rewriter.replaceOp(op, adds.getResult());
      return success();
    }
    case BinaryElementwiseKind::Sub: {
      auto subs = rewriter.create<cinm::SubsOp>(loc, resultTy, tensor,
                                                cloned.getResult());
      rewriter.replaceOp(op, subs.getResult());
      return success();
    }
    }
    llvm_unreachable("unhandled unary kind");
  }

  return failure();
}

static LogicalResult rewriteAddSubMap(linalg::MapOp op, IRRewriter &rewriter) {
  if (op->getResults().empty())
    return failure();
  Block &body = op.getMapper().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  Value yielded = yield.getOperand(0);

  if (op.getInputs().empty())
    return failure();

  auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultTy)
    return failure();

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  if (op.getInputs().size() == 2) {
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value lhsScalar = body.getArgument(0);
    Value rhsScalar = body.getArgument(1);

    auto kind = matchBinaryAddSub(yielded, lhsScalar, rhsScalar);
    if (!kind)
      return failure();

    if (lhs.getType() != resultTy || rhs.getType() != resultTy)
      return failure();

    switch (*kind) {
    case BinaryElementwiseKind::Add: {
      auto add = rewriter.create<cinm::AddOp>(loc, resultTy, lhs, rhs);
      rewriter.replaceOp(op, add.getResult());
      return success();
    }
    case BinaryElementwiseKind::Sub: {
      auto sub = rewriter.create<cinm::SubOp>(loc, resultTy, lhs, rhs);
      rewriter.replaceOp(op, sub.getResult());
      return success();
    }
    }
    llvm_unreachable("unhandled map binary kind");
  }

  if (op.getInputs().size() == 1) {
    Value tensor = op.getInputs()[0];
    Value scalarArg = body.getArgument(0);
    auto match = matchUnaryAddSubWithScalar(yielded, scalarArg);
    if (!match)
      return failure();

    if (tensor.getType() != resultTy)
      return failure();

    auto cloned = rewriter.create<arith::ConstantOp>(loc, match->second.getValue());
    switch (match->first) {
    case BinaryElementwiseKind::Add: {
      auto adds =
          rewriter.create<cinm::AddsOp>(loc, resultTy, tensor, cloned.getResult());
      rewriter.replaceOp(op, adds.getResult());
      return success();
    }
    case BinaryElementwiseKind::Sub: {
      auto subs =
          rewriter.create<cinm::SubsOp>(loc, resultTy, tensor, cloned.getResult());
      rewriter.replaceOp(op, subs.getResult());
      return success();
    }
    }
    llvm_unreachable("unhandled map unary kind");
  }

  return failure();
}

static bool isContiguous(const ReassociationIndices &indices) {
  if (indices.empty())
    return true;
  for (size_t i = 1; i < indices.size(); ++i)
    if (indices[i] != indices[i - 1] + 1)
      return false;
  return true;
}

static RankedTensorType computeCollapsedType(RankedTensorType type,
                                             ArrayRef<ReassociationIndices> groups) {
  SmallVector<int64_t> newShape;
  newShape.reserve(groups.size());
  for (const auto &group : groups) {
    if (group.empty())
      continue;
    bool dynamic = false;
    int64_t product = 1;
    for (int64_t dim : group) {
      int64_t size = type.getDimSize(dim);
      if (ShapedType::isDynamic(size)) {
        dynamic = true;
        break;
      }
      product *= size;
    }
    newShape.push_back(dynamic ? ShapedType::kDynamic : product);
  }
  if (newShape.empty())
    newShape.push_back(1);
  return RankedTensorType::get(newShape, type.getElementType());
}

static Value collapseTensor(PatternRewriter &rewriter, Location loc, Value value,
                            ArrayRef<ReassociationIndices> reassoc) {
  if (reassoc.empty())
    return value;
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return value;
  auto collapsedTy = computeCollapsedType(tensorTy, reassoc);
  return rewriter.create<tensor::CollapseShapeOp>(loc, collapsedTy, value, reassoc);
}

static Value expandTensor(PatternRewriter &rewriter, Location loc, Value value,
                          RankedTensorType targetType,
                          ArrayRef<ReassociationIndices> reassoc) {
  if (reassoc.empty())
    return value;
  return rewriter.create<tensor::ExpandShapeOp>(loc, targetType, value, reassoc);
}

struct MatmulToCinm : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = op.getInputs();
    ValueRange outputs = op.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected arity");

    Value A = inputs[0];
    Value B = inputs[1];
    Value C = outputs[0];

    if (!op->getResults().empty()) {
      if (!isTensor(A) || !isTensor(B) || !isTensor(op->getResult(0)))
        return rewriter.notifyMatchFailure(op, "requires tensor operands/results");
      auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
      auto gemm = rewriter.create<cinm::GemmOp>(op.getLoc(), resultTy, ValueRange{A, B});
      rewriter.replaceOp(op, gemm.getResult());
      return success();
    }

    if (!isMemRef(A) || !isMemRef(B) || !isMemRef(C))
      return rewriter.notifyMatchFailure(op, "requires memref operands/outs");
    rewriter.replaceOpWithNewOp<cinm::GemmMemRefOp>(op, A, B, C);
    return success();
  }
};

struct MatvecToCinm : public OpConversionPattern<linalg::MatvecOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatvecOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = op.getInputs();
    ValueRange outputs = op.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected arity");

    Value A = inputs[0];
    Value x = inputs[1];
    Value y = outputs[0];

    if (!op->getResults().empty()) {
      if (!isTensor(A) || !isTensor(x) || !isTensor(op->getResult(0)))
        return rewriter.notifyMatchFailure(op, "requires tensor operands/results");
      auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
      auto gemv = rewriter.create<cinm::GemvOp>(op.getLoc(), resultTy, A, x);
      rewriter.replaceOp(op, gemv.getResult());
      return success();
    }

    if (!isMemRef(A) || !isMemRef(x) || !isMemRef(y))
      return rewriter.notifyMatchFailure(op, "requires memref operands/outs");
    rewriter.replaceOpWithNewOp<cinm::GemvMemRefOp>(op, A, x, y);
    return success();
  }
};

struct GenericActivationToCinm : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(op))
      return rewriter.notifyMatchFailure(op, "not elementwise");
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(op, "not 1->1 elementwise");

    Block &body = op.getRegion().front();
    auto *term = body.getTerminator();
    auto yld = dyn_cast<linalg::YieldOp>(term);
    if (!yld || yld.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected region body");

    Value xScalar = body.getArgument(0);
    Value yScalar = yld.getOperand(0);

    std::optional<cinm::ActivationKind> kind;
    if (matchRelu(yScalar, xScalar))
      kind = cinm::ActivationKind::RELU;
    else if (matchSigmoid(yScalar, xScalar))
      kind = cinm::ActivationKind::SIGMOID;
    else if (matchTanh(yScalar, xScalar))
      kind = cinm::ActivationKind::TANH;

    if (!kind)
      return rewriter.notifyMatchFailure(op, "not a recognized activation");

    auto kindAttr = cinm::ActivationKindAttr::get(rewriter.getContext(), *kind);
    Value inputVal = op.getDpsInputs()[0];
    Value outputVal = op.getDpsInits()[0];

    if (!op->getResults().empty()) {
      if (!isTensor(inputVal) || !isTensor(op->getResult(0)))
        return rewriter.notifyMatchFailure(op, "requires tensor operands/results");
      auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
      auto act = rewriter.create<cinm::ActivateOp>(op.getLoc(), resultTy, kindAttr,
                                                   inputVal);
      rewriter.replaceOp(op, act.getResult());
      return success();
    }

    if (!isMemRef(inputVal) || !isMemRef(outputVal))
      return rewriter.notifyMatchFailure(op, "requires memref input/out");
    rewriter.replaceOpWithNewOp<cinm::ActivateMemRefOp>(op, kindAttr, inputVal,
                                                        outputVal);
    return success();
  }
};

struct MapActivationToCinm : public OpConversionPattern<linalg::MapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MapOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected single input for activation");

    Block &body = op.getMapper().front();
    auto *terminator = body.getTerminator();
    auto yield = dyn_cast<linalg::YieldOp>(terminator);
    if (!yield || yield.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected mapper body");

    Value xScalar = body.getArgument(0);
    Value yScalar = yield.getOperand(0);

    std::optional<cinm::ActivationKind> kind;
    if (matchRelu(yScalar, xScalar))
      kind = cinm::ActivationKind::RELU;
    else if (matchSigmoid(yScalar, xScalar))
      kind = cinm::ActivationKind::SIGMOID;
    else if (matchTanh(yScalar, xScalar))
      kind = cinm::ActivationKind::TANH;

    if (!kind)
      return rewriter.notifyMatchFailure(op, "not a recognized activation");

    auto kindAttr = cinm::ActivationKindAttr::get(rewriter.getContext(), *kind);
    Value inputVal = op.getInputs()[0];
    Value initVal = op.getInit();

    if (op->getResults().empty()) {
      if (!isMemRef(inputVal) || !isMemRef(initVal))
        return rewriter.notifyMatchFailure(op, "requires memref input/out");
      rewriter.replaceOpWithNewOp<cinm::ActivateMemRefOp>(op, kindAttr, inputVal,
                                                          initVal);
      return success();
    }

    if (!isTensor(inputVal) || !isTensor(op->getResult(0)))
      return rewriter.notifyMatchFailure(op, "requires tensor operands/results");

    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    auto act = rewriter.create<cinm::ActivateOp>(op.getLoc(), resultTy, kindAttr,
                                                 inputVal);
    rewriter.replaceOp(op, act.getResult());
    return success();
  }
};

struct MapAddSubToCinm : public OpConversionPattern<linalg::MapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MapOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getMapper().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected mapper body");

    Value yielded = yield.getOperand(0);

    if (op->getResults().empty())
      return rewriter.notifyMatchFailure(op, "memref path not supported yet");

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");

    Location loc = op.getLoc();

    if (op.getInputs().size() == 2) {
      Value lhs = op.getInputs()[0];
      Value rhs = op.getInputs()[1];
      Value lhsScalar = body.getArgument(0);
      Value rhsScalar = body.getArgument(1);

      auto kind = matchBinaryAddSub(yielded, lhsScalar, rhsScalar);
      if (!kind)
        return rewriter.notifyMatchFailure(op, "not add/sub pattern");

      switch (*kind) {
      case BinaryElementwiseKind::Add: {
        auto add = rewriter.create<cinm::AddOp>(loc, resultTy, lhs, rhs);
        rewriter.replaceOp(op, add.getResult());
        return success();
      }
      case BinaryElementwiseKind::Sub: {
        auto sub = rewriter.create<cinm::SubOp>(loc, resultTy, lhs, rhs);
        rewriter.replaceOp(op, sub.getResult());
        return success();
      }
      }
      llvm_unreachable("unhandled kind");
    }

    if (op.getInputs().size() == 1) {
      Value tensor = op.getInputs()[0];
      Value scalarArg = body.getArgument(0);
      auto match = matchUnaryAddSubWithScalar(yielded, scalarArg);
      if (!match)
        return rewriter.notifyMatchFailure(op, "not unary add/sub");

      BinaryElementwiseKind kind = match->first;
      auto cstOp = match->second;
      auto cloned = rewriter.create<arith::ConstantOp>(loc, cstOp.getValue());

      switch (kind) {
      case BinaryElementwiseKind::Add: {
        auto adds =
            rewriter.create<cinm::AddsOp>(loc, resultTy, tensor, cloned.getResult());
        rewriter.replaceOp(op, adds.getResult());
        return success();
      }
      case BinaryElementwiseKind::Sub: {
        auto subs =
            rewriter.create<cinm::SubsOp>(loc, resultTy, tensor, cloned.getResult());
        rewriter.replaceOp(op, subs.getResult());
        return success();
      }
      }
      llvm_unreachable("unhandled unary kind");
    }

    return rewriter.notifyMatchFailure(op, "unsupported number of inputs");
  }
};

struct BatchMatvecToCinm : public OpConversionPattern<linalg::BatchMatvecOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::BatchMatvecOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getResults().empty())
      return rewriter.notifyMatchFailure(op, "memref path not supported yet");

    auto lhsType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());

    if (!lhsType || !rhsType || !outType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensor operands/results");
    if (lhsType.getRank() != 3 || rhsType.getRank() != 2 || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "unexpected tensor ranks");

    auto dimEqual = [](int64_t a, int64_t b) {
      return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
    };

    if (!dimEqual(lhsType.getDimSize(0), rhsType.getDimSize(0)) ||
        !dimEqual(lhsType.getDimSize(0), outType.getDimSize(0)) ||
        !dimEqual(lhsType.getDimSize(2), rhsType.getDimSize(1)) ||
        !dimEqual(lhsType.getDimSize(1), outType.getDimSize(1)) ||
        lhsType.getElementType() != rhsType.getElementType() ||
        lhsType.getElementType() != outType.getElementType())
      return rewriter.notifyMatchFailure(op, "incompatible shapes");

    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    rewriter.replaceOpWithNewOp<cinm::BatchGemvOp>(op, resultTy, op.getInputs()[0],
                                                   op.getInputs()[1]);
    return success();
  }
};

struct GenericAddSubToCinm : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(op))
      return rewriter.notifyMatchFailure(op, "not elementwise");

    Block &body = op.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected region");
    Value yielded = yield.getOperand(0);

    if (op->getResults().empty())
      return rewriter.notifyMatchFailure(op, "memref path not supported yet");

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");

    Location loc = op.getLoc();

    if (op.getNumDpsInputs() == 2) {
      Value lhs = op.getDpsInputs()[0];
      Value rhs = op.getDpsInputs()[1];
      Value lhsScalar = body.getArgument(0);
      Value rhsScalar = body.getArgument(1);

      auto kind = matchBinaryAddSub(yielded, lhsScalar, rhsScalar);
      if (!kind)
        return rewriter.notifyMatchFailure(op, "not add/sub pattern");

      switch (*kind) {
      case BinaryElementwiseKind::Add: {
        auto add = rewriter.create<cinm::AddOp>(loc, resultTy, lhs, rhs);
        rewriter.replaceOp(op, add.getResult());
        return success();
      }
      case BinaryElementwiseKind::Sub: {
        auto sub = rewriter.create<cinm::SubOp>(loc, resultTy, lhs, rhs);
        rewriter.replaceOp(op, sub.getResult());
        return success();
      }
      }
      llvm_unreachable("unhandled kind");
    }

    if (op.getNumDpsInputs() == 1) {
      Value tensor = op.getDpsInputs()[0];
      Value scalarArg = body.getArgument(0);
      auto match = matchUnaryAddSubWithScalar(yielded, scalarArg);
      if (!match)
        return rewriter.notifyMatchFailure(op, "not unary add/sub");

      BinaryElementwiseKind kind = match->first;
      auto cstOp = match->second;
      auto cloned = rewriter.create<arith::ConstantOp>(loc, cstOp.getValue());

      switch (kind) {
      case BinaryElementwiseKind::Add: {
        auto adds = rewriter.create<cinm::AddsOp>(loc, resultTy, tensor,
                                                  cloned.getResult());
        rewriter.replaceOp(op, adds.getResult());
        return success();
      }
      case BinaryElementwiseKind::Sub: {
        auto subs = rewriter.create<cinm::SubsOp>(loc, resultTy, tensor,
                                                  cloned.getResult());
        rewriter.replaceOp(op, subs.getResult());
        return success();
      }
      }
      llvm_unreachable("unhandled unary kind");
    }

    return rewriter.notifyMatchFailure(op, "unhandled generic pattern");
  }
};

struct GenericContractionToCinm : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(op, "not a 2-input contraction");

    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    FailureOr<linalg::ContractionDimensions> dimsOr = linalg::inferContractionDims(linalgOp);
    if (failed(dimsOr))
      return rewriter.notifyMatchFailure(op, "not a contraction");
    linalg::ContractionDimensions dims = *dimsOr;
    if (dims.m.size() != 1 || dims.k.size() != 1)
      return rewriter.notifyMatchFailure(op, "unsupported contraction shape");

    Value lhs = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];

    if (op->getResults().empty())
      return rewriter.notifyMatchFailure(op, "memref contractions not supported yet");

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensor operands/results");

    Location loc = op.getLoc();

    auto indexingMaps = op.getIndexingMapsArray();
    AffineMap lhsMap = indexingMaps[0];
    AffineMap rhsMap = indexingMaps[1];

    auto contains = [](ArrayRef<unsigned> vec, unsigned value) {
      return llvm::is_contained(vec, value);
    };

    ReassociationIndices lhsBatchIdx, lhsMIdx, lhsKIdx;
    for (auto [dimIdx, expr] : llvm::enumerate(lhsMap.getResults())) {
      auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        return rewriter.notifyMatchFailure(op, "non-affine lhs map");
      unsigned pos = dimExpr.getPosition();
      if (contains(dims.batch, pos)) {
        lhsBatchIdx.push_back(static_cast<int64_t>(dimIdx));
        continue;
      }
      if (contains(dims.m, pos)) {
        lhsMIdx.push_back(static_cast<int64_t>(dimIdx));
        continue;
      }
      if (contains(dims.k, pos)) {
        lhsKIdx.push_back(static_cast<int64_t>(dimIdx));
        continue;
      }
      return rewriter.notifyMatchFailure(op, "unexpected lhs dimension");
    }

    ReassociationIndices rhsBatchIdx, rhsKIdx, rhsNIdx;
    for (auto [dimIdx, expr] : llvm::enumerate(rhsMap.getResults())) {
      auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        return rewriter.notifyMatchFailure(op, "non-affine rhs map");
      unsigned pos = dimExpr.getPosition();
      if (contains(dims.batch, pos)) {
        rhsBatchIdx.push_back(static_cast<int64_t>(dimIdx));
        continue;
      }
      if (contains(dims.k, pos)) {
        rhsKIdx.push_back(static_cast<int64_t>(dimIdx));
        continue;
      }
      if (!dims.n.empty() && contains(dims.n, pos)) {
        rhsNIdx.push_back(static_cast<int64_t>(dimIdx));
        continue;
      }
      return rewriter.notifyMatchFailure(op, "unexpected rhs dimension");
    }

    ReassociationIndices outBatchIdx, outMIdx, outNIdx;
    int64_t rank = resultType.getRank();
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (contains(dims.batch, static_cast<unsigned>(dim))) {
        outBatchIdx.push_back(dim);
        continue;
      }
      if (contains(dims.m, static_cast<unsigned>(dim))) {
        outMIdx.push_back(dim);
        continue;
      }
      if (!dims.n.empty() && contains(dims.n, static_cast<unsigned>(dim))) {
        outNIdx.push_back(dim);
        continue;
      }
      return rewriter.notifyMatchFailure(op, "unexpected result dimension");
    }

    auto requireNonEmpty = [&](const ReassociationIndices &indices,
                               StringRef message) -> LogicalResult {
      if (indices.empty())
        return rewriter.notifyMatchFailure(op, message);
      return success();
    };

    if (failed(requireNonEmpty(lhsMIdx, "missing lhs m dimensions")) ||
        failed(requireNonEmpty(lhsKIdx, "missing lhs k dimensions")) ||
        failed(requireNonEmpty(rhsKIdx, "missing rhs k dimensions")) ||
        failed(requireNonEmpty(outMIdx, "missing output m dimensions")))
      return failure();

    if (!dims.n.empty() && rhsNIdx.empty())
      return rewriter.notifyMatchFailure(op, "missing rhs n dimensions");
    if (!dims.n.empty() && outNIdx.empty())
      return rewriter.notifyMatchFailure(op, "missing output n dimensions");
    if (!dims.batch.empty() && lhsBatchIdx.empty())
      return rewriter.notifyMatchFailure(op, "missing lhs batch dimensions");
    if (!dims.batch.empty() && rhsBatchIdx.empty())
      return rewriter.notifyMatchFailure(op, "missing rhs batch dimensions");
    if (!dims.batch.empty() && outBatchIdx.empty())
      return rewriter.notifyMatchFailure(op, "missing output batch dimensions");

    if ((!lhsBatchIdx.empty() && !isContiguous(lhsBatchIdx)) ||
        !isContiguous(lhsMIdx) || !isContiguous(lhsKIdx) ||
        (!rhsBatchIdx.empty() && !isContiguous(rhsBatchIdx)) ||
        !isContiguous(rhsKIdx) || (!rhsNIdx.empty() && !isContiguous(rhsNIdx)) ||
        (!outBatchIdx.empty() && !isContiguous(outBatchIdx)) ||
        !isContiguous(outMIdx) || (!outNIdx.empty() && !isContiguous(outNIdx)))
      return rewriter.notifyMatchFailure(op, "non-contiguous dimensions");

    SmallVector<ReassociationIndices, 3> lhsReassoc;
    if (!lhsBatchIdx.empty())
      lhsReassoc.push_back(lhsBatchIdx);
    lhsReassoc.push_back(lhsMIdx);
    lhsReassoc.push_back(lhsKIdx);

    SmallVector<ReassociationIndices, 3> rhsReassoc;
    if (!rhsBatchIdx.empty())
      rhsReassoc.push_back(rhsBatchIdx);
    rhsReassoc.push_back(rhsKIdx);
    if (!rhsNIdx.empty())
      rhsReassoc.push_back(rhsNIdx);

    SmallVector<ReassociationIndices, 3> outReassoc;
    if (!outBatchIdx.empty())
      outReassoc.push_back(outBatchIdx);
    outReassoc.push_back(outMIdx);
    if (!outNIdx.empty())
      outReassoc.push_back(outNIdx);
    Value collapsedLhs = collapseTensor(rewriter, loc, lhs, lhsReassoc);
    Value collapsedRhs = collapseTensor(rewriter, loc, rhs, rhsReassoc);

    RankedTensorType collapsedResultType =
        computeCollapsedType(resultType, outReassoc);

    bool hasBatch = !dims.batch.empty();
    Value newResult;
    if (hasBatch) {
      if (!outNIdx.empty()) {
        auto batchGemm = rewriter.create<cinm::BatchGemmOp>(loc, collapsedLhs,
                                                            collapsedRhs);
        newResult = batchGemm.getResult();
      } else {
        auto batchGemv = rewriter.create<cinm::BatchGemvOp>(
            loc, collapsedResultType, collapsedLhs, collapsedRhs);
        newResult = batchGemv.getResult();
      }
    } else {
      if (!outNIdx.empty()) {
        auto gemm = rewriter.create<cinm::GemmOp>(loc, collapsedResultType,
                                                  ValueRange{collapsedLhs, collapsedRhs});
        newResult = gemm.getResult();
      } else {
        auto gemv = rewriter.create<cinm::GemvOp>(loc, collapsedResultType,
                                                  collapsedLhs, collapsedRhs);
        newResult = gemv.getResult();
      }
    }

    Value finalResult = newResult;
    if (collapsedResultType != resultType)
      finalResult = expandTensor(rewriter, loc, newResult, resultType, outReassoc);

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

struct ConvertLinalgToCinmPass
    : public ConvertLinalgToCinmBase<ConvertLinalgToCinmPass> {
  using Base = ConvertLinalgToCinmBase<ConvertLinalgToCinmPass>;
  using Base::Base;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    func::FuncOp func = getOperation();

    IRRewriter rewriter(&ctx);
    llvm::errs() << "ConvertLinalgToCinm on function " << func.getName() << "\n";

    SmallVector<linalg::GenericOp> genericOps;
    func.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });
    for (auto op : llvm::reverse(genericOps)) {
      if (succeeded(rewriteActivationGeneric(op, rewriter)))
        continue;
      if (succeeded(rewriteAddSubGeneric(op, rewriter)))
        continue;
    }

    SmallVector<linalg::MapOp> mapOps;
    func.walk([&](linalg::MapOp op) { mapOps.push_back(op); });
    for (auto op : llvm::reverse(mapOps)) {
      if (succeeded(rewriteActivationMap(op, rewriter)))
        continue;
      if (succeeded(rewriteAddSubMap(op, rewriter)))
        continue;
    }

    ConversionTarget target(ctx);
    target.addLegalDialect<cinm::CinmDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    target.addIllegalOp<linalg::MatmulOp>();
    target.addIllegalOp<linalg::MatvecOp>();
    target.addIllegalOp<linalg::BatchMatvecOp>();

    RewritePatternSet patterns(&ctx);
    patterns.insert<MatmulToCinm, MatvecToCinm, BatchMatvecToCinm,
                    GenericContractionToCinm>(&ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createConvertLinalgToCinmPass() {
  return std::make_unique<ConvertLinalgToCinmPass>();
}

void mlir::cinm::registerLinalgToCinmPipeline() {}
