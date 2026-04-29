#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Value.h>

using namespace mlir;

namespace {

static Value getReturnBuffer(OpBuilder &rewriter, Location loc,
                             RankedTensorType bufTy) {

  mlir::MemRefType dstMR = cast<mlir::MemRefType>(
      bufferization::getMemRefTypeWithStaticIdentityLayout(bufTy));
  // SmallVector<Value> dynDims;
  // for (int64_t d = 0; d < bufTy.getRank(); ++d) {
  //   if (bufTy.isDynamicDim(d)) {
  //     Value cd = arith::ConstantIndexOp::create(rewriter, loc, d);
  //     dynDims.push_back(tensor::DimOp::create(rewriter, loc, lhsT, cd));
  //   }
  // }
  return memref::AllocOp::create(rewriter, loc, dstMR, ValueRange{});
}

static Value materializeZeroLikeTensor(RewriterBase &rewriter, Location loc,
                                       Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getFloatAttr(ft, 0.0));
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getIntegerAttr(it, 0));
  return {};
}

static FailureOr<Value>
materializeAsExactMemref(RewriterBase &rewriter, Location loc, Value v,
                         MemRefType expectedTy,
                         const bufferization::BufferizationOptions &options,
                         const bufferization::BufferizationState &state) {
  Value m = v;
  if (!isa<MemRefType>(m.getType())) {
    FailureOr<Value> buf =
        bufferization::getBuffer(rewriter, m, options, state);
    if (failed(buf))
      return failure();
    m = *buf;
  }
  auto gotTy = cast<MemRefType>(m.getType());
  if (gotTy != expectedTy)
    m = memref::CastOp::create(rewriter, loc, expectedTy, m);
  return m;
}

static LogicalResult
bufferizeComputeResults(Operation *op, RewriterBase &rewriter,
                        const bufferization::BufferizationOptions &options,
                        bufferization::BufferizationState &state) {
  Location loc = op->getLoc();

  Operation *termOp = op->getRegion(0).front().getTerminator();
  auto term = dyn_cast<cinm::YieldOp>(termOp);
  if (!term)
    return op->emitError("expected cinm.yield as terminator"), failure();
  for (auto [i, yieldVal, res] :
       llvm::enumerate(term->getOpOperands(), op->getOpResults())) {
    Value v = yieldVal.get();
    if (llvm::dyn_cast_or_null<TensorType>(v.getType())) {
      FailureOr<Value> buf =
          bufferization::getBuffer(rewriter, v, options, state);
      if (failed(buf))
        return op->emitError("cinm.compute bufferize: result #")
               << i << " failed bufferization";
      yieldVal.set(*buf);

      auto tensorTy = res.getType();
      res.setType(buf->getType());

      rewriter.setInsertionPointAfter(op);
      auto totensor =
          bufferization::ToTensorOp::create(rewriter, loc, tensorTy, res);
      rewriter.replaceAllUsesExcept(res, totensor, totensor);
    }
  }
  return success();
}

static void getComputeYieldAliasingOpOperands(
    Operation *op, Value value, const bufferization::AnalysisState &,
    llvm::SmallVectorImpl<bufferization::AliasingOpOperand> &result) {

  if (auto res = llvm::dyn_cast_or_null<OpResult>(value);
      value.getDefiningOp() == op) {
    OpOperand &yielded =
        cast<cinm::YieldOp>(op->getRegion(0).front().getTerminator())
            ->getOpOperand(res.getResultNumber());
    result.emplace_back(&yielded, bufferization::BufferRelation::Equivalent,
                        true);
  }
}

struct ComputeBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ComputeBufferizableInterface, cinm::ComputeOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opnd,
                              const bufferization::AnalysisState &state) const {
    auto bbarg =
        cast<cinm::ComputeOp>(op).getBodyArguments()[opnd.getOperandNumber()];
    return state.isValueRead(bbarg);
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opnd,
                    const bufferization::AnalysisState &state) const {
    auto computeOp = cast<cinm::ComputeOp>(op);
    auto bbarg = computeOp.getBodyArguments()[opnd.getOperandNumber()];
    auto yield =
        cast<cinm::YieldOp>(computeOp.getBody().front().getTerminator());
    SmallVector<bufferization::AliasingValue> aliasing;
    for (auto [yielded, result] :
         llvm::zip(yield->getOperands(), computeOp->getOpResults())) {
      if (state.areEquivalentBufferizedValues(yielded, bbarg)) {
        aliasing.emplace_back(result, bufferization::BufferRelation::Equivalent,
                              true);
      }
    }
    return aliasing;
  }

  bufferization::AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const bufferization::AnalysisState &state) const {
    llvm::SmallVector<bufferization::AliasingOpOperand> result;
    getComputeYieldAliasingOpOperands(op, value, state, result);
    return std::move(result);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto oldCompute = cast<cinm::ComputeOp>(op);
    Location loc = op->getLoc();

    for (auto [arg, operand] : oldCompute.zipArgsWithOpOperands()) {
      if (llvm::dyn_cast_or_null<TensorType>(operand.get().getType())) {
        FailureOr<Value> buf =
            bufferization::getBuffer(rewriter, operand.get(), options, state);
        if (failed(buf))
          return op->emitError(
              "cinm.compute bufferize: operand failed bufferization");

        auto tensorTy = arg.getType();
        rewriter.setInsertionPointToStart(arg.getOwner());
        auto totensor =
            bufferization::ToTensorOp::create(rewriter, loc, tensorTy, arg);

        arg.setType(buf->getType());
        rewriter.replaceAllUsesExcept(arg, totensor, totensor);
        operand.set(*buf);
      }
    }
    return bufferizeComputeResults(op, rewriter, options, state);
  }
};

struct FlexComputeBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          FlexComputeBufferizableInterface, cinm::FlexComputeOp> {

  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }
  bufferization::AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const bufferization::AnalysisState &state) const {
    llvm::SmallVector<bufferization::AliasingOpOperand> result;
    getComputeYieldAliasingOpOperands(op, value, state, result);
    return std::move(result);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    return bufferizeComputeResults(op, rewriter, options, state);
  }
};

template <class Op>
static bufferization::AliasingValueList aliasBiasWithResult(Operation *op,
                                                            OpOperand &opnd) {

  auto gemmlike = cast<Op>(op);
  if (gemmlike.getBias() && opnd.get() == gemmlike.getBias())
    return {bufferization::AliasingValue(
        gemmlike->getOpResult(0), bufferization::BufferRelation::Equivalent,
        false)};
  return {};
}

struct GemmBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          GemmBufferizableInterface, cinm::GemmOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opnd,
                    const bufferization::AnalysisState &) const {
    return aliasBiasWithResult<cinm::GemmOp>(op, opnd);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto gemm = cast<cinm::GemmOp>(op);
    Location loc = op->getLoc();

    Value aT = gemm.getLhs();
    Value bT = gemm.getRhs();
    auto cRT = cast<RankedTensorType>(gemm.getResult().getType());
    Type elemTy = cRT.getElementType();

    auto aMem = bufferization::getBuffer(rewriter, aT, options, state);
    auto bMem = bufferization::getBuffer(rewriter, bT, options, state);
    if (failed(aMem) || failed(bMem))
      return failure();

    Value dst = getReturnBuffer(rewriter, loc, cRT);

    if (Value biasT = gemm.getBias()) {
      auto biasMem = bufferization::getBuffer(rewriter, biasT, options, state);
      if (failed(biasMem))
        return failure();
      memref::CopyOp::create(rewriter, loc, *biasMem, dst);
    } else {
      Value zero = materializeZeroLikeTensor(rewriter, loc, elemTy);
      if (!zero)
        return op->emitError("cinm.gemm bufferize: unsupported element type"),
               failure();
      (void)linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                   ValueRange{dst});
    }

    cinm::GemmOp::create(rewriter, loc, *aMem, *bMem, Value(), dst);

    Value t =
        bufferization::ToTensorOp::create(rewriter, loc, cRT, dst, true, true);
    rewriter.replaceOp(op, t);
    return success();
  }
};

struct GemvBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          GemvBufferizableInterface, cinm::GemvOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opnd,
                    const bufferization::AnalysisState &) const {
    return aliasBiasWithResult<cinm::GemvOp>(op, opnd);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto gemv = cast<cinm::GemvOp>(op);
    Location loc = gemv.getLoc();

    Value aT = gemv.getLhs();
    Value xT = gemv.getRhs();
    auto yRT = cast<RankedTensorType>(gemv.getResult().getType());

    auto aMem = bufferization::getBuffer(rewriter, aT, options, state);
    auto xMem = bufferization::getBuffer(rewriter, xT, options, state);
    if (failed(aMem) || failed(xMem))
      return failure();

    Value dst = getReturnBuffer(rewriter, loc, yRT);

    if (Value biasT = gemv.getBias()) {
      auto biasMem = bufferization::getBuffer(rewriter, biasT, options, state);
      if (failed(biasMem))
        return failure();
      memref::CopyOp::create(rewriter, loc, *biasMem, dst);
    } else {
      Value zero =
          materializeZeroLikeTensor(rewriter, loc, yRT.getElementType());
      if (!zero)
        return op->emitError("cinm.gemv bufferize: unsupported element type"),
               failure();
      (void)linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                   ValueRange{dst});
    }

    cinm::GemvOp::create(rewriter, loc, *aMem, *xMem, Value(), dst);

    Value yT =
        bufferization::ToTensorOp::create(rewriter, loc, yRT, dst, true, true);
    rewriter.replaceOp(op, yT);
    return success();
  }
};

struct ElementwiseBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ElementwiseBufferizableInterface, cinm::ElementwiseOp> {

  bool bufferizesToElementwiseAccess(Operation *,
                                     const bufferization::AnalysisState &,
                                     ArrayRef<OpOperand *>) const {
    return true;
  }
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opnd,
                              const bufferization::AnalysisState &) const {
    auto eltwise = cast<cinm::ElementwiseOp>(op);
    return opnd.get() != eltwise.getOut();
  }
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opnd,
                               const bufferization::AnalysisState &) const {
    auto eltwise = cast<cinm::ElementwiseOp>(op);
    return opnd.get() == eltwise.getOut();
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &,
                    const bufferization::AnalysisState &) const {
    auto result = op->getResult(0);
    return {bufferization::AliasingValue(
        result, bufferization::BufferRelation::Equivalent, false)};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto add = cast<cinm::ElementwiseOp>(op);
    Location loc = add.getLoc();

    auto lhsMem =
        bufferization::getBuffer(rewriter, add.getLhs(), options, state);
    if (failed(lhsMem))
      return failure();

    Value rhsMem;
    if (add.getRhs()) {
      auto rhsBuf =
          bufferization::getBuffer(rewriter, add.getRhs(), options, state);
      if (failed(rhsBuf))
        return failure();
      rhsMem = *rhsBuf;
    }
    Value dst;
    if (add.getOut()) {
      dst = add.getOut();
    } else {
      dst = getReturnBuffer(rewriter, loc, add.getResult().getType());
    }

    cinm::ElementwiseOp::create(rewriter, loc, add.getKind(), *lhsMem, rhsMem,
                                dst);

    if (!add.getOut()) {
      Value outT = bufferization::ToTensorOp::create(
          rewriter, loc, add.getResult().getType(), dst, true, true);
      rewriter.replaceOp(op, outT);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};
struct ReduceBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ReduceBufferizableInterface, cinm::ReduceOp> {

  bool bufferizesToElementwiseAccess(Operation *,
                                     const bufferization::AnalysisState &,
                                     ArrayRef<OpOperand *>) const {
    return true;
  }
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto dq = cast<cinm::ReduceOp>(op);

    auto &input = dq.getInputMutable();
    auto inputBuf =
        bufferization::getBuffer(rewriter, input.get(), options, state);
    if (llvm::failed(inputBuf))
      return failure();
    dq.getInputMutable().set(*inputBuf);
    return success();
  }
};

struct QuantizeBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          QuantizeBufferizableInterface, cinm::QuantizeOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto q = cast<cinm::QuantizeOp>(op);
    Location loc = q.getLoc();

    auto srcT = q.getSrc();
    auto srcRT = cast<RankedTensorType>(srcT.getType());
    auto dstRT = cast<RankedTensorType>(q.getResult().getType());

    auto srcMR = bufferization::getMemRefTypeWithFullyDynamicLayout(srcRT);
    Value srcMem =
        bufferization::ToBufferOp::create(rewriter, loc, srcMR, srcT, true);

    auto dstMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(dstRT));
    SmallVector<Value> dynDims;
    for (int64_t i = 0, e = dstRT.getRank(); i < e; ++i)
      if (dstRT.isDynamicDim(i)) {
        Value ci = arith::ConstantIndexOp::create(rewriter, loc, i);
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, srcT, ci));
      }
    Value dstMem = memref::AllocOp::create(rewriter, loc, dstMR, dynDims);

    FloatAttr scale = q.getScaleAttr();
    IntegerAttr zp = q.getZeroPointAttr();
    IntegerAttr axis = q.getAxisAttr();
    auto round = q.getRoundingAttr();
    auto narrow = q.getNarrowRangeAttr();

    cinm::QuantizeOp::create(rewriter, loc, Type(), srcMem, dstMem, scale, zp,
                             axis, round, narrow);

    Value dstT = bufferization::ToTensorOp::create(rewriter, loc, dstRT, dstMem,
                                                   true, true);
    rewriter.replaceOp(op, dstT);
    return success();
  }
};

struct DequantizeBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          DequantizeBufferizableInterface, cinm::DequantizeOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto dq = cast<cinm::DequantizeOp>(op);
    Location loc = dq.getLoc();

    auto srcT = dq.getSrc();
    auto srcRT = cast<RankedTensorType>(srcT.getType());
    auto dstRT = cast<RankedTensorType>(dq.getResult().getType());

    auto srcMR = bufferization::getMemRefTypeWithFullyDynamicLayout(srcRT);
    Value srcMem =
        bufferization::ToBufferOp::create(rewriter, loc, srcMR, srcT, true);

    auto dstMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(dstRT));
    SmallVector<Value> dynDims;
    for (int64_t i = 0, e = dstRT.getRank(); i < e; ++i)
      if (dstRT.isDynamicDim(i)) {
        Value ci = arith::ConstantIndexOp::create(rewriter, loc, i);
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, srcT, ci));
      }
    Value dstMem = memref::AllocOp::create(rewriter, loc, dstMR, dynDims);

    FloatAttr scale = dq.getScaleAttr();
    IntegerAttr zp = dq.getZeroPointAttr();
    IntegerAttr axis = dq.getAxisAttr();

    cinm::DequantizeOp::create(rewriter, loc, Type(), srcMem, dstMem, scale, zp,
                               axis);

    Value dstT = bufferization::ToTensorOp::create(rewriter, loc, dstRT, dstMem,
                                                   true, true);
    rewriter.replaceOp(op, dstT);
    return success();
  }
};

struct ActivateBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ActivateBufferizableInterface, cinm::ActivateOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto act = cast<cinm::ActivateOp>(op);
    Location loc = act.getLoc();

    Value inT = act->getOperand(0);
    auto inRT = cast<RankedTensorType>(inT.getType());
    auto outRT = cast<RankedTensorType>(act->getResult(0).getType());

    auto inMR = bufferization::getMemRefTypeWithFullyDynamicLayout(inRT);
    Value inMem =
        bufferization::ToBufferOp::create(rewriter, loc, inMR, inT, true);

    auto outMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(outRT));

    SmallVector<Value> dynDims;
    for (int64_t d = 0, e = outRT.getRank(); d < e; ++d)
      if (outRT.isDynamicDim(d)) {
        Value cd = arith::ConstantIndexOp::create(rewriter, loc, d);
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, inT, cd));
      }
    Value outMem = memref::AllocOp::create(rewriter, loc, outMR, dynDims);

    cinm::ActivationKind kind = act.getKind();

    cinm::ActivateOp::create(rewriter, loc, kind, inMem, outMem);

    Value outT = bufferization::ToTensorOp::create(rewriter, loc, outRT, outMem,
                                                   true, true);
    rewriter.replaceOp(op, outT);
    return success();
  }
};

struct ScfForBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ScfForBufferizableInterface, scf::ForOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return false;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpResult,
                    const bufferization::AnalysisState &) const {
    return {};
  }

  bool isWritable(Operation *, OpOperand &,
                  const bufferization::AnalysisState &) const {
    return false;
  }
  bool isWritable(Operation *, Value,
                  const bufferization::AnalysisState &) const {
    return false;
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *, Value v,
                const bufferization::BufferizationOptions &,
                const bufferization::AnalysisState &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return bufferization::getMemRefTypeWithFullyDynamicLayout(rtt);
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return bufferization::getMemRefTypeWithFullyDynamicLayout(rtt);
    return failure();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto oldFor = cast<scf::ForOp>(op);
    Location loc = oldFor.getLoc();

    ValueRange oldInits = oldFor.getInitArgs();
    if (oldInits.empty())
      return success();

    SmallVector<Value> memInitArgs;
    memInitArgs.reserve(oldInits.size());
    for (Value init : oldInits) {
      if (isa<MemRefType>(init.getType())) {
        memInitArgs.push_back(init);
        continue;
      }
      if (!isa<RankedTensorType>(init.getType()))
        return op->emitError("scf.for: unexpected non-tensor init arg"),
               failure();

      FailureOr<Value> buf =
          bufferization::getBuffer(rewriter, init, options, state);
      if (failed(buf))
        return op->emitError("scf.for: failed to get buffer for init arg"),
               failure();
      memInitArgs.push_back(*buf);
    }

    OpBuilder::InsertionGuard outerGuard(rewriter);
    rewriter.setInsertionPoint(oldFor);

    auto newFor = scf::ForOp::create(rewriter, loc, oldFor.getLowerBound(),
                                     oldFor.getUpperBound(), oldFor.getStep(),
                                     memInitArgs);

    Block *oldBody = oldFor.getBody();
    Block *newBody = newFor.getBody();

    IRMapping mapper;
    mapper.map(oldBody->getArgument(0), newBody->getArgument(0));

    {
      OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(newBody);

      for (unsigned i = 0, e = oldInits.size(); i < e; ++i) {
        BlockArgument newMem = newBody->getArgument(1 + i);
        BlockArgument oldArg = oldBody->getArgument(1 + i);

        if (isa<MemRefType>(oldArg.getType())) {
          mapper.map(oldArg, newMem);
        } else {
          auto tt = cast<RankedTensorType>(oldArg.getType());
          Value tview = bufferization::ToTensorOp::create(rewriter, loc, tt,
                                                          newMem, true, true);
          mapper.map(oldArg, tview);
        }
      }

      for (Operation &nested :
           llvm::make_early_inc_range(oldBody->without_terminator()))
        rewriter.clone(nested, mapper);

      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      SmallVector<Value> newYields;
      newYields.reserve(oldYield.getNumOperands());

      for (unsigned i = 0, e = oldYield.getNumOperands(); i < e; ++i) {
        Value mapped = mapper.lookup(oldYield.getOperand(i));

        auto expectedTy =
            cast<MemRefType>(newBody->getArgument(1 + i).getType());

        FailureOr<Value> exact = materializeAsExactMemref(
            rewriter, loc, mapped, expectedTy, options, state);
        if (failed(exact))
          return op->emitError(
                     "scf.for: failed to materialize exact memref for "
                     "yield #")
                     << i,
                 failure();

        newYields.push_back(*exact);
      }

      Operation *maybeTerm = nullptr;
      if (!newBody->empty()) {
        Operation &last = newBody->back();
        if (last.hasTrait<OpTrait::IsTerminator>())
          maybeTerm = &last;
      }

      if (maybeTerm) {
        rewriter.setInsertionPoint(maybeTerm);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(maybeTerm, newYields);
      } else {
        rewriter.setInsertionPointToEnd(newBody);
        scf::YieldOp::create(rewriter, loc, newYields);
      }
    }

    SmallVector<Value> replacements;
    replacements.reserve(newFor->getNumResults());
    rewriter.setInsertionPointAfter(newFor);
    for (auto it : llvm::enumerate(newFor->getResults())) {
      Type wantedT = oldFor->getResult(it.index()).getType();
      Value t = bufferization::ToTensorOp::create(rewriter, loc, wantedT,
                                                  it.value(), true, true);
      replacements.push_back(t);
    }
    rewriter.replaceOp(oldFor, replacements);
    return success();
  }
};
} // namespace

void mlir::cinm::registerCinmBufferizableOpInterfaces(
    DialectRegistry &registry) {
  registry.addExtension<::mlir::cinm::CinmDialect>(+[](MLIRContext *ctx,
                                                       ::mlir::cinm::CinmDialect
                                                           *) {
    ::mlir::cinm::ComputeOp::attachInterface<ComputeBufferizableInterface>(
        *ctx);
    ::mlir::cinm::FlexComputeOp::attachInterface<
        FlexComputeBufferizableInterface>(*ctx);
    ::mlir::cinm::GemmOp::attachInterface<GemmBufferizableInterface>(*ctx);
    ::mlir::cinm::GemvOp::attachInterface<GemvBufferizableInterface>(*ctx);
    ::mlir::cinm::ReduceOp::attachInterface<ReduceBufferizableInterface>(*ctx);
    ::mlir::cinm::ElementwiseOp::attachInterface<
        ElementwiseBufferizableInterface>(*ctx);
    ::mlir::cinm::QuantizeOp::attachInterface<QuantizeBufferizableInterface>(
        *ctx);
    ::mlir::cinm::DequantizeOp::attachInterface<
        DequantizeBufferizableInterface>(*ctx);
    ::mlir::cinm::ActivateOp::attachInterface<ActivateBufferizableInterface>(
        *ctx);
  });

  // registry.addExtension<::mlir::scf::SCFDialect>(
  //     +[](MLIRContext *ctx, ::mlir::scf::SCFDialect *) {
  //       ::mlir::scf::ForOp::attachInterface<
  //           ::mlir::cinm::ScfForBufferizableInterface>(*ctx);
  //     });
}
