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
#include "mlir/Support/TypeID.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace mlir::cinm {

static Value materializeZeroLikeTensor(RewriterBase &rewriter, Location loc,
                                       Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return arith::ConstantOp::create(rewriter,loc,
                                              rewriter.getFloatAttr(ft, 0.0));
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return arith::ConstantOp::create(rewriter,loc,
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
    m = memref::CastOp::create(rewriter,loc, expectedTy, m);
  return m;
}

struct ComputeBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ComputeBufferizableInterface, cinm::ComputeOp> {

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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto oldCompute = cast<cinm::ComputeOp>(op);
    Location loc = op->getLoc();

    Operation *termOp = oldCompute.getBody().front().getTerminator();
    auto term = dyn_cast<cinm::YieldOp>(termOp);
    if (!term)
      return op->emitError("expected cinm.yield as terminator"), failure();

    SmallVector<Value> newYieldVals;
    SmallVector<Type> newResultTypes;

    rewriter.setInsertionPoint(term);
    for (Value v : term.getOperands()) {
      if (auto mt = dyn_cast<MemRefType>(v.getType())) {
        newYieldVals.push_back(v);
        newResultTypes.push_back(mt);
        continue;
      }
      if (auto tt = dyn_cast<RankedTensorType>(v.getType())) {
        if (auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>()) {
          Value mem = toTensor.getBuffer();
          newYieldVals.push_back(mem);
          newResultTypes.push_back(mem.getType());
          continue;
        }
        BaseMemRefType mr =
            bufferization::getMemRefTypeWithFullyDynamicLayout(tt);
        Value mem =
            bufferization::ToBufferOp::create(rewriter,loc, mr, v, false);
        newYieldVals.push_back(mem);
        newResultTypes.push_back(mem.getType());
        continue;
      }
      return op->emitError("cinm.compute bufferize: result #")
                 << newYieldVals.size() << " is not a tensor or memref",
             failure();
    }
    term->setOperands(newYieldVals);

    OperationState st(loc, cinm::ComputeOp::getOperationName());
    st.addTypes(newResultTypes);
    st.addAttributes(oldCompute->getAttrDictionary().getValue());
    (void)st.addRegion();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(oldCompute);
    Operation *newOpGeneric = Operation::create(st);
    rewriter.insert(newOpGeneric);
    auto newComputeMR = cast<cinm::ComputeOp>(newOpGeneric);

    newComputeMR.getBody().takeBody(oldCompute.getBody());

    SmallVector<Value> replacement;
    replacement.reserve(newComputeMR->getNumResults());
    for (auto [idx, res] : llvm::enumerate(newComputeMR->getResults())) {
      Type wantedTensorTy = oldCompute->getResult(idx).getType();
      Value t = bufferization::ToTensorOp::create(rewriter,loc, wantedTensorTy,
                                                           res, true, true);
      replacement.push_back(t);
    }

    rewriter.replaceOp(oldCompute, replacement);
    return success();
  }
};

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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
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

    auto dstMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(cRT));
    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < cRT.getRank(); ++i)
      if (cRT.isDynamicDim(i)) {
        Value idx = arith::ConstantIndexOp::create(rewriter,loc, i);
        dynDims.push_back(
            tensor::DimOp::create(rewriter,loc, gemm.getResult(), idx));
      }
    Value dst = memref::AllocOp::create(rewriter,loc, dstMR, dynDims);

    if (Value biasT = gemm.getBias()) {
      auto biasMem = bufferization::getBuffer(rewriter, biasT, options, state);
      if (failed(biasMem))
        return failure();
      memref::CopyOp::create(rewriter,loc, *biasMem, dst);
    } else {
      Value zero = materializeZeroLikeTensor(rewriter, loc, elemTy);
      if (!zero)
        return op->emitError("cinm.gemm bufferize: unsupported element type"),
               failure();
      (void)linalg::FillOp::create(rewriter,loc, ValueRange{zero},
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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto gemv = cast<cinm::GemvOp>(op);
    Location loc = gemv.getLoc();

    Value aT = gemv.getLhs();
    Value xT = gemv.getRhs();
    Value bT = gemv.getBias();
    auto aRT = cast<RankedTensorType>(aT.getType());
    auto xRT = cast<RankedTensorType>(xT.getType());
    auto yRT = cast<RankedTensorType>(gemv.getResult().getType());
    Type elt = yRT.getElementType();

    auto aMR = bufferization::getMemRefTypeWithFullyDynamicLayout(aRT);
    auto xMR = bufferization::getMemRefTypeWithFullyDynamicLayout(xRT);
    Value aMem = bufferization::ToBufferOp::create(rewriter,loc, aMR, aT, true);
    Value xMem = bufferization::ToBufferOp::create(rewriter,loc, xMR, xT, true);

    auto yMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(yRT));

    SmallVector<Value> dynDims;
    dynDims.reserve(yRT.getRank());
    if (yRT.isDynamicDim(0)) {
      Value c0 = arith::ConstantIndexOp::create(rewriter,loc, 0);
      dynDims.push_back(tensor::DimOp::create(rewriter,loc, aT, c0));
    }
    for (int64_t d = 1; d < yRT.getRank(); ++d) {
      if (yRT.isDynamicDim(d)) {
        Value cd = arith::ConstantIndexOp::create(rewriter,loc, d);
        dynDims.push_back(
            tensor::DimOp::create(rewriter,loc, gemv.getResult(), cd));
      }
    }

    Value yMem = memref::AllocOp::create(rewriter,loc, yMR, dynDims);

    if (bT) {
      auto bMRdyn = bufferization::getMemRefTypeWithFullyDynamicLayout(yRT);
      Value bMem =
          bufferization::ToBufferOp::create(rewriter,loc, bMRdyn, bT, true);
      memref::CopyOp::create(rewriter,loc, bMem, yMem);
    } else {
      Value zero = materializeZeroLikeTensor(rewriter, loc, elt);
      if (!zero)
        return op->emitError("cinm.gemv bufferize: unsupported element type"),
               failure();
      (void)linalg::FillOp::create(rewriter,loc, ValueRange{zero},
                                            ValueRange{yMem});
    }

    cinm::GemvOp::create(rewriter,loc, Type(), aMem, xMem, Value(), yMem);

    Value yT =
        bufferization::ToTensorOp::create(rewriter,loc, yRT, yMem, true, true);
    rewriter.replaceOp(op, yT);
    return success();
  }
};

struct ElementwiseBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ElementwiseBufferizableInterface, cinm::ElementwiseOp> {
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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto add = cast<cinm::ElementwiseOp>(op);
    Location loc = add.getLoc();

    Value lhsT = add.getLhs();
    Value rhsT = add.getRhs();
    auto lhsRT = dyn_cast<RankedTensorType>(lhsT.getType());
    auto rhsRT = dyn_cast<RankedTensorType>(rhsT.getType());
    auto resRT = dyn_cast<RankedTensorType>(add.getResult().getType());
    if (!lhsRT || !rhsRT || !resRT)
      return op->emitError("cinm.add bufferize: expected ranked tensor types"),
             failure();
    if (lhsRT.getShape() != rhsRT.getShape() ||
        resRT.getShape() != lhsRT.getShape() ||
        lhsRT.getElementType() != rhsRT.getElementType() ||
        resRT.getElementType() != lhsRT.getElementType())
      return op->emitError(
                 "cinm.elementwise bufferize: mismatched shapes/element types"),
             failure();

    auto lhsMR = bufferization::getMemRefTypeWithFullyDynamicLayout(lhsRT);
    auto rhsMR = bufferization::getMemRefTypeWithFullyDynamicLayout(rhsRT);
    Value lhsMem =
        bufferization::ToBufferOp::create(rewriter,loc, lhsMR, lhsT, true);
    Value rhsMem =
        bufferization::ToBufferOp::create(rewriter,loc, rhsMR, rhsT, true);

    auto dstMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(resRT));
    SmallVector<Value> dynDims;
    for (int64_t d = 0; d < resRT.getRank(); ++d) {
      if (resRT.isDynamicDim(d)) {
        Value cd = arith::ConstantIndexOp::create(rewriter,loc, d);
        dynDims.push_back(tensor::DimOp::create(rewriter,loc, lhsT, cd));
      }
    }
    Value dst = memref::AllocOp::create(rewriter,loc, dstMR, dynDims);

    cinm::ElementwiseOp::create(rewriter,loc, Type(), add.getKind(), lhsMem,
                                         rhsMem, dst);

    Value outT =
        bufferization::ToTensorOp::create(rewriter,loc, resRT, dst, true, true);
    rewriter.replaceOp(op, outT);
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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
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
        bufferization::ToBufferOp::create(rewriter,loc, srcMR, srcT, true);

    auto dstMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(dstRT));
    SmallVector<Value> dynDims;
    for (int64_t i = 0, e = dstRT.getRank(); i < e; ++i)
      if (dstRT.isDynamicDim(i)) {
        Value ci = arith::ConstantIndexOp::create(rewriter,loc, i);
        dynDims.push_back(tensor::DimOp::create(rewriter,loc, srcT, ci));
      }
    Value dstMem = memref::AllocOp::create(rewriter,loc, dstMR, dynDims);

    FloatAttr scale = q.getScaleAttr();
    IntegerAttr zp = q.getZeroPointAttr();
    IntegerAttr axis = q.getAxisAttr();
    auto round = q.getRoundingAttr();
    auto narrow = q.getNarrowRangeAttr();

    cinm::QuantizeOp::create(rewriter,loc, Type(), srcMem, dstMem, scale, zp,
                                      axis, round, narrow);

    Value dstT = bufferization::ToTensorOp::create(rewriter,loc, dstRT, dstMem,
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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
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
        bufferization::ToBufferOp::create(rewriter,loc, srcMR, srcT, true);

    auto dstMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(dstRT));
    SmallVector<Value> dynDims;
    for (int64_t i = 0, e = dstRT.getRank(); i < e; ++i)
      if (dstRT.isDynamicDim(i)) {
        Value ci = arith::ConstantIndexOp::create(rewriter,loc, i);
        dynDims.push_back(tensor::DimOp::create(rewriter,loc, srcT, ci));
      }
    Value dstMem = memref::AllocOp::create(rewriter,loc, dstMR, dynDims);

    FloatAttr scale = dq.getScaleAttr();
    IntegerAttr zp = dq.getZeroPointAttr();
    IntegerAttr axis = dq.getAxisAttr();

    cinm::DequantizeOp::create(rewriter,loc, Type(), srcMem, dstMem, scale, zp,
                                        axis);

    Value dstT = bufferization::ToTensorOp::create(rewriter,loc, dstRT, dstMem,
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
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
  }
  FailureOr<BaseMemRefType> getBufferType(
      Operation *, Value v, const bufferization::BufferizationOptions &,
      const bufferization::BufferizationState &, SmallVector<Value> &) const {
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return MemRefType::get(rtt.getShape(), rtt.getElementType());
    return failure();
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
        bufferization::ToBufferOp::create(rewriter,loc, inMR, inT, true);

    auto outMR = cast<MemRefType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(outRT));

    SmallVector<Value> dynDims;
    for (int64_t d = 0, e = outRT.getRank(); d < e; ++d)
      if (outRT.isDynamicDim(d)) {
        Value cd = arith::ConstantIndexOp::create(rewriter,loc, d);
        dynDims.push_back(tensor::DimOp::create(rewriter,loc, inT, cd));
      }
    Value outMem = memref::AllocOp::create(rewriter,loc, outMR, dynDims);

    cinm::ActivationKindAttr kindAttr = act.getKindAttr();
    if (!kindAttr) {
      if (auto intAttr = act->getAttrOfType<IntegerAttr>("kind")) {
        auto kind = static_cast<cinm::ActivationKind>(intAttr.getInt());
        kindAttr = cinm::ActivationKindAttr::get(op->getContext(), kind);
      } else {
        return op->emitError("cinm.activate missing 'kind' attribute"),
               failure();
      }
    }

    cinm::ActivateOp::create(rewriter,loc, Type(), kindAttr, inMem, outMem);

    Value outT = bufferization::ToTensorOp::create(rewriter,loc, outRT, outMem,
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

    auto newFor = scf::ForOp::create(rewriter,loc, oldFor.getLowerBound(),
                                              oldFor.getUpperBound(),
                                              oldFor.getStep(), memInitArgs);

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
          Value tview = bufferization::ToTensorOp::create(rewriter,
              loc, tt, newMem, true, true);
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
        scf::YieldOp::create(rewriter,loc, newYields);
      }
    }

    SmallVector<Value> replacements;
    replacements.reserve(newFor->getNumResults());
    rewriter.setInsertionPointAfter(newFor);
    for (auto it : llvm::enumerate(newFor->getResults())) {
      Type wantedT = oldFor->getResult(it.index()).getType();
      Value t = bufferization::ToTensorOp::create(rewriter,
          loc, wantedT, it.value(), true, true);
      replacements.push_back(t);
    }
    rewriter.replaceOp(oldFor, replacements);
    return success();
  }
};

void registerCinmBufferizableOpInterfaces(DialectRegistry &registry) {
  registry.addExtension<::mlir::cinm::CinmDialect>(
      +[](MLIRContext *ctx, ::mlir::cinm::CinmDialect *) {
        ::mlir::cinm::ComputeOp::attachInterface<
            ::mlir::cinm::ComputeBufferizableInterface>(*ctx);
        ::mlir::cinm::GemmOp::attachInterface<
            ::mlir::cinm::GemmBufferizableInterface>(*ctx);
        ::mlir::cinm::GemvOp::attachInterface<
            ::mlir::cinm::GemvBufferizableInterface>(*ctx);
        ::mlir::cinm::ElementwiseOp::attachInterface<
            ::mlir::cinm::ElementwiseBufferizableInterface>(*ctx);
        ::mlir::cinm::QuantizeOp::attachInterface<
            ::mlir::cinm::QuantizeBufferizableInterface>(*ctx);
        ::mlir::cinm::DequantizeOp::attachInterface<
            ::mlir::cinm::DequantizeBufferizableInterface>(*ctx);
        ::mlir::cinm::ActivateOp::attachInterface<
            ::mlir::cinm::ActivateBufferizableInterface>(*ctx);
      });

  registry.addExtension<::mlir::scf::SCFDialect>(
      +[](MLIRContext *ctx, ::mlir::scf::SCFDialect *) {
        ::mlir::scf::ForOp::attachInterface<
            ::mlir::cinm::ScfForBufferizableInterface>(*ctx);
      });
}

} // namespace mlir::cinm
