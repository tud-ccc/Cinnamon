#include "cinm-mlir/Conversion/CommonPatterns.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

namespace {

struct DeleteToMemref : public OpConversionPattern<bufferization::ToMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::ToMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {

    auto toCast = createOrFoldUnrealizedConversionCast(
        op->getLoc(), rewriter0, op.getResult().getType(), adaptor.getTensor());

    rewriter0.replaceAllUsesWith(op.getResult(), toCast);
    rewriter0.eraseOp(op);
    return success();
  }
};
struct DeleteToTensor : public OpConversionPattern<bufferization::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {

    auto toCast = createOrFoldUnrealizedConversionCast(
        op->getLoc(), rewriter0, op.getResult().getType(), adaptor.getMemref());

    rewriter0.replaceAllUsesWith(op.getResult(), toCast);
    rewriter0.eraseOp(op);
    return success();
  }
};
} // namespace

void populateFinalBufferizationPatterns(RewritePatternSet &set) {
  set.insert<DeleteToMemref>(set.getContext());
  set.insert<DeleteToTensor>(set.getContext());
}

Value createOrFoldUnrealizedConversionCast(Location loc, OpBuilder &builder,
                                           Type dstType, Value value) {
  if (value.getType() == dstType) {
    return value;
  }

  SmallVector<Value, 1> tmp;
  builder.createOrFold<UnrealizedConversionCastOp>(tmp, loc, TypeRange{dstType},
                                                   ValueRange{value});
  return tmp[0];
}

SmallVector<Value> createNestedAffineForLoops(OpBuilder &builder, Location loc,
                                              ArrayRef<int64_t> loopSizes,
                                              ArrayRef<int64_t> loopSteps,
                                              ValueRange iterArgsInit,
                                              BodyBuilderCallback bodyBuilder) {
  assert(loopSizes.size() == loopSteps.size());

  SmallVector<affine::AffineForOp> loops;
  SmallVector<Value> indices;
  ValueRange iterArgs = iterArgsInit;

  for (auto [size, step] : llvm::zip(loopSizes, loopSteps)) {
    affine::AffineForOp current =
        builder.create<affine::AffineForOp>(loc, 0, size, step, iterArgs);
    if (!loops.empty() && !iterArgs.empty()) {
      builder.create<affine::AffineYieldOp>(loc, current.getResults());
    }
    loops.push_back(current);
    indices.push_back(current.getRegion().front().getArguments().front());
    iterArgs = current.getRegion().front().getArguments().drop_front();
    builder.setInsertionPointToStart(&current.getRegion().front());
  }

  SmallVector<Value> result = bodyBuilder(builder, loc, indices, iterArgs);
  if (!iterArgs.empty()) {
    builder.create<affine::AffineYieldOp>(loc, result);
  }

  builder.setInsertionPointAfter(loops.front());
  return loops.front().getResults();
}

LogicalResult ConvertCnmSetZeroToAffine::matchAndRewrite(
    cnm::SetZeroOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const {
  const Value dst = rewriter.getRemappedValue(op.getOperand());

  const MemRefType type = cast<MemRefType>(dst.getType());
  const SmallVector<int64_t> loopSizes{type.getShape()};
  const SmallVector<int64_t> loopSteps(loopSizes.size(), 1);

  createNestedAffineForLoops(
      rewriter, op.getLoc(), loopSizes, loopSteps, ValueRange{},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange) -> SmallVector<Value> {
        const Value zero = builder.create<arith::ConstantOp>(
            loc, builder.getZeroAttr(op.getType().getElementType()));
        rewriter.create<memref::StoreOp>(loc, zero, dst, indices);
        return {};
      });

  rewriter.replaceOp(op, {dst});
  return success();
}

SmallVector<Value> createAffineApply(OpBuilder &builder, Location loc,
                                     AffineMap map, ValueRange values) {
  SmallVector<Value> result;
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    result.push_back(
        builder.create<affine::AffineApplyOp>(loc, map.getSubMap({i}), values));
  }
  return result;
}

void createMemrefSubviewCopy(OpBuilder &builder, Location loc, Value src,
                             Value dst, ArrayRef<int64_t> sliceShape,
                             ValueRange srcOffsets, ValueRange dstOffsets) {
  MemRefType srcType = cast<MemRefType>(src.getType());
  MemRefType dstType = cast<MemRefType>(dst.getType());

  SmallVector<int64_t> srcStaticOffsets(srcType.getRank(), 0);
  SmallVector<int64_t> srcStaticSizes{srcType.getShape()};
  SmallVector<int64_t> srcStaticStrides(srcType.getRank(), 1);
  for (unsigned i = 0; i < srcOffsets.size(); i++) {
    srcStaticSizes[i] = 1;
    srcStaticOffsets[i] = ShapedType::kDynamic;
  }

  SmallVector<int64_t> dstStaticOffsets(dstType.getRank(), 0);
  SmallVector<int64_t> dstStaticSizes{dstType.getShape()};
  SmallVector<int64_t> dstStaticStrides(dstType.getRank(), 1);
  for (unsigned i = 0; i < dstOffsets.size(); i++) {
    dstStaticSizes[i] = 1;
    dstStaticOffsets[i] = ShapedType::kDynamic;
  }

  const Type sliceType = memref::SubViewOp::inferRankReducedResultType(
      sliceShape, dstType, dstStaticOffsets, dstStaticSizes, dstStaticStrides);

  const Value src_slice = builder.create<memref::SubViewOp>(
      loc, sliceType, src, srcOffsets, ValueRange{}, ValueRange{},
      srcStaticOffsets, srcStaticSizes, srcStaticStrides);
  const Value dst_slice = builder.create<memref::SubViewOp>(
      loc, sliceType, dst, dstOffsets, ValueRange{}, ValueRange{},
      dstStaticOffsets, dstStaticSizes, dstStaticStrides);

  builder.create<memref::CopyOp>(loc, src_slice, dst_slice);
}

} // namespace mlir
