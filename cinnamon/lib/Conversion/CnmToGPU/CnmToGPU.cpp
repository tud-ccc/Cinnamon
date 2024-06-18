#include "cinm-mlir/Conversion/CnmToGPU/CnmToGPU.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Casting.h>

#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#define GEN_PASS_DEF_CONVERTCNMTOGPUPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
namespace cnmtogpu {
SmallVector<int64_t, 2> getBufferTypeShape(cnm::BufferType bufferType) {
  SmallVector<int64_t> shape{bufferType.getShape()};
  while (shape.size() < bufferType.getWorkgroupShape().size()) {
    shape.insert(shape.begin(), 1);
  }
  return shape;
}

MemRefType convertCnmBufferToMemRefType(cnm::BufferType bufferType) {
  ArrayRef<int64_t> workgroupShape = bufferType.getWorkgroupShape();
  SmallVector<int64_t, 2> shape = getBufferTypeShape(bufferType);
  for (size_t i = 0; i < workgroupShape.size(); i++) {
    shape[i] *= workgroupShape[i];
  }
  return MemRefType::get(shape, bufferType.getElementType());
}

SmallVector<Value, 2> createCalculateScatterIndices(Location loc,
                                                    OpBuilder &builder,
                                                    const AffineMap &scatterMap,
                                                    ValueRange indices,
                                                    BufferType bufferType) {
  SmallVector<Value> bufferIndices;
  ArrayRef<int64_t> workgroupShape = bufferType.getWorkgroupShape();
  for (size_t i = 0; i < workgroupShape.size(); i++) {
    const AffineExpr indexExpr =
        scatterMap.getResult(i) * workgroupShape[i] +
        scatterMap.getResult(workgroupShape.size() + i);
    bufferIndices.push_back(builder.create<affine::AffineApplyOp>(
        loc, AffineMap::get(indices.size(), 0, indexExpr), indices));
  }
  return bufferIndices;
}

void convertLaunchParameter(ConversionPatternRewriter &rewriter, Location loc,
                            Value buffer, ValueRange threadIds,
                            ArrayRef<int64_t> workgroupShape,
                            BlockArgument arg) {
  const BufferType bufferType = buffer.getType().dyn_cast<cnm::BufferType>();
  const SmallVector<int64_t, 2> bufferShape = getBufferTypeShape(bufferType);

  const Value source = createOrFoldUnrealizedConversionCast(
      loc, rewriter, convertCnmBufferToMemRefType(bufferType),
      rewriter.getRemappedValue(buffer));

  const SmallVector<int64_t, 2> staticOffsets(workgroupShape.size(),
                                              ShapedType::kDynamic);
  const SmallVector<int64_t, 2> staticSizes{bufferShape};
  const SmallVector<int64_t, 2> staticStrides(workgroupShape.size(), 1);

  SmallVector<Value, 2> dynamicOffsets;
  for (size_t i = 0; i < workgroupShape.size(); i++) {
    const AffineExpr indexExpr = rewriter.getAffineDimExpr(0) * bufferShape[i];
    dynamicOffsets.push_back(rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, 0, indexExpr), ValueRange{threadIds[i]}));
  }

  const Type resultType = memref::SubViewOp::inferRankReducedResultType(
      bufferType.getShape(), convertCnmBufferToMemRefType(bufferType),
      staticOffsets, staticSizes, staticStrides);

  const Value subview =
      rewriter
          .create<memref::SubViewOp>(loc, resultType, source, dynamicOffsets,
                                     ValueRange{}, ValueRange{}, staticOffsets,
                                     staticSizes, staticStrides)
          .getResult();

  arg.replaceAllUsesWith(subview);
}

struct ConvertCnmWorkgroupToGPU : public OpConversionPattern<cnm::WorkgroupOp> {
  using OpConversionPattern<cnm::WorkgroupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::WorkgroupOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }
};

struct ConvertCnmAllocToGPU : public OpConversionPattern<cnm::AllocOp> {
  using OpConversionPattern<cnm::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::AllocOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, convertCnmBufferToMemRefType(op.getType()));
    return success();
  }
};

struct ConvertCnmScatterToGPU : public OpConversionPattern<cnm::ScatterOp> {
  using OpConversionPattern<cnm::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::ScatterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const cnm::BufferType bufferType =
        op.getOperandTypes()[1].dyn_cast<cnm::BufferType>();
    const SmallVector<int64_t, 2> bufferShape = getBufferTypeShape(bufferType);

    Value memref = rewriter.getRemappedValue(op.getOperand(1));
    memref = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, convertCnmBufferToMemRefType(bufferType),
        memref);

    const Value tensor = op.getOperand(0);
    const RankedTensorType tensorType =
        tensor.getType().dyn_cast<RankedTensorType>();

    SmallVector<affine::AffineForOp, 2> loops;
    SmallVector<Value> indices;

    for (int64_t size : tensorType.getShape()) {
      affine::AffineForOp loop =
          rewriter.create<affine::AffineForOp>(op.getLoc(), 0, size, 1);
      loops.push_back(loop);
      indices.push_back(loop.getBody()->getArgument(0));
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // inner most loop body
    const AffineMap scatterMap = op.getScatterMap();
    SmallVector<Value> bufferIndices = createCalculateScatterIndices(
        op.getLoc(), rewriter, scatterMap, indices, bufferType);

    const Value element =
        rewriter.create<tensor::ExtractOp>(op.getLoc(), tensor, indices);
    rewriter.create<memref::StoreOp>(op.getLoc(), element, memref,
                                     bufferIndices);

    // replace token with const 0
    rewriter.setInsertionPointAfter(loops[0]);
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);

    return success();
  }
};

struct ConvertCnmGatherToGPU : public OpConversionPattern<cnm::GatherOp> {
  using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::GatherOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const cnm::BufferType bufferType =
        op.getOperandTypes()[0].dyn_cast<cnm::BufferType>();
    const SmallVector<int64_t, 2> bufferShape = getBufferTypeShape(bufferType);

    Value memref = rewriter.getRemappedValue(op.getOperand(0));
    memref = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, convertCnmBufferToMemRefType(bufferType),
        memref);

    const RankedTensorType tensorType =
        op.getResultTypes()[0].cast<RankedTensorType>();
    const Value tensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType());

    SmallVector<affine::AffineForOp, 2> loops;
    SmallVector<Value> indices;

    for (int64_t size : tensorType.getShape()) {
      const Value iterArg =
          loops.empty() ? tensor : loops.back().getBody()->getArgument(1);
      affine::AffineForOp loop = rewriter.create<affine::AffineForOp>(
          op.getLoc(), 0, size, 1, SmallVector<Value, 1>{iterArg});
      indices.push_back(loop.getBody()->getArgument(0));

      if (!loops.empty()) {
        rewriter.create<affine::AffineYieldOp>(op.getLoc(), loop.getResult(0));
      }

      rewriter.setInsertionPointToStart(loop.getBody());
      loops.push_back(loop);
    }

    // inner most loop body
    const Value iterArg = loops.back().getBody()->getArgument(1);

    const AffineMap gatherMap = op.getGatherMap();
    SmallVector<Value> bufferIndices = createCalculateScatterIndices(
        op.getLoc(), rewriter, gatherMap, indices, bufferType);
    const Value element =
        rewriter.create<memref::LoadOp>(op.getLoc(), memref, bufferIndices);
    const Value result = rewriter.create<tensor::InsertOp>(op.getLoc(), element,
                                                           iterArg, indices);
    rewriter.create<affine::AffineYieldOp>(op.getLoc(), result);

    // replace token with const 0
    rewriter.setInsertionPointAfter(loops[0]);
    const Value token = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    rewriter.replaceOp(op, {loops.front().getResult(0), token});

    return success();
  }
};

struct ConvertCnmLaunchToGPU : public OpConversionPattern<cnm::LaunchOp> {
  using OpConversionPattern<cnm::LaunchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::LaunchOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const WorkgroupType workgroupType = op.getWg().getType();
    const ArrayRef<int64_t> workgroupShape = workgroupType.getShape();

    const Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    const Value gridSizeX = one, gridSizeY = one, gridSizeZ = one;
    const Value blockSizeX =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), workgroupShape[0]);
    const Value blockSizeY =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), workgroupShape[1]);
    const Value blockSizeZ = one;

    const Value dynamicSharedMemorySize;
    const Type asyncTokenType;
    const ValueRange asyncDependencies;
    const TypeRange workgroupAttributions;
    const TypeRange privateAttributions;

    gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(
        op.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY,
        blockSizeZ, dynamicSharedMemorySize, asyncTokenType, asyncDependencies,
        workgroupAttributions, privateAttributions);

    const SmallVector<Value, 3> threadIds{
        launchOp.getThreadIds().x,
        launchOp.getThreadIds().y,
        launchOp.getThreadIds().z,
    };

    rewriter.setInsertionPointToEnd(&launchOp.getBody().front());

    // convert cnm.buffer parameters to memref subviews
    int64_t i = 0;
    for (const Value &buffer : op.getParams()) {
      convertLaunchParameter(rewriter, op.getLoc(), buffer, threadIds,
                             workgroupShape, op.getBody().getArgument(i++));
    }

    launchOp.getBody().front().getOperations().splice(
        launchOp.getBody().front().end(), op.getBody().front().getOperations());

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCnmTerminatorToGPU
    : public OpConversionPattern<cnm::TerminatorOp> {
  using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const ValueRange values;
    rewriter.replaceOpWithNewOp<gpu::TerminatorOp>(op, values);
    return success();
  }
};
} // namespace cnmtogpu

void populateCnmToGPUFinalTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](cnm::BufferType bufferType) -> std::optional<Type> {
        return cnmtogpu::convertCnmBufferToMemRefType(bufferType);
      });
}

void populateCnmToGPUConversionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  patterns
      .add<cnmtogpu::ConvertCnmWorkgroupToGPU, cnmtogpu::ConvertCnmAllocToGPU,
           ConvertCnmSetZeroToAffine, cnmtogpu::ConvertCnmScatterToGPU,
           cnmtogpu::ConvertCnmGatherToGPU, cnmtogpu::ConvertCnmLaunchToGPU,
           cnmtogpu::ConvertCnmTerminatorToGPU>(&typeConverter.getContext());
}

struct ConvertCnmToGPUPass
    : public ::impl::ConvertCnmToGPUPassBase<ConvertCnmToGPUPass> {
  void runOnOperation() final {
    TypeConverter converter(&getContext());
    populateCnmToGPUFinalTypeConversions(converter);
    const auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateCnmToGPUConversionPatterns(converter, patterns);
    populateReconcileUnrealizedCastsPatterns(patterns);

    ConversionTarget target(getContext());
    // target.addIllegalDialect<cnm::CnmDialect>();
    target.addIllegalOp<cnm::WorkgroupOp>();
    target.addIllegalOp<cnm::AllocOp>();
    target.addIllegalOp<cnm::SetZeroOp>();
    target.addIllegalOp<cnm::ScatterOp>();
    target.addIllegalOp<cnm::GatherOp>();
    target.addIllegalOp<cnm::LaunchOp>();
    target.addIllegalOp<cnm::TerminatorOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createConvertCnmToGPUPass() {
  return std::make_unique<ConvertCnmToGPUPass>();
}
} // namespace mlir::cnm
