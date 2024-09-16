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
#include <mlir/IR/Location.h>
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
MemRefType convertCnmBufferToMemRefType(cnm::BufferType bufferType) {
  SmallVector<int64_t> shape{bufferType.getWorkgroupShape()};
  shape.append(bufferType.getShape().begin(), bufferType.getShape().end());
  return MemRefType::get(shape, bufferType.getElementType());
}

void convertLaunchParameter(ConversionPatternRewriter &rewriter, Location loc,
                            Value buffer, ValueRange threadIds,
                            BlockArgument arg) {
  const BufferType bufferType = buffer.getType().dyn_cast<cnm::BufferType>();
  const MemRefType memrefType = convertCnmBufferToMemRefType(bufferType);

  const Value source = createOrFoldUnrealizedConversionCast(
      loc, rewriter, convertCnmBufferToMemRefType(bufferType),
      rewriter.getRemappedValue(buffer));

  SmallVector<int64_t> staticOffsets(memrefType.getRank(), 0);
  SmallVector<int64_t> staticSizes{memrefType.getShape()};
  const SmallVector<int64_t> staticStrides(memrefType.getRank(), 1);
  for (unsigned i = 0; i < threadIds.size(); i++) {
    staticSizes[i] = 1;
    staticOffsets[i] = ShapedType::kDynamic;
  }

  const Type resultType = memref::SubViewOp::inferRankReducedResultType(
      bufferType.getShape(), convertCnmBufferToMemRefType(bufferType),
      staticOffsets, staticSizes, staticStrides);

  const Value subview =
      rewriter
          .create<memref::SubViewOp>(loc, resultType, source, threadIds,
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
    Type asyncToken;
    ValueRange asyncDependencies;
    ValueRange dynamicSizes;
    ValueRange symbolOperands;
    UnitAttr hostShared;

    rewriter.replaceOpWithNewOp<gpu::AllocOp>(
        op, convertCnmBufferToMemRefType(op.getType()), asyncToken,
        asyncDependencies, dynamicSizes, symbolOperands, hostShared);
    return success();
  }
};

struct ConvertCnmScatterToGPU : public OpConversionPattern<cnm::ScatterOp> {
  using OpConversionPattern<cnm::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::ScatterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const WorkgroupType workgroupType = op.getWg().getType();
    const ArrayRef<int64_t> workgroupShape = workgroupType.getShape();
    const cnm::BufferType bufferType =
        op.getOperandTypes()[1].dyn_cast<cnm::BufferType>();

    Value src = rewriter.getRemappedValue(op.getOperand(0));
    Value dst = rewriter.getRemappedValue(op.getOperand(1));
    dst = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, convertCnmBufferToMemRefType(bufferType), dst);

    const SmallVector<int64_t> loopSteps(workgroupShape.size(), 1);
    createNestedAffineForLoops(
        rewriter, op.getLoc(), workgroupShape, loopSteps, ValueRange{},
        [&](OpBuilder &builder, Location loc, ValueRange indices,
            ValueRange) -> SmallVector<Value> {
          const SmallVector<Value> mappedIndices =
              createAffineApply(builder, loc, op.getScatterMap(), indices);
          createMemrefSubviewCopy(builder, loc, src, dst, bufferType.getShape(),
                                  mappedIndices, indices);
          return {};
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCnmGatherToGPU : public OpConversionPattern<cnm::GatherOp> {
  using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::GatherOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const WorkgroupType workgroupType = op.getWg().getType();
    const ArrayRef<int64_t> workgroupShape = workgroupType.getShape();
    const cnm::BufferType bufferType =
        op.getOperandTypes()[0].dyn_cast<cnm::BufferType>();

    Value src = rewriter.getRemappedValue(op.getOperand(0));
    src = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, convertCnmBufferToMemRefType(bufferType), src);
    Value dst = rewriter.getRemappedValue(op.getOperand(2));

    const SmallVector<int64_t> loopSteps(workgroupShape.size(), 1);
    createNestedAffineForLoops(
        rewriter, op.getLoc(), workgroupShape, loopSteps, ValueRange{},
        [&](OpBuilder &builder, Location loc, ValueRange indices,
            ValueRange) -> SmallVector<Value> {
          const SmallVector<Value> mappedIndices =
              createAffineApply(builder, loc, op.getGatherMap(), indices);
          createMemrefSubviewCopy(builder, loc, src, dst, bufferType.getShape(),
                                  indices, mappedIndices);
          return {};
        });

    rewriter.eraseOp(op);
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
    SmallVector<Value, 6> launchDimensions(6, one);
    for (size_t i = 0; i < workgroupShape.size(); i++) {
      launchDimensions[i] = rewriter.create<arith::ConstantIndexOp>(
          op.getLoc(), workgroupShape[i]);
    }

    const Value dynamicSharedMemorySize;
    const Type asyncTokenType;
    const ValueRange asyncDependencies;
    const TypeRange workgroupAttributions;
    const TypeRange privateAttributions;

    gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(
        op.getLoc(), launchDimensions[0], launchDimensions[1],
        launchDimensions[2], launchDimensions[3], launchDimensions[4],
        launchDimensions[5], dynamicSharedMemorySize, asyncTokenType,
        asyncDependencies, workgroupAttributions, privateAttributions);

    const SmallVector<Value, 6> allThreadIds{
        launchOp.getBlockIds().x,  launchOp.getBlockIds().y,
        launchOp.getBlockIds().z,  launchOp.getThreadIds().x,
        launchOp.getThreadIds().y, launchOp.getThreadIds().z,
    };
    const ValueRange usedThreadIds =
        ValueRange{allThreadIds}.take_front(workgroupShape.size());

    rewriter.setInsertionPointToEnd(&launchOp.getBody().front());

    // convert cnm.buffer parameters to memref subviews
    size_t i = 0;
    for (const Value &buffer : op.getParams()) {
      convertLaunchParameter(rewriter, op.getLoc(), buffer, usedThreadIds,
                             op.getBody().getArgument(i++));
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

  typeConverter.addConversion([&](cnm::WorkgroupType t) -> std::optional<Type> {
    return IndexType::get(t.getContext());
  });
}

void populateCnmToGPUConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns
      .add<cnmtogpu::ConvertCnmWorkgroupToGPU, cnmtogpu::ConvertCnmAllocToGPU,
           ConvertCnmSetZeroToAffine, cnmtogpu::ConvertCnmScatterToGPU,
           cnmtogpu::ConvertCnmGatherToGPU, cnmtogpu::ConvertCnmLaunchToGPU,
           cnmtogpu::ConvertCnmTerminatorToGPU>(ctx);
}

struct ConvertCnmToGPUPass
    : public ::impl::ConvertCnmToGPUPassBase<ConvertCnmToGPUPass> {
  void runOnOperation() final {
    TypeConverter converter;
    populateCnmToGPUFinalTypeConversions(converter);
    const auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateCnmToGPUConversionPatterns(patterns, &getContext());
    populateReconcileUnrealizedCastsPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<cnm::CnmDialect>();
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
