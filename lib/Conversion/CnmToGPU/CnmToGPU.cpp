#include "cinm-mlir/Conversion/CnmToGPU/CnmToGPU.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <cstdint>
#include <iostream>
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
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
    MemRefType getMemrefType(cnm::BufferType bufferType) {
        SmallVector<int64_t> shape{bufferType.getShape()};
        shape.front() = ShapedType::kDynamic;
        return MemRefType::get(shape, bufferType.getElementType());
    }

    MemRefType getMemrefType(cnm::BufferType bufferType, cnm::WorkgroupType workgroupType) {
        SmallVector<int64_t> shape{bufferType.getShape()};
        for (const int64_t s : workgroupType.getShape()) {
            shape.front() *= s;
        }
        return MemRefType::get(shape, bufferType.getElementType());
    }

    AffineMap getBufferOffsetAffineMap(MLIRContext *ctx, BufferType bufferType, WorkgroupType workgroupType) {
        const int64_t strideX = bufferType.getShape()[0];
        const int64_t strideY = workgroupType.getShape()[0];
        const int64_t strideZ = workgroupType.getShape()[1];

        // (d0, d1, d2) -> (strideX * (d0 + strideY * (d1 + strideZ * d2)))
        return AffineMap::get(3, 0, getAffineConstantExpr(strideX, ctx) * (
            getAffineDimExpr(0, ctx) + getAffineConstantExpr(strideY, ctx) * (
                getAffineDimExpr(1, ctx) + getAffineConstantExpr(strideZ, ctx) * getAffineDimExpr(2, ctx)
            )
        ));
    }

    struct ConvertCnmWorkgroupToGPU : public OpConversionPattern<cnm::WorkgroupOp> {
        using OpConversionPattern<cnm::WorkgroupOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(cnm::WorkgroupOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
            return success();
        }
    };

    struct ConvertCnmAllocToGPU : public OpConversionPattern<cnm::AllocOp> {
        using OpConversionPattern<cnm::AllocOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(cnm::AllocOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            const BufferType bufferType = op.getType();
            const WorkgroupType workgroupType = op.getWg().getType();
            const MemRefType allocType = getMemrefType(bufferType, workgroupType);
            rewriter.replaceOp(op, rewriter.create<memref::AllocOp>(op.getLoc(), allocType));
            return success();
        }
    };

    struct ConvertCnmSetZeroToGPU : public OpConversionPattern<cnm::SetZeroOp> {
        using OpConversionPattern<cnm::SetZeroOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(cnm::SetZeroOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            const Type asyncToken;
            const ValueRange asyncDependencies;
            const Value dst = rewriter.getRemappedValue(op.getOperand());
            const Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getZeroAttr(op.getType().getElementType()));
            (void)rewriter.create<gpu::MemsetOp>(op.getLoc(), asyncToken, asyncDependencies, dst, zero);
            rewriter.replaceOp(op, {dst});
            return success();
        }
    };

    struct ConvertCnmScatterToGPU : public OpConversionPattern<cnm::ScatterOp> {
        using OpConversionPattern<cnm::ScatterOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(cnm::ScatterOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            const cnm::BufferType bufferType = op.getOperandTypes()[1].dyn_cast<cnm::BufferType>();
            const cnm::WorkgroupType workgroupType = op.getOperandTypes()[2].dyn_cast<cnm::WorkgroupType>();

            Value memref = rewriter.getRemappedValue(op.getOperand(1));
            if (memref.getType() != getMemrefType(bufferType, workgroupType)) {
                SmallVector<Value, 1> tmp;
                rewriter.createOrFold<UnrealizedConversionCastOp>(tmp, op.getLoc(), TypeRange{getMemrefType(bufferType, workgroupType)}, ValueRange{memref});
                memref = tmp[0];
            }

            const Value tensor = op.getOperand(0);
            const RankedTensorType tensorType = tensor.getType().dyn_cast<RankedTensorType>();

            SmallVector<affine::AffineForOp, 2> loops;
            SmallVector<Value> indices;

            for (int64_t size : tensorType.getShape()) {
                affine::AffineForOp loop = rewriter.create<affine::AffineForOp>(op.getLoc(), 0, size, 1);
                loops.push_back(loop);
                indices.push_back(loop.getBody()->getArgument(0));
                rewriter.setInsertionPointToStart(loop.getBody());
            }

            // inner most loop body
            const AffineMap map = op.getOperation()->getAttr("map").dyn_cast<AffineMapAttr>().getAffineMap();
            SmallVector<Value> workgroupIndices;
            for (size_t i = 0; i < workgroupType.getShape().size(); i++) {
                workgroupIndices.push_back(rewriter.create<affine::AffineApplyOp>(op.getLoc(), map.getSubMap(i), indices));
            }
            workgroupIndices.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));

            AffineMap workgroupOffsetMap = getBufferOffsetAffineMap(op.getContext(), bufferType, workgroupType);
            Value workgroupOffset = rewriter.create<affine::AffineApplyOp>(op.getLoc(), workgroupOffsetMap, workgroupIndices);

            SmallVector<Value> bufferIndices;
            for (size_t i = 0; i < bufferType.getShape().size(); i++) {
                bufferIndices.push_back(rewriter.create<affine::AffineApplyOp>(op.getLoc(), map.getSubMap(i + workgroupType.getShape().size()), indices));
            }
            bufferIndices.front() = rewriter.create<arith::MulIOp>(op.getLoc(), workgroupOffset, bufferIndices.front());

            const Value element = rewriter.create<tensor::ExtractOp>(op.getLoc(), tensor, indices);
            rewriter.create<memref::StoreOp>(op.getLoc(), element, memref, bufferIndices);

            // replace token with const 0
            rewriter.setInsertionPointAfter(loops[0]);
            rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);

            return success();
        }
    };

    struct ConvertCnmGatherToGPU : public OpConversionPattern<cnm::GatherOp> {
        using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(cnm::GatherOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            const cnm::BufferType bufferType = op.getOperandTypes()[0].dyn_cast<cnm::BufferType>();
            const cnm::WorkgroupType workgroupType = op.getOperandTypes()[1].dyn_cast<cnm::WorkgroupType>();

            Value memref = rewriter.getRemappedValue(op.getOperand(0));
            if (memref.getType() != getMemrefType(bufferType, workgroupType)) {
                SmallVector<Value, 1> tmp;
                rewriter.createOrFold<UnrealizedConversionCastOp>(tmp, op.getLoc(), TypeRange{getMemrefType(bufferType, workgroupType)}, ValueRange{memref});
                memref = tmp[0];
            }

            const RankedTensorType tensorType = op.getResultTypes()[0].cast<RankedTensorType>();
            const Value tensor = rewriter.create<bufferization::AllocTensorOp>(op.getLoc(), tensorType, ValueRange{}).getResult();

            SmallVector<affine::AffineForOp, 2> loops;
            SmallVector<Value> indices;

            for (int64_t size : tensorType.getShape()) {
                const Value iterArg = loops.empty() ? tensor : loops.back().getBody()->getArgument(1);
                affine::AffineForOp loop = rewriter.create<affine::AffineForOp>(op.getLoc(), 0, size, 1, SmallVector<Value, 1>{iterArg});
                indices.push_back(loop.getBody()->getArgument(0));

                if (!loops.empty()) {
                    rewriter.create<affine::AffineYieldOp>(op.getLoc(), loop.getResult(0));
                }

                rewriter.setInsertionPointToStart(loop.getBody());
                loops.push_back(loop);
            }

            // inner most loop body
            const Value iterArg = loops.back().getBody()->getArgument(1);

            const AffineMap map = op.getOperation()->getAttr("map").dyn_cast<AffineMapAttr>().getAffineMap();
            SmallVector<Value> workgroupIndices;
            for (size_t i = 0; i < workgroupType.getShape().size(); i++) {
                workgroupIndices.push_back(rewriter.create<affine::AffineApplyOp>(op.getLoc(), map.getSubMap(i), indices));
            }
            workgroupIndices.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));

            AffineMap workgroupOffsetMap = getBufferOffsetAffineMap(op.getContext(), bufferType, workgroupType);
            Value workgroupOffset = rewriter.create<affine::AffineApplyOp>(op.getLoc(), workgroupOffsetMap, workgroupIndices);

            SmallVector<Value> bufferIndices;
            for (size_t i = 0; i < bufferType.getShape().size(); i++) {
                bufferIndices.push_back(rewriter.create<affine::AffineApplyOp>(op.getLoc(), map.getSubMap(i + workgroupType.getShape().size()), indices));
            }
            bufferIndices.front() = rewriter.create<arith::MulIOp>(op.getLoc(), workgroupOffset, bufferIndices.front());

            const Value element = rewriter.create<memref::LoadOp>(op.getLoc(), memref, bufferIndices);
            const Value result = rewriter.create<tensor::InsertOp>(op.getLoc(), element, iterArg, indices);
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

        LogicalResult matchAndRewrite(cnm::LaunchOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            const WorkgroupType workgroupType = op.getWg().getType();
            const ArrayRef<int64_t> workgroupShape = workgroupType.getShape();

            const Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
            Value gridSizeX = one, gridSizeY = one, gridSizeZ = one;
            Value blockSizeX = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), workgroupShape[0]);
            Value blockSizeY = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), workgroupShape[1]);
            Value blockSizeZ = one;

            Value dynamicSharedMemorySize;
            Type asyncTokenType;
            ValueRange asyncDependencies;
            TypeRange workgroupAttributions;
            TypeRange privateAttributions;

            gpu::LaunchOp launchOp = rewriter.create<gpu::LaunchOp>(op.getLoc(),
                gridSizeX, gridSizeY, gridSizeZ,
                blockSizeX, blockSizeY, blockSizeZ,
                dynamicSharedMemorySize,
                asyncTokenType, asyncDependencies,
                workgroupAttributions, privateAttributions
            );

            rewriter.setInsertionPointToEnd(&launchOp.getBody().front());

            // convert cnm.buffer parameters to memref subviews
            for (size_t i = 0; i < op.getParams().size(); i++) {
                const Value param = op.getParams()[i];
                const BufferType bufferType = param.getType().dyn_cast_or_null<cnm::BufferType>();
                if (!bufferType) {
                    op.getBody().getArgument(i).replaceAllUsesWith(param);
                    continue;
                }

                Value source = rewriter.getRemappedValue(param);
                if (source.getType() != getMemrefType(bufferType, workgroupType)) {
                    SmallVector<Value, 1> tmp;
                    rewriter.createOrFold<UnrealizedConversionCastOp>(tmp, op.getLoc(), TypeRange{getMemrefType(bufferType, workgroupType)}, ValueRange{source});
                    source = tmp[0];
                }

                SmallVector<int64_t> staticOffsets(bufferType.getShape().size(), 0);
                SmallVector<int64_t> staticSizes{bufferType.getShape()};
                SmallVector<int64_t> staticStrides(bufferType.getShape().size(), 1);

                staticOffsets.front() = ShapedType::kDynamic;
                Value dynamicOffset = rewriter.create<affine::AffineApplyOp>(launchOp.getLoc(),
                    getBufferOffsetAffineMap(op.getContext(), bufferType, workgroupType),
                    SmallVector<Value, 3>{
                        launchOp.getThreadIds().x,
                        launchOp.getThreadIds().y,
                        launchOp.getThreadIds().z
                    }
                );

                Type resultType = memref::SubViewOp::inferRankReducedResultType(bufferType.getShape(), getMemrefType(bufferType, workgroupType), staticOffsets, staticSizes, staticStrides);

                const Value subview = rewriter.create<memref::SubViewOp>(launchOp.getLoc(), resultType, source,
                    ValueRange{dynamicOffset}, ValueRange{}, ValueRange{},
                    staticOffsets, staticSizes, staticStrides
                ).getResult();

                op.getBody().getArgument(i).replaceAllUsesWith(subview);
            }

            launchOp.getBody().front().getOperations().splice(
                launchOp.getBody().front().end(),
                op.getBody().front().getOperations()
            );

            rewriter.eraseOp(op);
            return success();
        }
    };

    struct ConvertCnmTerminatorToGPU : public OpConversionPattern<cnm::TerminatorOp> {
        using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(cnm::TerminatorOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
            const ValueRange values;
            rewriter.replaceOpWithNewOp<gpu::TerminatorOp>(op, values);
            return success();
        }
    };

    void populateCnmToGPUFinalTypeConversions(LLVMTypeConverter &typeConverter) {
        typeConverter.addConversion(
            [&](cnm::WorkgroupType) -> Type {
                return IndexType::get(&typeConverter.getContext());
            }
        );

        typeConverter.addConversion(
            [&](cnm::BufferType t) -> Type {
                return getMemrefType(t);
            }
        );

        typeConverter.addConversion(
            [&](cnm::LaunchTokenType) -> Type {
                return IndexType::get(&typeConverter.getContext());
            }
        );

        typeConverter.addConversion(
            [&](cnm::ScatterTokenType) -> Type {
                return IndexType::get(&typeConverter.getContext());
            }
        );

        typeConverter.addConversion(
            [&](cnm::GatherTokenType) -> Type {
                return IndexType::get(&typeConverter.getContext());
            }
        );
    }

    void populateCnmToGPUConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
        patterns.add<
            ConvertCnmWorkgroupToGPU,
            ConvertCnmAllocToGPU,
            ConvertCnmSetZeroToGPU,
            ConvertCnmScatterToGPU,
            ConvertCnmGatherToGPU,
            ConvertCnmLaunchToGPU,
            ConvertCnmTerminatorToGPU
        >(&typeConverter.getContext());
    }

    struct ConvertCnmToGPUPass : public ConvertCnmToGPUPassBase<ConvertCnmToGPUPass> {
        void runOnOperation() final {
            LLVMTypeConverter converter(&getContext());
            populateCnmToGPUFinalTypeConversions(converter);
            const auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
                return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
            };
            converter.addSourceMaterialization(addUnrealizedCast);
            converter.addTargetMaterialization(addUnrealizedCast);

            RewritePatternSet patterns(&getContext());
            populateCnmToGPUConversionPatterns(converter, patterns);
            populateReconcileUnrealizedCastsPatterns(patterns);

            ConversionTarget target(getContext());
            //target.addIllegalDialect<cnm::CnmDialect>();
            target.addIllegalOp<cnm::WorkgroupOp>();
            target.addIllegalOp<cnm::AllocOp>();
            target.addIllegalOp<cnm::SetZeroOp>();
            target.addIllegalOp<cnm::ScatterOp>();
            target.addIllegalOp<cnm::GatherOp>();
            target.addIllegalOp<cnm::LaunchOp>();
            target.addIllegalOp<cnm::TerminatorOp>();

            target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

            if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
                signalPassFailure();
            }
        }
    };

    std::unique_ptr<Pass> createConvertCnmToGPUPass() {
        return std::make_unique<ConvertCnmToGPUPass>();
    }
}
