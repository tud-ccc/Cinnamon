#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ValueRange.h>

#define GEN_PASS_DEF_CONVERTCNMTOUPMEMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
namespace cnmtoupmem {

template <typename T> T reduceMul(ArrayRef<T> arr) {
  T result{1};
  for (const T &elem : arr) {
    result *= elem;
  }
  return result;
}

MemRefType convertCnmBufferToMemRefType(cnm::BufferType bufferType) {
  assert(bufferType.getWorkgroupShape().size() == 4);
  SmallVector<int64_t, 3> shape{bufferType.getWorkgroupShape().take_front(2)};
  const int64_t perDPUMemory =
      reduceMul(bufferType.getWorkgroupShape().take_back(2)) *
      reduceMul(bufferType.getShape());
  shape.push_back(perDPUMemory);
  return MemRefType::get(shape, bufferType.getElementType());
}

AffineMap getElementToMRAMOffsetAffineMap(MLIRContext *ctx,
                                          size_t chunksPerTasklet,
                                          size_t chunkSize) {
  const AffineExpr tasklet = getAffineDimExpr(0, ctx);
  const AffineExpr chunk = getAffineDimExpr(1, ctx);
  const AffineExpr element = getAffineDimExpr(2, ctx);
  const AffineExpr result =
      tasklet * (chunksPerTasklet * chunkSize) + chunk * chunkSize + element;
  return AffineMap::get(3, 0, result);
}

AffineMap getMRAMOffsetToElementAffineMap(MLIRContext *ctx,
                                          size_t chunksPerTasklet,
                                          size_t chunkSize) {
  const AffineExpr offset = getAffineDimExpr(0, ctx);
  const AffineExpr tasklet = offset.floorDiv(chunkSize * chunksPerTasklet);
  const AffineExpr chunk = offset.floorDiv(chunkSize) % chunksPerTasklet;
  const AffineExpr element = offset % chunkSize;
  return AffineMap::get(1, 0, {tasklet, chunk, element}, ctx);
}

struct ConvertCnmWorkgroupToUPMEM
    : public OpConversionPattern<cnm::WorkgroupOp> {
  using OpConversionPattern<cnm::WorkgroupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::WorkgroupOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getType().getShape().size() == 4);
    const upmem::DeviceHierarchyType hierarchy =
        upmem::DeviceHierarchyType::get(getContext(),
                                        op.getType().getShape().take_front(3));
    rewriter.replaceOpWithNewOp<upmem::AllocDPUsOp>(op, hierarchy);
    return success();
  }
};

struct ConvertCnmAllocToUPMEM : public OpConversionPattern<cnm::AllocOp> {
  using OpConversionPattern<cnm::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::AllocOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, convertCnmBufferToMemRefType(op.getType()));
    return success();
  }
};

struct ConvertCnmScatterToUPMEM : public OpConversionPattern<cnm::ScatterOp> {
  using OpConversionPattern<cnm::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::ScatterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Value tensor = op.getOperand(0);
    const Value buffer = op.getOperand(1);
    const RankedTensorType tensorType =
        tensor.getType().cast<RankedTensorType>();
    const BufferType bufferType = buffer.getType().cast<BufferType>();
    const MemRefType memrefType = convertCnmBufferToMemRefType(bufferType);
    const AffineMap scatterMap = op.getScatterMap();

    const Value memref = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, memrefType, rewriter.getRemappedValue(buffer));

    const size_t tensorSize = reduceMul(tensorType.getShape());
    const size_t chunkCount = reduceMul(bufferType.getWorkgroupShape());
    const size_t chunksPerTasklet = bufferType.getWorkgroupShape().back();
    const size_t chunkSize = tensorSize / chunkCount;
    const AffineMap mramMap = getElementToMRAMOffsetAffineMap(
        op.getContext(), chunksPerTasklet, chunkSize);

    createNestedAffineForLoops(
        rewriter, op.getLoc(), tensorType.getShape(),
        SmallVector<int64_t>(tensorType.getRank(), 1), ValueRange{},
        [&](OpBuilder &builder, Location loc, ValueRange tensorIndices,
            ValueRange) -> SmallVector<Value> {
          const Value element =
              builder.create<tensor::ExtractOp>(loc, tensor, tensorIndices);

          SmallVector<Value> bufferIndices;
          for (size_t i = 0; i < scatterMap.getNumResults(); i++) {
            bufferIndices.push_back(builder.create<affine::AffineApplyOp>(
                loc,
                AffineMap::get(tensorIndices.size(), 0,
                               scatterMap.getResult(i)),
                tensorIndices));
          }

          const Value mramOffset = builder.create<affine::AffineApplyOp>(
              loc, mramMap, ValueRange{bufferIndices}.take_back(3));
          bufferIndices.pop_back_n(3);
          bufferIndices.push_back(mramOffset);

          builder.create<affine::AffineStoreOp>(loc, element, memref,
                                                bufferIndices);

          return {};
        });

    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
    return success();
  }
};

struct ConvertCnmGatherToUPMEM : public OpConversionPattern<cnm::GatherOp> {
  using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::GatherOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Value buffer = op.getOperand(0);
    const RankedTensorType tensorType =
        op.getResultTypes()[0].cast<RankedTensorType>();
    const BufferType bufferType = buffer.getType().cast<BufferType>();
    const MemRefType memrefType = convertCnmBufferToMemRefType(bufferType);
    const AffineMap gatherMap = op.getGatherMap();

    const Value memref = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, memrefType, rewriter.getRemappedValue(buffer));

    const size_t tensorSize = reduceMul(tensorType.getShape());
    const size_t chunkCount = reduceMul(bufferType.getWorkgroupShape());
    const size_t chunksPerTasklet = bufferType.getWorkgroupShape().back();
    const size_t chunkSize = tensorSize / chunkCount;
    const AffineMap mramMap = getMRAMOffsetToElementAffineMap(
        op.getContext(), chunksPerTasklet, chunkSize);

    const Value resultInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType());

    SmallVector<Value> results = createNestedAffineForLoops(
        rewriter, op.getLoc(), memrefType.getShape(),
        SmallVector<int64_t>(memrefType.getRank(), 1), ValueRange{resultInit},
        [&](OpBuilder &builder, Location loc, ValueRange memrefIndices,
            ValueRange iterArgs) -> SmallVector<Value> {
          const Value mramOffset = memrefIndices.back();
          SmallVector<Value> bufferIndices{memrefIndices.drop_back()};
          for (AffineExpr e : mramMap.getResults()) {
            bufferIndices.push_back(builder.create<affine::AffineApplyOp>(
                loc, AffineMap::get(1, 0, e), mramOffset));
          }

          SmallVector<Value> tensorIndices;
          for (size_t i = 0; i < gatherMap.getNumResults(); i++) {
            tensorIndices.push_back(builder.create<affine::AffineApplyOp>(
                loc,
                AffineMap::get(bufferIndices.size(), 0, gatherMap.getResult(i)),
                bufferIndices));
          }

          const Value element =
              builder.create<affine::AffineLoadOp>(loc, memref, memrefIndices);
          return {builder.create<tensor::InsertOp>(loc, element, iterArgs[0],
                                                   tensorIndices)};
        });

    results.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ConvertCnmLaunchToUPMEM : public OpConversionPattern<cnm::LaunchOp> {
  using OpConversionPattern<cnm::LaunchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::LaunchOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Value wg = rewriter.getRemappedValue(op.getWg());
    const ArrayRef<int64_t> wgShape = op.getWg().getType().getShape();

    const Value rankCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[0]);
    const Value dpuCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[1]);
    const Value taskletCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[2]);
    const Value chunkCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[3]);

    // scatter results from host memory to dpu memory
    Value addr = rewriter.create<upmem::BaseDPUMemOffsetOp>(op.getLoc());
    for (Value buffer : op.getInBuffers()) {
      const MemRefType memrefType =
          convertCnmBufferToMemRefType(buffer.getType().cast<BufferType>());
      const Value memref = createOrFoldUnrealizedConversionCast(
          op.getLoc(), rewriter, memrefType, rewriter.getRemappedValue(buffer));
      addr = rewriter.create<upmem::ScatterOp>(op.getLoc(), memref, addr, wg);
    }

    // build launch op body
    upmem::LaunchOp launchOp = rewriter.create<upmem::LaunchOp>(
        op.getLoc(), wg, rankCount, dpuCount, taskletCount);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    // calculate address of all buffer slices for the current tasklet & allocate
    // wram to store the chunk data
    struct BufferSliceInfo {
      Value chunkSize;
      Value mramAddr;   // address in mram for current tasklet
      Value wramMemref; // memref allocated on wram to store the current chunk
    };

    Value dpuHeapAddr = rewriter.create<upmem::BaseMRAMAddrOp>(op.getLoc());
    llvm::DenseMap<Value, BufferSliceInfo> bufferSlices;
    size_t i = 0;
    for (Value buffer : op.getParams()) {
      const BufferType bufferType = buffer.getType().cast<BufferType>();
      const size_t totalChunks = reduceMul(bufferType.getWorkgroupShape());
      const size_t totalSize = totalChunks * reduceMul(bufferType.getShape());
      const size_t chunkSize = totalSize / totalChunks;
      const size_t memoryPerTasklet = wgShape[3] * chunkSize;
      const size_t memoryPerDPU = wgShape[2] * memoryPerTasklet;

      const MemRefType sliceType =
          MemRefType::get(bufferType.getShape(), bufferType.getElementType());

      const BufferSliceInfo slice{
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), chunkSize),
          dpuHeapAddr,
          rewriter.create<upmem::PrivateWRAMAllocOp>(op.getLoc(), sliceType),
      };

      dpuHeapAddr = rewriter.create<arith::AddIOp>(
          op.getLoc(), dpuHeapAddr,
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), memoryPerDPU));

      bufferSlices[buffer] = slice;
      op.getBody().getArgument(i++).replaceAllUsesWith(slice.wramMemref);
    }

    // loop over chunks & execute launch body
    affine::AffineForOp loop =
        rewriter.create<affine::AffineForOp>(op.getLoc(), 0, wgShape[3], 1);

    rewriter.setInsertionPointToStart(&loop.getRegion().front());
    const Value currentChunk = loop.getRegion().front().getArguments().front();
    // copy data from the input buffers to wram
    for (Value buffer : op.getInBuffers()) {
      const BufferSliceInfo slice = bufferSlices.at(buffer);
      Value offset = rewriter.create<arith::MulIOp>(op.getLoc(), currentChunk,
                                                    slice.chunkSize);
      Value mramAddr =
          rewriter.create<arith::AddIOp>(op.getLoc(), slice.mramAddr, offset);
      rewriter.create<upmem::MemcpyOp>(
          op.getLoc(), upmem::MemcpyDirOp::MRAMToWRAM, slice.wramMemref,
          slice.chunkSize, mramAddr);
    }

    // insert original launch body into loop before the AffineYieldOp implicitly
    // created by the AffineForOp
    loop.getRegion().front().getOperations().splice(
        (--loop.getRegion().front().end()),
        op.getBody().front().getOperations());
    rewriter.setInsertionPoint(&loop.getRegion().front().back());

    // copy data from wram to output buffers
    for (Value buffer : op.getOutBuffers()) {
      const BufferSliceInfo slice = bufferSlices.at(buffer);
      Value offset = rewriter.create<arith::MulIOp>(op.getLoc(), currentChunk,
                                                    slice.chunkSize);
      Value mramAddr =
          rewriter.create<arith::AddIOp>(op.getLoc(), slice.mramAddr, offset);
      rewriter.create<upmem::MemcpyOp>(
          op.getLoc(), upmem::MemcpyDirOp::WRAMToMRAM, slice.wramMemref,
          slice.chunkSize, mramAddr);
    }

    rewriter.setInsertionPointToEnd(&launchOp.getBody().front());
    rewriter.create<upmem::TerminatorOp>(op.getLoc());

    // gather results from dpu memory to host memory
    rewriter.setInsertionPointAfter(launchOp);
    for (Value buffer : op.getOutBuffers()) {
      const MemRefType memrefType =
          convertCnmBufferToMemRefType(buffer.getType().cast<BufferType>());
      const Value memref = createOrFoldUnrealizedConversionCast(
          op.getLoc(), rewriter, memrefType, rewriter.getRemappedValue(buffer));
      addr = rewriter.create<upmem::GatherOp>(op.getLoc(), memref, addr, wg);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCnmTerminatorToUPMEM
    : public OpConversionPattern<cnm::TerminatorOp> {
  using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op); // gets generated by ConvertCnmLaunchToUPMEM
    return success();
  }
};

} // namespace cnmtoupmem

void populateCnmToUPMEMFinalTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](cnm::WorkgroupType wgType) -> std::optional<Type> {
        return upmem::DeviceHierarchyType::get(wgType.getContext(),
                                               wgType.getShape().take_front(3));
      });

  typeConverter.addConversion(
      [&](cnm::BufferType bufferType) -> std::optional<Type> {
        return cnmtoupmem::convertCnmBufferToMemRefType(bufferType);
      });
}

void populateCnmToUPMEMConversionPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.add<cnmtoupmem::ConvertCnmWorkgroupToUPMEM,
               cnmtoupmem::ConvertCnmAllocToUPMEM, ConvertCnmSetZeroToAffine,
               cnmtoupmem::ConvertCnmScatterToUPMEM,
               cnmtoupmem::ConvertCnmGatherToUPMEM,
               cnmtoupmem::ConvertCnmLaunchToUPMEM,
               cnmtoupmem::ConvertCnmTerminatorToUPMEM>(
      &typeConverter.getContext());
}

struct ConvertCnmToUPMEMPass
    : public ::impl::ConvertCnmToUPMEMPassBase<ConvertCnmToUPMEMPass> {
  void runOnOperation() final {
    LLVMTypeConverter converter(&getContext());
    populateCnmToUPMEMFinalTypeConversions(converter);
    const auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateCnmToUPMEMConversionPatterns(converter, patterns);
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

std::unique_ptr<Pass> createConvertCnmToUPMEMPass() {
  return std::make_unique<ConvertCnmToUPMEMPass>();
}

} // namespace mlir::cnm
