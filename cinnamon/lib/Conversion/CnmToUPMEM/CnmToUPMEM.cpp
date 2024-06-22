#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <cstdint>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/DialectConversion.h>

#define GEN_PASS_DEF_CONVERTCNMTOUPMEMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
namespace {

template <typename T> T reduceMul(ArrayRef<T> arr) {
  T result{1};
  for (const T &elem : arr) {
    result *= elem;
  }
  return result;
}

MemRefType convertTensorToMemref(ShapedType ty) {
  if (ty.isa<MemRefType>())
    return ty.cast<MemRefType>();

  return MemRefType::get(ty.getShape(), ty.getElementType());
}

static const StringRef BUFFER_OFFSET_ATTR = "upmem.bufferOffset";
struct ConvertCnmWorkgroupToUPMEM
    : public OpConversionPattern<cnm::WorkgroupOp> {
  using OpConversionPattern<cnm::WorkgroupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::WorkgroupOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getType().getShape().size() != 3)
      return op->emitOpError(
          "cannot be converted to UPMEM dialect. "
          "UPMEM translation requires workgroup with 3 dimensions.");
    rewriter.replaceOpWithNewOp<upmem::AllocDPUsOp>(
        op, getTypeConverter()->convertType(op.getType()));

    int64_t dpuMemOffset = 0;
    for (auto use : op.getResult().getUsers()) {
      if (cnm::AllocOp alloc = llvm::dyn_cast_or_null<cnm::AllocOp>(use)) {
        alloc->setAttr(BUFFER_OFFSET_ATTR,
                       rewriter.getI64IntegerAttr(dpuMemOffset));
        dpuMemOffset += alloc.getResult().getType().getSizeInBytes();

        // buffer offsets must be 8 byte aligned
        if (dpuMemOffset % 8 != 0) {
          dpuMemOffset += 8 - (dpuMemOffset % 8);
        }
      }
    }

    return success();
  }
};

struct ConvertCnmFreeWorkgroup
    : public OpConversionPattern<cnm::FreeWorkgroupOp> {
  using OpConversionPattern<cnm::FreeWorkgroupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::FreeWorkgroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<upmem::FreeDPUsOp>(op, adaptor.getWorkgroup());
    return success();
  }
};

struct ConvertCnmScatterToUPMEM : public OpConversionPattern<cnm::ScatterOp> {
  using OpConversionPattern<cnm::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const Value tensor = adaptor.getInput();
    const ShapedType inputTy = op.getInput().getType();

    const Value inputAsMemref = createOrFoldUnrealizedConversionCast(
        op.getLoc(), rewriter, convertTensorToMemref(inputTy), tensor);

    const int64_t transferCount = op.getTransferCountInItems();
    const int64_t dpuMemOffset =
        llvm::cast<cnm::AllocOp>(op.getBuffer().getDefiningOp())
            ->getAttrOfType<IntegerAttr>(BUFFER_OFFSET_ATTR)
            .getInt();

    rewriter.create<upmem::ScatterOp>(op->getLoc(), inputAsMemref, dpuMemOffset,
                                      transferCount, op.getScatterMap(),
                                      adaptor.getWg());

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCnmGatherToUPMEM : public OpConversionPattern<cnm::GatherOp> {
  using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value outputBuf = adaptor.getOutputBuf();
    bool isBufferized = op.getOutputBuf().getType().isa<BaseMemRefType>();
    if (!isBufferized) {
      outputBuf = rewriter.create<memref::AllocOp>(
          op->getLoc(), convertTensorToMemref(op.getOutputBuf().getType()));
    }

    const int64_t transferCount = op.getTransferCountInItems();
    const int64_t dpuMemOffset =
        llvm::cast<cnm::AllocOp>(op.getBuffer().getDefiningOp())
            ->getAttrOfType<IntegerAttr>(BUFFER_OFFSET_ATTR)
            .getInt();

    rewriter.create<upmem::GatherOp>(op->getLoc(), outputBuf, dpuMemOffset,
                                     transferCount, op.getGatherMap(),
                                     adaptor.getWg());

    if (!isBufferized) {
      Value outputAsTensor = createOrFoldUnrealizedConversionCast(
          op->getLoc(), rewriter, op.getOutput().getType(), outputBuf);

      rewriter.replaceAllUsesWith(op.getOutput(), outputAsTensor);
    }
    rewriter.eraseOp(op);
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

    const size_t availableWRAM = 32 * 1024;
    size_t requiredWRAM = 0;
    for (Value buffer : op.getParams()) {
      const BufferType bufferType = buffer.getType().cast<BufferType>();
      const size_t elementSize =
          bufferType.getElementType().getIntOrFloatBitWidth() / 8;
      requiredWRAM += reduceMul(bufferType.getShape()) * elementSize;
    }

    if (requiredWRAM > availableWRAM) {
      emitError(op.getLoc(), "required wram (" + std::to_string(requiredWRAM) +
                                 " bytes) exceeds available wram (" +
                                 std::to_string(availableWRAM) + " bytes)");
      return failure();
    }

    const Value rankCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[0]);
    const Value dpuCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[1]);
    const Value taskletCount =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), wgShape[2]);
    const size_t chunksPerTasklet = wgShape.size() == 4 ? wgShape.back() : 1;

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
      const size_t chunkSize = reduceMul(bufferType.getShape());
      const size_t memoryPerTasklet = chunksPerTasklet * chunkSize;
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
    if (chunksPerTasklet == 1) {
      // copy data from the input buffers to wram
      for (Value buffer : op.getInBuffers()) {
        const BufferSliceInfo slice = bufferSlices.at(buffer);
        rewriter.create<upmem::MemcpyOp>(
            op.getLoc(), upmem::MemcpyDirOp::MRAMToWRAM, slice.wramMemref,
            slice.chunkSize, slice.mramAddr);
      }

      // insert original launch body into loop before the AffineYieldOp
      // implicitly created by the AffineForOp
      launchOp.getBody().front().getOperations().splice(
          launchOp.getBody().front().end(),
          op.getBody().front().getOperations());

      // copy data from wram to output buffers
      for (Value buffer : op.getOutBuffers()) {
        const BufferSliceInfo slice = bufferSlices.at(buffer);
        rewriter.create<upmem::MemcpyOp>(
            op.getLoc(), upmem::MemcpyDirOp::WRAMToMRAM, slice.wramMemref,
            slice.chunkSize, slice.mramAddr);
      }
    } else {
      affine::AffineForOp loop = rewriter.create<affine::AffineForOp>(
          op.getLoc(), 0, chunksPerTasklet, 1);

      rewriter.setInsertionPointToStart(&loop.getRegion().front());
      const Value currentChunk =
          loop.getRegion().front().getArguments().front();

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

      // insert original launch body into loop before the AffineYieldOp
      // implicitly created by the AffineForOp
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
    }

    rewriter.create<upmem::TerminatorOp>(op.getLoc());

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

} // namespace

void populateCnmToUPMEMFinalTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](cnm::WorkgroupType wgType) -> Type {
    return upmem::DeviceHierarchyType::get(wgType.getContext(),
                                           wgType.getShape());
  });

  typeConverter.addConversion([](ShapedType st) -> Type { return st; });
}

void populateCnmToUPMEMConversionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.add<ConvertCnmWorkgroupToUPMEM, ConvertCnmSetZeroToAffine,
               ConvertCnmScatterToUPMEM, ConvertCnmGatherToUPMEM,
               ConvertCnmLaunchToUPMEM, ConvertCnmTerminatorToUPMEM,
               ConvertCnmFreeWorkgroup>(
      typeConverter, patterns.getContext());
}

struct ConvertCnmToUPMEMPass
    : public ::impl::ConvertCnmToUPMEMPassBase<ConvertCnmToUPMEMPass> {
  void runOnOperation() final {
    TypeConverter converter;
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
    populateFinalBufferizationPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<cnm::CnmDialect>();
    // alloc ops are deleted in second pass
    target.addLegalOp<cnm::AllocOp>();
    // target.addIllegalDialect<bufferization::BufferizationDialect>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }

    getOperation()->walk([](cnm::AllocOp op) { op->erase(); });
  }
};

std::unique_ptr<Pass> createConvertCnmToUPMEMPass() {
  return std::make_unique<ConvertCnmToUPMEMPass>();
}

} // namespace mlir::cnm
