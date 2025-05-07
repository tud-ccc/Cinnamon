#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
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
#include <mlir/Support/LLVM.h>
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
  if (isa<MemRefType>(ty))
    return cast<MemRefType>(ty);

  return MemRefType::get(ty.getShape(), ty.getElementType());
}

inline static constexpr size_t alignTo(size_t v, size_t alignment) {
  return v % alignment == 0 ? v : v + alignment - v % alignment;
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

    SmallVector<cnm::AllocOp> allocs;
    for (auto use : op.getResult().getUsers()) {
      if (cnm::AllocOp alloc = llvm::dyn_cast_or_null<cnm::AllocOp>(use)) {
        allocs.push_back(alloc);
      }
    }

    int64_t dpuMemOffset = 0;
    const size_t numTasklets = op.getType().getShape()[2];
    // getUsers returns users in reverse order so we have to reverse again to
    // calculate the offset in the correct order
    for (auto alloc : llvm::reverse(allocs)) {
      alloc->setAttr(BUFFER_OFFSET_ATTR,
                     rewriter.getI64IntegerAttr(dpuMemOffset));
      const size_t memoryPerTasklet =
          alloc.getResult().getType().getSizeInBytes();
      dpuMemOffset += memoryPerTasklet * numTasklets;

      // buffer offsets must be 8 byte aligned
      dpuMemOffset = alignTo(dpuMemOffset, 8);
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

    const size_t numTasklets = op.getWg().getType().getShape()[2];
    const int64_t transferCount = op.getTransferCountInItems() * numTasklets;
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
    bool isBufferized = isa<BaseMemRefType>(op.getOutputBuf().getType());
    if (!isBufferized) {
      outputBuf = rewriter.create<memref::AllocOp>(
          op->getLoc(), convertTensorToMemref(op.getOutputBuf().getType()));
    }

    const size_t numTasklets = op.getWg().getType().getShape()[2];
    const int64_t transferCount = op.getTransferCountInItems() * numTasklets;
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
      const BufferType bufferType = cast<BufferType>(buffer.getType());
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
      Value mramCopySize;
      Value mramAddr;   // address in mram for current tasklet
      Value wramMemref; // memref allocated on wram to store the current chunk
    };

    Value dpuHeapAddr = rewriter.create<upmem::BaseMRAMAddrOp>(op.getLoc());
    Value taskletId = rewriter.create<upmem::TaskletIDOp>(op.getLoc());
    llvm::DenseMap<Value, BufferSliceInfo> bufferSlices;
    size_t i = 0;
    for (Value buffer : op.getParams()) {
      if (!dyn_cast<BufferType>(buffer.getType())) {
        continue;
      }

      const int64_t dpuMemOffset =
          llvm::cast<cnm::AllocOp>(buffer.getDefiningOp())
              ->getAttrOfType<IntegerAttr>(BUFFER_OFFSET_ATTR)
              .getInt();

      const BufferType bufferType = llvm::cast<BufferType>(buffer.getType());
      const size_t elementTypeSize = bufferType.getElementTypeBitWidth() / 8;
      const size_t chunkSize = reduceMul(bufferType.getShape());
      size_t mramCopySize = chunkSize;

      if (!bufferType.getShape().empty() &&
          chunkSize * elementTypeSize % 8 != 0) {
        emitError(op.getLoc(), "chunksize (" +
                                   std::to_string(chunkSize * elementTypeSize) +
                                   ") isn't 8-byte aligned");
        return failure();
      }

      size_t memoryPerTasklet = chunksPerTasklet * chunkSize *
                                bufferType.getElementTypeBitWidth() / 8;

      const MemRefType sliceType =
          MemRefType::get(bufferType.getShape(), bufferType.getElementType());

      Value mramAddr;
      bool isInput = false;
      for (Value other : op.getInputs()) {
        if (buffer == other) {
          isInput = true;
          break;
        }
      }

      // When loading a value from a scalar buffer on the dpu we ignore the
      // tasklet id to avoid alignment issues: when copying a single value for
      // tasklet > 0 the mram address may not be 8-byte aligned. The values in
      // the mram are the same for all tasklets anyway.
      if (isInput && bufferType.getShape().empty()) {
        mramAddr = dpuHeapAddr;
      } else {
        if (!isInput && memoryPerTasklet < 8) {
          memoryPerTasklet = 8;
          mramCopySize = 8 / elementTypeSize;
        }

        mramAddr = rewriter.create<arith::AddIOp>(
            op.getLoc(), dpuHeapAddr,
            rewriter.create<arith::MulIOp>(
                op.getLoc(), taskletId,
                rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                                        memoryPerTasklet)));
      }

      // To get to the minimum mram_read() size (8 bytes) we can copy
      // additional bytes of the input as we don't care if they are garbage
      // because we only use the first element. This doesn't apply to single
      // output values as they would interfere with / overwrite the result
      // values of other tasklets.
      if (isInput && chunkSize * elementTypeSize < 8) {
        mramCopySize = 8 / elementTypeSize;
      }

      const BufferSliceInfo slice{
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), chunkSize),
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), mramCopySize),
          mramAddr,
          rewriter.create<upmem::PrivateWRAMAllocOp>(op.getLoc(), sliceType),
      };

      const size_t memoryPerDPU = alignTo(wgShape[2] * memoryPerTasklet, 8);
      dpuHeapAddr = rewriter.create<arith::AddIOp>(
          op.getLoc(), dpuHeapAddr,
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), memoryPerDPU));

      bufferSlices[buffer] = slice;
      op.getBody().getArgument(i++).replaceAllUsesWith(slice.wramMemref);
    }

    // loop over chunks & execute launch body
    if (chunksPerTasklet == 1) {
      // copy data from the input buffers to wram
      for (Value buffer : op.getInputs()) {
        if (!dyn_cast<BufferType>(buffer.getType())) {
          continue;
        }

        const BufferSliceInfo slice = bufferSlices.at(buffer);
        rewriter.create<upmem::MemcpyOp>(
            op.getLoc(), upmem::MemcpyDirOp::MRAMToWRAM, slice.wramMemref,
            slice.mramCopySize, slice.mramAddr);
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
            slice.mramCopySize, slice.mramAddr);
      }
    } else {
      affine::AffineForOp loop = rewriter.create<affine::AffineForOp>(
          op.getLoc(), 0, chunksPerTasklet, 1);

      rewriter.setInsertionPointToStart(&loop.getRegion().front());
      const Value currentChunk =
          loop.getRegion().front().getArguments().front();

      // copy data from the input buffers to wram
      for (Value buffer : op.getInputs()) {
        if (!dyn_cast<BufferType>(buffer.getType())) {
          continue;
        }

        const BufferSliceInfo slice = bufferSlices.at(buffer);
        Value offset = rewriter.create<arith::MulIOp>(op.getLoc(), currentChunk,
                                                      slice.chunkSize);
        Value mramAddr =
            rewriter.create<arith::AddIOp>(op.getLoc(), slice.mramAddr, offset);
        rewriter.create<upmem::MemcpyOp>(
            op.getLoc(), upmem::MemcpyDirOp::MRAMToWRAM, slice.wramMemref,
            slice.mramCopySize, mramAddr);
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
            slice.mramCopySize, mramAddr);
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
               ConvertCnmFreeWorkgroup>(typeConverter, patterns.getContext());
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
