#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <cinm-mlir/Dialect/UPMEM/Transforms/Utils.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
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
    auto upmemWg = llvm::cast<upmem::DeviceHierarchyType>(
        getTypeConverter()->convertType(op.getType()));

    rewriter.replaceOpWithNewOp<upmem::AllocDPUsOp>(op, upmemWg,
                                                    SymbolRefAttr{});

    SmallVector<cnm::AllocOp> allocs;
    for (auto use : op.getResult().getUsers()) {
      if (cnm::AllocOp alloc = llvm::dyn_cast_or_null<cnm::AllocOp>(use)) {
        allocs.push_back(alloc);
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

static LogicalResult convertCnmGatherToUpmem(RewriterBase &rewriter,
                                             cnm::GatherOp op,
                                             upmem::AllocDPUsOp upmemWgAlloc,
                                             SymbolRefAttr refToBuffer) {

  rewriter.setInsertionPoint(op);
  Value outputBuf = op.getOutputBuf();
  bool isBufferized = isa<BaseMemRefType>(op.getOutputBuf().getType());
  if (!isBufferized) {
    outputBuf = rewriter.create<memref::AllocOp>(
        op->getLoc(), convertTensorToMemref(op.getOutputBuf().getType()));
  }

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const int64_t transferCount = op.getTransferCountInItems() * numTasklets;

  rewriter.create<upmem::GatherOp>(op->getLoc(), outputBuf, refToBuffer,
                                   transferCount, op.getGatherMap(),
                                   upmemWgAlloc.getResult());

  if (!isBufferized) {
    Value outputAsTensor = createOrFoldUnrealizedConversionCast(
        op->getLoc(), rewriter, op.getOutput().getType(), outputBuf);

    rewriter.replaceAllUsesWith(op.getOutput(), outputAsTensor);
  }
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult convertCnmScatterToUpmem(RewriterBase &rewriter,
                                              cnm::ScatterOp op,
                                              upmem::AllocDPUsOp upmemWgAlloc,
                                              SymbolRefAttr refToBuffer) {

  rewriter.setInsertionPoint(op);
  const Value tensor = op.getInput();
  const ShapedType inputTy = op.getInput().getType();

  const Value inputAsMemref = createOrFoldUnrealizedConversionCast(
      op.getLoc(), rewriter, convertTensorToMemref(inputTy), tensor);

  const size_t numTasklets = upmemWgAlloc.getType().getNumTaskletsPerDpu();
  const int64_t transferCount = op.getTransferCountInItems() * numTasklets;

  rewriter.create<upmem::ScatterOp>(op->getLoc(), inputAsMemref, refToBuffer,
                                    transferCount, op.getScatterMap(),
                                    upmemWgAlloc.getResult());

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult convertCnmLaunchToUpmem(cnm::LaunchOp launch,
                                             RewriterBase &rewriter,
                                             ModuleOp dpuKernelModule) {

  rewriter.clearInsertionPoint();

  auto wg = launch.getWg().getType().getShape();
  if (wg.size() != 3)
    return launch.emitOpError("Should have working group with 3 entries");

  const auto upmemTy =
      rewriter.getType<upmem::DeviceHierarchyType>(wg[0], wg[1], wg[2]);

  auto dpuProgram = rewriter.create<upmem::DpuProgramOp>(
      launch->getLoc(), "program", upmemTy.getNumTaskletsPerDpu());

  SymbolTable symTable(dpuKernelModule);
  symTable.insert(dpuProgram);

  auto wgAlloc = cast<cnm::WorkgroupOp>(launch.getWg().getDefiningOp());
  rewriter.setInsertionPoint(wgAlloc);
  auto upmemWgAlloc = rewriter.create<upmem::AllocDPUsOp>(
      wgAlloc->getLoc(), upmemTy, dpuProgram.getSymNameAttr());

  llvm::MapVector<Value, upmem::StaticAllocOp> buffersToMramBuf;

  // rewriter.setInsertionPointToStart(&dpuProgram.getBody().front());

  // create named static MRAM buffers for each cnm.alloc operation, put them in
  // the dpu program

  SymbolTable dpuProgramSymTable(dpuProgram);
  for (auto user : launch.getWg().getUsers()) {
    if (auto alloc = llvm::dyn_cast_or_null<cnm::AllocOp>(user)) {
      auto bufferType = alloc.getType();

      const MemRefType memrefTy =
          MemRefType::get(bufferType.getShape(), bufferType.getElementType(),
                          MemRefLayoutAttrInterface{}, bufferType.getLevel());

      auto mrambuf = rewriter.create<upmem::StaticAllocOp>(
          alloc->getLoc(), memrefTy, false, false, "buf");
      dpuProgramSymTable.insert(mrambuf); // does renaming
      buffersToMramBuf[alloc.getResult()] = mrambuf;
    }
  }

  // then, replace all scatter/gather with the upmem equivalents

  for (auto user : launch.getWg().getUsers()) {

    if (auto scatter = llvm::dyn_cast_or_null<cnm::ScatterOp>(user)) {
      auto alloc = buffersToMramBuf[scatter.getBuffer()];
      if (alloc) {
        auto ref = upmem::getSymbolPath(
            SymbolTable::getNearestSymbolTable(scatter), alloc);
        if (succeeded(ref)) {
          if (failed(convertCnmScatterToUpmem(rewriter, scatter, upmemWgAlloc,
                                              *ref)))
            return failure();
          continue;
        }
      }
    }
    if (auto gather = llvm::dyn_cast_or_null<cnm::GatherOp>(user)) {
      auto alloc = buffersToMramBuf[gather.getBuffer()];
      if (alloc) {
        auto ref = upmem::getSymbolPath(
            SymbolTable::getNearestSymbolTable(gather), alloc);
        if (succeeded(ref)) {
          if (failed(convertCnmGatherToUpmem(rewriter, gather, upmemWgAlloc,
                                             *ref)))
            return failure();
          continue;
        }
      }
    }
  }
  // At this point we have replaced (and deleted) the scatter and gather.
  // We still need to move the body of the launch into the new DPU program,
  // and delete all remaining CNM ops.

  // wait no the memrefs inside have the type of the wram buffers.
  // the mram buffers have tasklets * wram buffer size.
  IRMapping mapping;
  for (auto [cnmBuf, memref] : llvm::zip_equal(
           llvm::concat<Value>(launch.getInputs(), launch.getOutBuffers()),
           launch.getBody().getArguments())) {

    mapping.map(memref, buffersToMramBuf[cnmBuf].getResult());
  }

  return success();
}

struct ConvertCnmLaunchToUPMEM : public OpConversionPattern<cnm::LaunchOp> {
  ModuleOp kernelModuleOp;
  ConvertCnmLaunchToUPMEM(MLIRContext *ctx, ModuleOp kernelModuleOp)
      : mlir::OpConversionPattern<cnm::LaunchOp>(ctx),
        kernelModuleOp(kernelModuleOp) {}

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

    const size_t chunksPerTasklet = wgShape.size() == 4 ? wgShape.back() : 1;

    // build launch op body
    upmem::InlineDpuProgramOp launchOp =
        rewriter.create<upmem::InlineDpuProgramOp>(op.getLoc(), wg);
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

    rewriter.create<upmem::ReturnOp>(op.getLoc());

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
    return upmem::DeviceHierarchyType::get(
        wgType.getContext(), wgType.getShape()[0], wgType.getShape()[1],
        wgType.getShape()[2]);
  });

  typeConverter.addConversion([](ShapedType st) -> Type { return st; });
}

void populateCnmToUPMEMConversionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ModuleOp dpuKernelModule) {
  patterns.add<ConvertCnmSetZeroToAffine, ConvertCnmTerminatorToUPMEM>(
      typeConverter, patterns.getContext());

  patterns.add<ConvertCnmWorkgroupToUPMEM, ConvertCnmScatterToUPMEM,
               ConvertCnmGatherToUPMEM, ConvertCnmLaunchToUPMEM,
               ConvertCnmFreeWorkgroup>(typeConverter, patterns.getContext(),
                                        dpuKernelModule);
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

    auto rootModule = getOperation();
    auto sym = SymbolTable::lookupSymbolIn(rootModule, "dpu_kernels");
    ModuleOp dpuKernelModule = llvm::dyn_cast_or_null<ModuleOp>(sym);
    if (!dpuKernelModule && sym) {
      mlir::emitError(sym->getLoc(), "Should be a module");
      signalPassFailure();
      return;
    }
    if (!dpuKernelModule) {
      OpBuilder builder(&getContext());
      builder.setInsertionPointToEnd(&rootModule.getBodyRegion().front());
      dpuKernelModule =
          builder.create<ModuleOp>(rootModule->getLoc(), "dpu_kernels");
    }

    SmallVector<LaunchOp> launchOps;
    rootModule->walk(
        [&](cnm::LaunchOp launch) { launchOps.push_back(launch); });

    for (auto launch : launchOps) {
    }

    RewritePatternSet patterns(&getContext());
    populateCnmToUPMEMConversionPatterns(converter, patterns, dpuKernelModule);
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
