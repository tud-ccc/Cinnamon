#include "cinm-mlir/Conversion/CnmToUPMEM/CnmToUPMEM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

#define GEN_PASS_DEF_CONVERTCNMTOUPMEMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
namespace cnmtoupmem {

MemRefType convertCnmBufferToMemRefType(cnm::BufferType bufferType) {
  assert(bufferType.getWorkgroupShape().size() == 4);
  SmallVector<int64_t, 3> shape{bufferType.getWorkgroupShape().take_front(2)};
  int64_t perDPUMemory = 1;
  for (int64_t dim : bufferType.getWorkgroupShape().take_back(2)) {
    perDPUMemory *= dim;
  }
  for (int64_t dim : bufferType.getShape()) {
    perDPUMemory *= dim;
  }
  shape.push_back(perDPUMemory);
  return MemRefType::get(shape, bufferType.getElementType());
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

struct ConvertCnmTerminatorToUPMEM
    : public OpConversionPattern<cnm::TerminatorOp> {
  using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<upmem::TerminatorOp>(op);
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
               // ConvertCnmScatterToUPMEM,
               // ConvertCnmGatherToUPMEM,
               // ConvertCnmLaunchToUPMEM,
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
    // target.addIllegalDialect<cnm::CnmDialect>();

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
