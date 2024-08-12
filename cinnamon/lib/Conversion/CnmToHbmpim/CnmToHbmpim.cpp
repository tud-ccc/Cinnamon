#include "cinm-mlir/Conversion/CnmToHbmpim/CnmToHbmpim.h"
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

#define GEN_PASS_DEF_CONVERTCNMTOHBMPIMPASS
#include "cinm-mlir/Conversion/CnmPasses.h.inc"

namespace mlir::cnm {
namespace {

// template <typename T> T reduceMul(ArrayRef<T> arr) {
//   T result{1};
//   for (const T &elem : arr) {
//     result *= elem;
//   }
//   return result;
// }

MemRefType convertTensorToMemref(ShapedType ty) {
  if (ty.isa<MemRefType>())
    return ty.cast<MemRefType>();

  return MemRefType::get(ty.getShape(), ty.getElementType());
}

// static const StringRef BUFFER_OFFSET_ATTR = "upmem.bufferOffset";
struct ConvertCnmWorkgroupToHbmpim
    : public OpConversionPattern<cnm::WorkgroupOp> {
  using OpConversionPattern<cnm::WorkgroupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::WorkgroupOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getType().getShape().size() != 4)
      return op->emitOpError(
          "cannot be converted to Hbmpim dialect. "
          "Hbmpim translation requires workgroup with 4 dimensions.");
    rewriter.replaceOpWithNewOp<hbmpim::SetDeviceConfigOp>(
        op, getTypeConverter()->convertType(op.getType()));
    return success();
  }
};

struct ConvertCnmFreeWorkgroup
    : public OpConversionPattern<cnm::FreeWorkgroupOp> {
  using OpConversionPattern<cnm::FreeWorkgroupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::FreeWorkgroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// struct ConvertCnmScatterToHbmpim : public OpConversionPattern<cnm::ScatterOp> {
//   using OpConversionPattern<cnm::ScatterOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(cnm::ScatterOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     const Value tensor = adaptor.getInput();
//     const ShapedType inputTy = op.getInput().getType();

//     const Value inputAsMemref = createOrFoldUnrealizedConversionCast(
//         op.getLoc(), rewriter, convertTensorToMemref(inputTy), tensor);

//     const int64_t transferCount = op.getTransferCountInItems();
//     const int64_t dpuMemOffset =
//         llvm::cast<cnm::AllocOp>(op.getBuffer().getDefiningOp())
//             ->getAttrOfType<IntegerAttr>(BUFFER_OFFSET_ATTR)
//             .getInt();

//     rewriter.create<upmem::ScatterOp>(op->getLoc(), inputAsMemref, dpuMemOffset,
//                                       transferCount, op.getScatterMap(),
//                                       adaptor.getWg());

//     rewriter.eraseOp(op);
//     return success();
//   }
// };

// struct ConvertCnmGatherToUPMEM : public OpConversionPattern<cnm::GatherOp> {
//   using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(cnm::GatherOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Value outputBuf = adaptor.getOutputBuf();
//     bool isBufferized = op.getOutputBuf().getType().isa<BaseMemRefType>();
//     if (!isBufferized) {
//       outputBuf = rewriter.create<memref::AllocOp>(
//           op->getLoc(), convertTensorToMemref(op.getOutputBuf().getType()));
//     }

//     const int64_t transferCount = op.getTransferCountInItems();
//     const int64_t dpuMemOffset =
//         llvm::cast<cnm::AllocOp>(op.getBuffer().getDefiningOp())
//             ->getAttrOfType<IntegerAttr>(BUFFER_OFFSET_ATTR)
//             .getInt();

//     rewriter.create<upmem::GatherOp>(op->getLoc(), outputBuf, dpuMemOffset,
//                                      transferCount, op.getGatherMap(),
//                                      adaptor.getWg());

//     if (!isBufferized) {
//       Value outputAsTensor = createOrFoldUnrealizedConversionCast(
//           op->getLoc(), rewriter, op.getOutput().getType(), outputBuf);

//       rewriter.replaceAllUsesWith(op.getOutput(), outputAsTensor);
//     }
//     rewriter.eraseOp(op);
//     return success();
//   }
// };

struct ConvertCnmLaunchToHbmpim : public OpConversionPattern<cnm::LaunchOp> {
  using OpConversionPattern<cnm::LaunchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::LaunchOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto wg = op.getWg().getType();
    const ArrayRef<int64_t> wgShape = wg.getShape();
    // llvm::dbgs() << "operation count " << op.getBody().front().getOperations().size() << "\n";
    
    for(auto &operation: op.getBody().front().getOperations()){
      if(dyn_cast<linalg::AddOp>(operation)){
        // llvm::dbgs() << "Add operation found\n";
        // dim calculation 
        int64_t totalPE = wg.getNumElements();
        int64_t dim = totalPE/wgShape[0];
        auto cst0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto dimVal = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dim);
        auto in1Val = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto resVal = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 128);
        auto in2Val = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 256);
        auto inBufs = op.getInBuffers();
        SmallVector
        auto inBuf1Uses = inBufs[0].getUses(); 
        for (auto &use : inBuf1Uses){
          if (dyn_cast<cnm::ScatterOp>(use.getOwner())){
            cnm::ScatterOp scatterOp = dyn_cast<cnm::ScatterOp>(use.getOwner());
            Value tensor = scatterOp.getInput();
            ShapedType inputTy = scatterOp.getInput().getType();
            Value inputAsMemref = createOrFoldUnrealizedConversionCast(
                op.getLoc(), rewriter, convertTensorToMemref(inputTy), tensor);
            rewriter.create<hbmpim::PreloadNoReplacementOp>(op.getLoc(), inputAsMemref, in1Val, cst0);
            llvm::dbgs() << "Found one of the scatter Op\n";
          }
        }
        rewriter.create<hbmpim::ExecuteElementwiseOp>(op.getLoc(), dimVal, hbmpim::PimBankType::ALL_BANK, 
                hbmpim::PimKernelType::ADD, in1Val, resVal, in2Val);
      } else if (!dyn_cast<cnm::TerminatorOp>(operation)){
        return op->emitOpError("operation not supported yet") ;
      }     
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCnmTerminatorToHbmpim
    : public OpConversionPattern<cnm::TerminatorOp> {
  using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op); 
    return success();
  }
};

} // namespace

void populateCnmToHbmpimFinalTypeConversions(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](cnm::WorkgroupType wgType) -> Type {
    return hbmpim::DeviceConfigurationType::get(wgType.getContext(),
                                           wgType.getShape());
  });
  typeConverter.addConversion([](ShapedType st) -> Type { return st; });
}

void populateCnmToHbmpimConversionPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {

  patterns.add<
  ConvertCnmWorkgroupToHbmpim
    // ,ConvertCnmScatterToUPMEM
    // ,ConvertCnmGatherToUPMEM
    ,ConvertCnmLaunchToHbmpim
    // ,ConvertCnmTerminatorToUPMEM
    // ,ConvertCnmFreeWorkgroup
    >(typeConverter, patterns.getContext());
}

struct ConvertCnmToHbmpimPass
    : public ::impl::ConvertCnmToHbmpimPassBase<ConvertCnmToHbmpimPass> {
  void runOnOperation() final {
    TypeConverter converter;
    populateCnmToHbmpimFinalTypeConversions(converter);
    const auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateCnmToHbmpimConversionPatterns(converter, patterns);
    populateReconcileUnrealizedCastsPatterns(patterns);
    populateFinalBufferizationPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<cnm::CnmDialect>();
    // alloc ops are deleted in second pass
    target.addLegalOp<cnm::AllocOp>();
    target.addLegalOp<cnm::ScatterOp>();
    // target.addIllegalDialect<bufferization::BufferizationDialect>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
    getOperation()->walk([](cnm::ScatterOp op) { op->erase(); });
    getOperation()->walk([](cnm::AllocOp op) { op->erase(); });

  }
};

std::unique_ptr<Pass> createConvertCnmToHbmpimPass() {
  return std::make_unique<ConvertCnmToHbmpimPass>();
}

} // namespace mlir::cnm
