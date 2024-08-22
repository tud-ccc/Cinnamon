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


MemRefType convertTensorToMemref(ShapedType ty) {
  if (ty.isa<MemRefType>())
    return ty.cast<MemRefType>();

  return MemRefType::get(ty.getShape(), ty.getElementType());
}


Value trimStaticShape(OpBuilder &builder, Location loc, Value value) {
  ShapedType type = dyn_cast<ShapedType>(value.getType());
  if (type.isa<RankedTensorType>()) {
    SmallVector<int64_t> newShape;
    bool trim = true;
    for (auto dim : type.getShape())
      if (dim != 1 || !trim) 
        newShape.push_back(dim);
      else trim = false;

    auto reifiedShape = builder.create<arith::ConstantOp>(
        loc, RankedTensorType::get({static_cast<int64_t>(newShape.size())}, builder.getI64Type()),
        builder.getI64TensorAttr(newShape));

    auto newType = RankedTensorType::get(newShape, type.getElementType());
    // return builder.create<tensor::ReshapeOp>(RankedTensorType::get(newShape, type.getElementType()), value, reifiedShape);
    return builder.create<tensor::ReshapeOp>(loc, newType, value, reifiedShape);
  }
  // todo memref
  assert(false && "not handled for memrefs for now");
}

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


hbmpim::LaunchOp createVALaunch(ImplicitLocOpBuilder &builder, int64_t totalPE, Value remapped_wg, SmallVector<Value> inputs, Value output) {


  auto wgShape = dyn_cast<hbmpim::DeviceConfigurationType>(remapped_wg.getType()).getShape();
  int64_t dim = totalPE/wgShape[0];
  hbmpim::LaunchOp launchOp =
    builder.create<hbmpim::LaunchOp>(remapped_wg);
  // auto &launchBlock = launchOp.getBody().emplaceBlock();
  // for (auto input : launchOp.getParams()) {
  //   if (auto inputTy = input.getType().cast<MemRefType>()) {
  //     auto mappedTy =
  //       MemRefType::get(inputTy.getShape(), inputTy.getElementType());
  //     launchBlock.addArgument(mappedTy, input.getLoc());
  //   } else {
  //     launchBlock.addArgument(input.getType(), input.getLoc());
  //   }
  // } 
  // builder.setInsertionPointToStart(&launchBlock);
  // auto cst0 = builder.create<arith::ConstantIndexOp>(0);
  // auto dimVal = builder.create<arith::ConstantIndexOp>(dim);
  // auto in1Val = builder.create<arith::ConstantIndexOp>(0);
  // auto in2Val = builder.create<arith::ConstantIndexOp>(128);
  // auto resVal = builder.create<arith::ConstantIndexOp>(256);
  // builder.create<hbmpim::PreloadNoReplacementOp>(launchBlock.getArguments()[0], in1Val, cst0);
  // builder.create<hbmpim::PreloadNoReplacementOp>(launchBlock.getArguments()[1], in2Val, cst0);
  // builder.create<hbmpim::ExecuteElementwiseOp>(dimVal, hbmpim::PimBankType::ALL_BANK, 
  //         hbmpim::PimKernelType::ADD, in1Val, resVal, in2Val); 
  // builder.create<hbmpim::ReadDataOp>(launchBlock.getArguments()[2], resVal, cst0);
  builder.create<hbmpim::TerminatorOp>(); 
  return launchOp;
}




hbmpim::LaunchOp createGEMVLaunch(ImplicitLocOpBuilder &builder, Value remapped_wg, SmallVector<Value> inputs, Value output) {

  auto wgShape = dyn_cast<hbmpim::DeviceConfigurationType>(remapped_wg.getType()).getShape();
  int64_t numGrf = wgShape[3];
  int64_t totalPimBlocks = wgShape[0] * wgShape[1] / 2;

  builder.create<hbmpim::HostPreloadGemvOp>(hbmpim::DataDimType::weight_npbst_, inputs[0]);
  builder.create<hbmpim::HostPreloadGemvOp>(hbmpim::DataDimType::input_npbst_, inputs[1] );

  hbmpim::LaunchOp launchOp = builder.create<hbmpim::LaunchOp>(remapped_wg);
  auto &launchBlock = launchOp.getBody().emplaceBlock();
  auto currPoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&launchBlock);

  auto cst0 = builder.create<arith::ConstantIndexOp>(0);
  auto cst1 = builder.create<arith::ConstantIndexOp>(1);
  auto cst2 = builder.create<arith::ConstantIndexOp>(2);


  auto in1Shape = dyn_cast<ShapedType>(inputs[0].getType()).getShape();
  int64_t M = in1Shape[0];
  int64_t N = in1Shape[1];
  auto mValue = builder.create<arith::ConstantIndexOp>(M);
  auto nValue = builder.create<arith::ConstantIndexOp>(N);
  auto totalPimBlocksVal = builder.create<arith::ConstantIndexOp>(totalPimBlocks);
  auto numGrfVal = builder.create<arith::ConstantIndexOp>(numGrf);

  auto outputTiles = builder.create<arith::DivUIOp>(nValue, totalPimBlocksVal);
  auto numInputTiles = builder.create<arith::DivUIOp>(mValue, numGrfVal);
  auto temp1 = builder.create<arith::MulIOp>(outputTiles, numInputTiles);
  auto temp2 = builder.create<arith::MulIOp>(numGrfVal, numGrfVal);
  auto temp3 = builder.create<arith::MulIOp>(temp2, cst2);
  builder.create<hbmpim::PreloadGemvOp>(hbmpim::DataDimType::weight_npbst_, cst0, cst0);
  builder.create<hbmpim::ExecuteGemvOp>(cst1, mValue, nValue, 
          outputTiles, numInputTiles, false, wgShape[3], wgShape[2], wgShape[1], wgShape[0]);
  auto resultCol = builder.create<arith::DivUIOp>(temp1, temp3);
  builder.create<hbmpim::ReadResultOp>(hbmpim::DeviceBurstType::result, 
          hbmpim::PimBankType::ODD_BANK, nValue, cst0, cst0, resultCol);
  builder.create<hbmpim::TerminatorOp>(); 
  builder.restoreInsertionPoint(currPoint);
  builder.create<hbmpim::HostReadResultOp>(hbmpim::DeviceBurstType::result, output);

  return launchOp;
}

struct ConvertCnmLaunchToHbmpim : public OpConversionPattern<cnm::LaunchOp> {
  using OpConversionPattern<cnm::LaunchOp>::OpConversionPattern;
  
  LogicalResult
  matchAndRewrite(cnm::LaunchOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto wg = op.getWg();
    auto wgType = wg.getType();
    const ArrayRef<int64_t> wgShape = wgType.getShape();
    const Value remapped_wg = rewriter.getRemappedValue(wg);
    int64_t totalPE = wgType.getNumElements();
    int64_t totalPimBlocks = wgShape[0] * wgShape[1] / 2;

    
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;
    for (auto inBuf: op.getInBuffers()){
      for (auto &use : inBuf.getUses()){
        if (dyn_cast<cnm::ScatterOp>(use.getOwner())){
          cnm::ScatterOp scatterOp = dyn_cast<cnm::ScatterOp>(use.getOwner());
          Value tensor = scatterOp.getInput();
          Value trimmed_tensor = trimStaticShape(builder, op.getLoc(), tensor);
          ShapedType inputTy = dyn_cast<ShapedType>(trimmed_tensor.getType());
          Value inputAsMemref = createOrFoldUnrealizedConversionCast(
              op.getLoc(), rewriter, convertTensorToMemref(inputTy), trimmed_tensor);
          inputs.push_back(inputAsMemref);
        }
      }
    }
    for (auto outBuf: op.getOutBuffers()){
      for (auto &use : outBuf.getUses()){
        if (dyn_cast<cnm::GatherOp>(use.getOwner())){
          cnm::GatherOp gatherOp = dyn_cast<cnm::GatherOp>(use.getOwner());
          Value tensor = gatherOp.getOutputBuf();
          Value trimmed_tensor = trimStaticShape(builder, op.getLoc(), tensor);
          ShapedType outputTy = dyn_cast<ShapedType>(trimmed_tensor.getType());
          Value outputAsMemref = createOrFoldUnrealizedConversionCast(
              op.getLoc(), rewriter, convertTensorToMemref(outputTy), trimmed_tensor);
          outputs.push_back(outputAsMemref); 
        }
      }
    }

    for(auto &operation: op.getBody().front().getOperations()){
      if(dyn_cast<linalg::AddOp>(operation)){
        createVALaunch(builder, totalPE, remapped_wg, inputs, outputs[0]);
      } else if (dyn_cast<linalg::MatvecOp>(operation)){
        createGEMVLaunch(builder, remapped_wg, inputs, outputs[0]);
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
    ,ConvertCnmLaunchToHbmpim
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
    target.addLegalOp<cnm::GatherOp>();
    // target.addIllegalDialect<bufferization::BufferizationDialect>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
    getOperation()->walk([](cnm::GatherOp op) { op->erase(); });
    getOperation()->walk([](cnm::ScatterOp op) { op->erase(); });
    getOperation()->walk([](cnm::AllocOp op) { op->erase(); });

  }
};

std::unique_ptr<Pass> createConvertCnmToHbmpimPass() {
  return std::make_unique<ConvertCnmToHbmpimPass>();
}

} // namespace mlir::cnm
