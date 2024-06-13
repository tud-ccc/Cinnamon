#include "cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"
#include "cinm-mlir/Conversion/UPMEMPasses.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ValueRange.h>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir{
#define GEN_PASS_DEF_CONVERTUPMEMTOLLVMPASS
#include "cinm-mlir/Conversion/UPMEMPasses.h.inc"
}

// namespace mlir{
// static func::FuncOp createToOffsetCalcFunction(OpBuilder &builder, upmem::ScatterOp scatterOp, AffineMap scatterMap){

    // OpBuilder builder(scatterOp.getContext());
    // auto offsetCalcFuncType = FunctionType::get(
    //     rewriter.getContext(), rewriter.getIntegerType(32), rewriter.getIntegerType(32));
    // std::string funcName("dpu_offset_calc");
    // MLIRContext *context = module.getContext();
    // auto result = SymbolRefAttr::get(context, funcName);
    // auto funcOp = module.lookupSymbol<func::FuncOp>(result.getAttr());
    // if (!funcOp){

      // FunctionType funcType = FunctionType::get(
      //     builder.getContext(), {}, {});
      // funcOp = builder.create<func::FuncOp>(funcName, funcType);

      // OpBuilder::InsertionGuard insertionGuard(builder);
      // builder.setInsertionPoint(insertPoint);
      // Location loc = insertPoint.getLoc();
      // func = builder.create<func::FuncOp>(
      //     loc, nameOstream.str(),
      //     FunctionType::get(context, operands.getTypes(), resultTypes));
      // func.setPrivate();
    // }

    // auto offsetCalcFuncType = FunctionType::get(
    //     builder.getContext(), {}, {});

    // auto funcAttr = StringAttr::get(op->getContext(), funcName);
    // Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    // if (funcOp)
    //   return cast<LLVMFuncOp>(*funcOp);

    // auto offsetCalcFunc = builder.create<func::FuncOp>(scatterOp.getLoc(), funcName, offsetCalcFuncType);
    // Block *funcBody = offsetCalcFunc.addEntryBlock();
    // builder.setInsertionPointToEnd(funcBody);
    
    // // rewriter.create<func::ReturnOp>(loc, offsetCalcFunc.getArgument(0));
    // builder.create<func::ReturnOp>(scatterOp.getLoc());
    // return funcOp;
    // Region &outlinedFuncBody = offsetCalcFunc.getBody();
    // Block &outlinedEntryBlock = outlinedFuncBody.front();

    // // 
    // llvm::SmallVector<AffineExpr> affineDims;
    // upmem::DeviceHierarchyType hierarchyType = op.getHierarchy().getType();
    // ArrayRef<int64_t> hierarchy = hierarchyType.getShape();
    // AffineExpr index = mlir::getAffineDimExpr(0, hierarchyType.getContext());
    // affineDims.push_back(index.floorDiv(hierarchy[1]*hierarchy[2]));
    // affineDims.push_back(index.floorDiv(hierarchy[2]) % hierarchy[1]);
    // affineDims.push_back(index % hierarchy[2]);

    // AffineMap toLinearMap = AffineMap::get(3, 0,
    //                           affineDims, hierarchyType.getContext());
    // newIndex = rewriter.create<AffineApplyOp>(
    //   loc, annotation.getMap().compose(lowerAndStep),
    //   ValueRange{operand, step, lowerBound});

    // builder.setInsertionPointToEnd(&outlinedEntryBlock);

    // for (auto &op : launchOpEntry.without_terminator()) {
    //   builder.clone(op, map);
    // }
    // builder.create<upmem::ReturnOp>(launchOpEntry.getTerminator()->getLoc());


// }
// }


namespace mlir::upmem {
namespace upmemtollvm {

static LLVM::LLVMFuncOp createCalcOffset(Operation *op) { 
  using LLVM::LLVMFuncOp;
  StringRef funcName = "calc_scatter_offset";
  Type funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(op->getContext()), {});

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);


  mlir::OpBuilder b(op->getParentOfType<FunctionOpInterface>());
  auto newFunc =  b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);

  Region &newFuncBody = newFunc.getBody();
  newFuncBody.emplaceBlock();
  Block &newFuncBlock = newFuncBody.front();
  b.setInsertionPointToEnd(&newFuncBlock);
  // b.create<LLVM::ReturnOp>(op.getLoc(), ValueRange());
  return newFunc;
}

static Value maybeCast(Value operand, PatternRewriter &rewriter){
  Type type = operand.getType();
  if (!isa<Float16Type>(type))
    return operand;

  return rewriter.create<LLVM::FPExtOp>(
      operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
}

static Type getFunctionType(Type resultType, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

static LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type funcType,
                                          Operation *op) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  mlir::OpBuilder b(op->getParentOfType<FunctionOpInterface>());
  return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
}

struct AllocDPUOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::AllocDPUsOp> {
public:
  explicit AllocDPUOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::AllocDPUsOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::AllocDPUsOp op,
                  typename upmem::AllocDPUsOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;
    const ArrayRef<int64_t> hierarchyShape =
        op.getResult().getType().getShape();
    const Value rankCount = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(hierarchyShape[0]));
    const Value dpuCount = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(hierarchyShape[1]));

    SmallVector<Value, 1> castedOperands;
    castedOperands.push_back(maybeCast(rankCount, rewriter));
    castedOperands.push_back(maybeCast(dpuCount, rewriter));
    Type resultType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type funcType = getFunctionType(resultType, castedOperands);

    LLVMFuncOp funcOp = appendOrGetFuncOp("alloc_dpu", funcType, op);
    // auto callOp =
    //     rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, castedOperands);
    return success();
  }
};

struct ScatterOpToFuncCallLowering : public ConvertOpToLLVMPattern<upmem::ScatterOp> {
public:
  explicit ScatterOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::ScatterOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::ScatterOp op,
                  typename upmem::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    SmallVector<Value, 1> castedOperands;
    castedOperands.push_back(adaptor.getHierarchy());
    Value promoted_memref = getTypeConverter()->promoteOneMemRefDescriptor(
        op.getLoc(), adaptor.getHostBuffer(), rewriter);
    castedOperands.push_back(promoted_memref);
    // castedOperands.push_back(maybeCast(op.getHostData(), rewriter));
    castedOperands.push_back(rewriter.getRemappedValue(op.getDpuMemOffset()));

    // for (Value operand : adaptor.getOperands())
    // castedOperands.push_back(maybeCast(operand, rewriter));

    SmallVector<Type> operandTypes;
    operandTypes.push_back(castedOperands[0].getType());
    operandTypes.push_back(castedOperands[1].getType());
    operandTypes.push_back(rewriter.getIntegerType(64));

    Type funcType = LLVM::LLVMFunctionType::get(rewriter.getIntegerType(32), {});

    AffineMap scatter_map = op.getScatterMap();
    LLVMFuncOp funcOp = createCalcOffset(op);
    rewriter.create<LLVM::CallOp>(op.getLoc(), funcOp, ValueRange());

    rewriter.eraseOp(op);
    // rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct GatherOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::GatherOp> {
public:
  explicit GatherOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::GatherOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::GatherOp op, typename upmem::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    SmallVector<Value, 1> castedOperands;
    castedOperands.push_back(adaptor.getHierarchy());
    Value promoted_memref = getTypeConverter()->promoteOneMemRefDescriptor(
        op.getLoc(), adaptor.getHostBuffer(), rewriter);
    castedOperands.push_back(promoted_memref);
    // castedOperands.push_back(maybeCast(op.getHostData(), rewriter));
    castedOperands.push_back(adaptor.getDpuMemOffset());
    // for (Value operand : adaptor.getOperands())
    // castedOperands.push_back(maybeCast(operand, rewriter));

    Type resultType = rewriter.getIntegerType(32);
    Type funcType = getFunctionType(resultType, castedOperands);

    LLVMFuncOp funcOp = appendOrGetFuncOp("upmem_gather", funcType, op);
    // auto callOp =
    //     rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, castedOperands);
    // rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct LaunchFuncOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::LaunchFuncOp> {
public:
  explicit LaunchFuncOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::LaunchFuncOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::LaunchFuncOp op,
                  typename upmem::LaunchFuncOp ::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    SmallVector<Value, 1> castedOperands;

    castedOperands.push_back(rewriter.getRemappedValue(op.getUpmemToken()));

    Type resultType = LLVM::LLVMVoidType::get(rewriter.getContext());
    Type funcType = getFunctionType(resultType, castedOperands);

    LLVMFuncOp funcOp = appendOrGetFuncOp("upmem_launch_dpu", funcType, op);
    // auto callOp =
    //     rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp, castedOperands);
    // rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct BaseDPUMemOffsetOpLowering
    : public OpConversionPattern<upmem::BaseDPUMemOffsetOp> {
public:
  using OpConversionPattern<upmem::BaseDPUMemOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(upmem::BaseDPUMemOffsetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getI32IntegerAttr(0));
    return success();
  }
};

struct EraseUPMEMModule : public OpConversionPattern<upmem::UPMEMModuleOp> {
  using OpConversionPattern<upmem::UPMEMModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(upmem::UPMEMModuleOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// struct ConvertCnmGatherToUPMEM : public OpConversionPattern<cnm::GatherOp> {
//   using OpConversionPattern<cnm::GatherOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(cnm::GatherOp op, OpAdaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     // rewriter.replaceOp(op, results);
//     return success();
//   }
// };

// struct ConvertCnmLaunchToUPMEM : public OpConversionPattern<cnm::LaunchOp> {
//   using OpConversionPattern<cnm::LaunchOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(cnm::LaunchOp op, OpAdaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     return success();
//   }
// };

// struct ConvertCnmTerminatorToUPMEM
//     : public OpConversionPattern<cnm::TerminatorOp> {
//   using OpConversionPattern<cnm::TerminatorOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(cnm::TerminatorOp op, OpAdaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     rewriter.eraseOp(op); // gets generated by ConvertCnmLaunchToUPMEM
//     return success();
//   }
// };

} // namespace upmemtollvm

void populateUPMEMToLLVMFinalTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](upmem::DeviceHierarchyType hierarchyType) -> std::optional<Type> {
        return LLVM::LLVMPointerType::get(hierarchyType.getContext(), 0);
      });

  // typeConverter.addConversion(
  //     [&](cnm::BufferType bufferType) -> std::optional<Type> {
  //       return cnmtoupmem::convertCnmBufferToMemRefType(bufferType);
  //     });
}

void populateUPMEMToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  patterns.add<upmemtollvm::AllocDPUOpToFuncCallLowering>(typeConverter);
  patterns.add<upmemtollvm::ScatterOpToFuncCallLowering>(typeConverter);
  patterns.add<upmemtollvm::GatherOpToFuncCallLowering>(typeConverter);
  patterns.add<upmemtollvm::LaunchFuncOpToFuncCallLowering>(typeConverter);
  patterns.add<upmemtollvm::BaseDPUMemOffsetOpLowering>(
      &typeConverter.getContext());
  patterns.add<upmemtollvm::EraseUPMEMModule>(&typeConverter.getContext());
}

struct ConvertUPMEMToLLVMPass
    : public impl::ConvertUPMEMToLLVMPassBase<ConvertUPMEMToLLVMPass> {
  void runOnOperation() final {
    // ModuleOp module = getOperation();
    LLVMTypeConverter converter(&getContext());
    populateUPMEMToLLVMFinalTypeConversions(converter);
    const auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateUPMEMToLLVMConversionPatterns(converter, patterns);
    populateReconcileUnrealizedCastsPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<upmem::UPMEMDialect>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createConvertUPMEMToLLVMPass() {
  return std::make_unique<ConvertUPMEMToLLVMPass>();
}

} // namespace mlir::upmem
