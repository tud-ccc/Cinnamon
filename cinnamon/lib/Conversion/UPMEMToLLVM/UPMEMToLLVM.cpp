#include "cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Conversion/UPMEMPasses.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTUPMEMTOLLVMPASS
#include "cinm-mlir/Conversion/UPMEMPasses.h.inc"
} // namespace mlir

namespace mlir::upmem {
namespace {

static LLVM::LLVMPointerType untypedPtrType(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx, 0);
}

static LLVM::LLVMPointerType functionPtrTy(Type resultTy, ArrayRef<Type>) {
  return untypedPtrType(resultTy.getContext());
}

static Value reifyAsIndex(ImplicitLocOpBuilder &builder,
                          LLVMTypeConverter const *converter, int64_t value) {
  return builder.create<LLVM::ConstantOp>(converter->getIndexType(), value);
}

static SmallString<20> getUniqueFunctionName(ModuleOp moduleOp,
                                             const char prefix[]) {
  // Get a unique global name.
  unsigned stringNumber = 0;
  size_t prefixLen = strlen(prefix);
  assert(20 > 3 + prefixLen); // make sure this is bigger than the prefix
                              // (prefixes are literals)
  SmallString<20> name(prefix);
  do {
    name.truncate(prefixLen);
    name.append(std::to_string(stringNumber++));
  } while (moduleOp.lookupSymbol(name));
  return name;
}

// TODO these two functions are duplicated from CinmToCnm.cpp, share them

// Turn an index in the index space of the given shape into a linear index.
AffineExpr linearizeIndices(MLIRContext *ctx, ArrayRef<int64_t> shape) {

  AffineExpr index = getAffineConstantExpr(0, ctx);
  int64_t dimIndex = shape.size() - 1;
  int64_t trailing = 1;
  for (auto it = shape.rbegin(); it != shape.rend(); it++) {
    auto dim = *it;
    index = trailing * getAffineDimExpr(dimIndex, ctx) + index;
    trailing *= dim;
    dimIndex--;
  }
  return index;
}

// inflate a linear index into the given shape
void structureIndex(AffineExpr index, ArrayRef<int64_t> shape,
                    SmallVectorImpl<AffineExpr> &map) {

  int64_t sizeOfTrailing = computeProduct(shape) / shape[0];
  map.push_back(index.floorDiv(sizeOfTrailing));

  AffineExpr gatherExpr = index * sizeOfTrailing;
  size_t i = 1;

  for (auto dim : llvm::drop_begin(shape, 1)) {
    index = index % sizeOfTrailing;
    sizeOfTrailing /= dim;
    map.push_back(index.floorDiv(sizeOfTrailing));
    gatherExpr = gatherExpr +
                 mlir::getAffineDimExpr(i, index.getContext()) * sizeOfTrailing;
    i++;
  }
}
static AffineMap linearizeAffineMap(AffineMap map, ArrayRef<int64_t> inputShape,
                                    ArrayRef<int64_t> outputShape) {
  auto ctx = map.getContext();
  SmallVector<AffineExpr> inflatedIndices;
  structureIndex(getAffineDimExpr(0, ctx), inputShape, inflatedIndices);
  AffineMap inflateMap = AffineMap::get(1, 0, inflatedIndices, ctx);

  // complete map with zero dims
  // todo do that in CNM->UPMEM
  auto zero = getAffineConstantExpr(0, ctx);
  for (int i = map.getNumResults(); i < outputShape.size(); i++) {
    map = map.insertResult(zero, i);
  }

  auto linearIndex = linearizeIndices(ctx, outputShape);
  AffineMap linearMap = AffineMap::get(outputShape.size(), 0, linearIndex);

  auto result = MutableAffineMap(linearMap.compose(map).compose(inflateMap));
  result.simplify();
  assert(result.getNumResults() == 1 && result.getNumDims() == 1);
  return result.getAffineMap();
}

/*
void upmemrt_scatter_dpu(struct dpu_set_t *dpu_set, void *A, size_t input_size,
                       size_t copy_bytes, size_t offset_in_dpu,
                       size_t (*base_offset)(size_t));
*/
static LLVM::LLVMFuncOp
getScatterOrGatherFunc(ModuleOp moduleOp, LLVMTypeConverter const *tyConverter,
                       StringRef name) {
  auto ctx = moduleOp->getContext();
  auto ptrTy = untypedPtrType(ctx);
  auto sizeTy = tyConverter->getIndexType();
  auto funPtrTy = functionPtrTy(sizeTy, {sizeTy});
  return LLVM::lookupOrCreateFn(
      moduleOp, name, {ptrTy, ptrTy, sizeTy, sizeTy, sizeTy, funPtrTy},
      LLVM::LLVMVoidType::get(ctx));
}

static LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type resultType,
                                          ArrayRef<Type> paramTypes,
                                          Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  return LLVM::lookupOrCreateFn(module, funcName, paramTypes, resultType);
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
    const ArrayRef<int64_t> hierarchyShape =
        op.getResult().getType().getShape();
    const Value rankCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(hierarchyShape[0]));
    const Value dpuCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(hierarchyShape[1]));

    // struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus);
    Type resultType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetFuncOp("upmemrt_dpu_alloc", resultType,
                          {rewriter.getI32Type(), rewriter.getI32Type()}, op);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp,
                                              ValueRange{rankCount, dpuCount});
    return success();
  }
};

static std::optional<LLVM::LLVMFuncOp>
outlineAffineMap(ImplicitLocOpBuilder &rewriter,
                 LLVMTypeConverter const *tyConverter, ModuleOp moduleOp,
                 AffineMap map, DeviceHierarchyType hierarchyTy,
                 ShapedType bufferTy) {

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto sizeTy =
      tyConverter->convertType(IndexType::get(moduleOp->getContext()));
  auto linearMap =
      linearizeAffineMap(map, hierarchyTy.getShape(), bufferTy.getShape());
  auto affineFunTy = LLVM::LLVMFunctionType::get(sizeTy, {sizeTy});
  LLVM::LLVMFuncOp existingOp;
  moduleOp.getBodyRegion().walk<WalkOrder::PreOrder>([&](LLVM::LLVMFuncOp op) {
    if (auto map = op->getAttrOfType<AffineMapAttr>("upmem.generated_from"))
      if (map.getAffineMap() == linearMap) {
        existingOp = op;
        return WalkResult::interrupt();
      }
    return WalkResult::skip();
  });
  if (existingOp)
    return existingOp;

  auto affineMapFun = rewriter.create<LLVM::LLVMFuncOp>(
      getUniqueFunctionName(moduleOp, "scatter_map_"), affineFunTy,
      LLVM::Linkage::Private);

  // to find it later
  affineMapFun->setAttr("upmem.generated_from", AffineMapAttr::get(linearMap));

  rewriter = ImplicitLocOpBuilder::atBlockBegin(rewriter.getLoc(),
                                                affineMapFun.addEntryBlock());
  Value arg = affineMapFun.getArgument(0);
  // affine expects to deal with index type only
  arg = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                             rewriter.getIndexType(), arg);

  if (auto resOpt = affine::expandAffineMap(rewriter, rewriter.getLoc(),
                                            linearMap, ValueRange{arg})) {
    auto result = (*resOpt)[0];
    result = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                                  sizeTy, result);
    rewriter.create<LLVM::ReturnOp>(ValueRange{result});
    return affineMapFun;
  }
  return std::nullopt;
}

template <class Op>
static LogicalResult lowerScatterOrGather(Op op, typename Op::Adaptor adaptor,
                                          LLVMTypeConverter const *tyConverter,
                                          ConversionPatternRewriter &rewriter0,
                                          bool isGather) {
  ImplicitLocOpBuilder rewriter(op->getLoc(), rewriter0);

  /*
  The scatter/gather op does these things:
  - generate a function that implements the affine map
  - call upmemrt_[scatter/gather]_dpu
  - return the new offset in the dpu (only scatter, not gather)
  */

  // generate the function
  auto moduleOp = op->template getParentOfType<ModuleOp>();

  auto affineMapFunOpt = outlineAffineMap(
      rewriter, tyConverter, moduleOp, op.getScatterMap(),
      op.getHierarchy().getType(), op.getHostBuffer().getType());
  if (!affineMapFunOpt)
    return emitError(op->getLoc(), "Cannot emit affine map");
  /*
  void upmemrt_scatter_dpu(struct dpu_set_t *dpu_set, void *A, size_t
  input_size, size_t copy_bytes, size_t offset_in_dpu, size_t
  (*base_offset)(size_t));
  */
  LLVM::LLVMFuncOp runtimeScatterFun = getScatterOrGatherFunc(
      moduleOp, tyConverter,
      isGather ? "upmemrt_dpu_gather" : "upmemrt_dpu_gather");

  auto funPtrOp = rewriter.create<LLVM::AddressOfOp>(*affineMapFunOpt);
  auto numBytesCopied = reifyAsIndex(
      rewriter, tyConverter,
      op.getCount() * op.getHostBuffer().getType().getElementTypeBitWidth());

  Value bareHostBuf = adaptor.getHostBuffer();
  if (adaptor.getHostBuffer().getType().template isa<LLVM::LLVMStructType>()) {
    // converted memref
    bareHostBuf =
        rewriter.create<LLVM::ExtractValueOp>(adaptor.getHostBuffer(), 1);
  } else {
    return emitError(op->getLoc(), "Unhandled buffer type: ")
           << adaptor.getHostBuffer().getType();
  }

  rewriter.create<LLVM::CallOp>(
      runtimeScatterFun,
      ValueRange{
          adaptor.getHierarchy(), bareHostBuf,
          reifyAsIndex(rewriter, tyConverter,
                       computeProduct(op.getHostBuffer().getType().getShape())),
          numBytesCopied, adaptor.getDpuMemOffset(), funPtrOp.getRes()});

  if (op->getNumResults() > 0) {
    // only for scatter
    auto resultOffValue = rewriter.create<LLVM::AddOp>(
        tyConverter->getIndexType(), adaptor.getDpuMemOffset(), numBytesCopied);

    rewriter0.replaceAllUsesWith(op->getResult(0), resultOffValue.getResult());
  }
  rewriter0.eraseOp(op);
  return success();
}

struct ScatterOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::ScatterOp> {
public:
  explicit ScatterOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::ScatterOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::ScatterOp op,
                  typename upmem::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerScatterOrGather(op, adaptor, getTypeConverter(), rewriter0,
                                false);
  }
};

struct GatherOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::GatherOp> {
public:
  explicit GatherOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::GatherOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::GatherOp op, typename upmem::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerScatterOrGather(op, adaptor, getTypeConverter(), rewriter0,
                                true);
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

    Type resultType = LLVM::LLVMVoidType::get(rewriter.getContext());

    // void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set) {
    LLVM::LLVMFuncOp funcOp = appendOrGetFuncOp(
        "upmemrt_dpu_launch", resultType,
        {getTypeConverter()->convertType(op.getUpmemToken().getType())}, op);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp,
                                              adaptor.getUpmemToken());
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
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
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

} // namespace

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
  patterns.add<AllocDPUOpToFuncCallLowering>(typeConverter);
  patterns.add<ScatterOpToFuncCallLowering>(typeConverter);
  patterns.add<GatherOpToFuncCallLowering>(typeConverter);
  patterns.add<LaunchFuncOpToFuncCallLowering>(typeConverter);
  patterns.add<BaseDPUMemOffsetOpLowering>(&typeConverter.getContext());
  patterns.add<EraseUPMEMModule>(&typeConverter.getContext());
}

struct ConvertUPMEMToLLVMPass
    : public impl::ConvertUPMEMToLLVMPassBase<ConvertUPMEMToLLVMPass> {
  void runOnOperation() final {
    // ModuleOp module = getOperation();
    LowerToLLVMOptions convOptions(&getContext());
    // necessary for C interop
    convOptions.useBarePtrCallConv = true;
    LLVMTypeConverter converter(&getContext(), convOptions);
    populateUPMEMToLLVMFinalTypeConversions(converter);
    const auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs,
                                      Location loc) -> Value {
      // if (type.isa<BaseMemRefType>() && inputs.size() == 1 &&
      //     inputs[0].getType().isa<RankedTensorType>()) {
      //   return builder.create<bufferization::ToMemrefOp>(loc, type, inputs)
      //       .getResult();
      // }
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    converter.addConversion([](RankedTensorType ty) -> std::optional<Type> {
      return MemRefType::get(ty.getShape(), ty.getElementType());
    });

    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateUPMEMToLLVMConversionPatterns(converter, patterns);
    populateReconcileUnrealizedCastsPatterns(patterns);
    populateFinalBufferizationPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<upmem::UPMEMDialect>();
    target.addIllegalDialect<bufferization::BufferizationDialect>();

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
