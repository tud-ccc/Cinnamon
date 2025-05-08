#include "cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Conversion/UPMEMPasses.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"
#include <cinm-mlir/Utils/CinmUtils.h>

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>

#define DEBUG_TYPE "upmem-to-llvm"

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

static LLVM::GlobalOp
declareStringConstant(ModuleOp moduleOp, Location loc, StringRef value,
                      bool zeroTerminated = true,
                      std::optional<StringRef> globalName = std::nullopt) {
  llvm::SmallString<20> str(value);
  if (zeroTerminated)
    str.push_back('\0'); // Null terminate for C

  OpBuilder rewriter(moduleOp->getContext());
  auto globalType =
      LLVM::LLVMArrayType::get(rewriter.getI8Type(), str.size_in_bytes());
  auto twine = llvm::Twine("const", value).str();
  auto name = getUniqueFunctionName(moduleOp, globalName.value_or(twine));

  rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  return rewriter.create<LLVM::GlobalOp>(
      loc, globalType,
      /*isConstant=*/true, LLVM::Linkage::Private, rewriter.getStringAttr(name),
      rewriter.getStringAttr(str),
      /*allignment=*/0);
}

static Value reifyAsString(ImplicitLocOpBuilder &builder, ModuleOp container,
                           StringRef value) {
  LLVM::GlobalOp global = declareStringConstant(container, builder.getLoc(),
                                                value, true, "prog_name");
  return builder.create<LLVM::AddressOfOp>(global);
}

/// Linearize the scatter map.
/// The map is from (rank, dpu) -> tensor, both index spaces are
/// multidimensional. The input shape is the WG shape, the output shape is the
/// tensor shape.
///
static FailureOr<AffineMap> linearizeAffineMap(AffineMap map,
                                               ArrayRef<int64_t> inputShape,
                                               MemRefType bufferTy) {

  auto ctx = map.getContext();
  SmallVector<AffineExpr> inflatedIndices;
  cinm::structureIndex(getAffineDimExpr(0, ctx), inputShape, inflatedIndices);
  AffineMap inflateMap = AffineMap::get(1, 0, inflatedIndices, ctx);

  // complete map with zero dims
  auto outputShape = bufferTy.getShape();
  auto layoutMap = bufferTy.getLayout().getAffineMap();
  if (isa<StridedLayoutAttr>(bufferTy.getLayout())) {
    // Replace offsets with 0 to delete the symbols.
    // Offset is calculated outside of the affine map.
    layoutMap = layoutMap.replaceDimsAndSymbols(
        {}, {getAffineConstantExpr(0, ctx)}, layoutMap.getNumDims(), 0);
  } else if (bufferTy.getLayout().isIdentity()) {
    auto linearIndex = cinm::linearizeIndices(ctx, outputShape);
    layoutMap = AffineMap::get(outputShape.size(), 0, linearIndex);
  } else {
    return failure();
  }
  LLVM_DEBUG(llvm::errs() << "linearize composition :\n");
  LLVM_DEBUG(llvm::errs() << "- output type " << bufferTy << '\n');
  LLVM_DEBUG(llvm::errs() << "- layout map " << layoutMap << '\n');
  LLVM_DEBUG(llvm::errs() << "- map " << map << '\n');
  LLVM_DEBUG(llvm::errs() << "- inflate map " << inflateMap << '\n');

  auto result = MutableAffineMap(layoutMap.compose(map).compose(inflateMap));
  LLVM_DEBUG(llvm::errs() << "- before simplification " << result.getAffineMap() << '\n');
  result.simplify();
  LLVM_DEBUG(llvm::errs() << "- after simplification " << result.getAffineMap() << '\n');

  assert(result.getNumResults() == 1 && result.getNumDims() == 1);

  // last step is making sure this map operates on bytes and not on elements
  auto resExpr = result.getResult(0);
  result.setResult(0, resExpr * (bufferTy.getElementTypeBitWidth() / 8));
  result.simplify();
  LLVM_DEBUG(llvm::errs() << "- result " << result.getAffineMap() << '\n');
  return success(result.getAffineMap());
}

/*
size_t upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t element_size, size_t num_elements,
                           size_t num_elements_per_tasklet, size_t copy_bytes,
                           char* buffer_id,
                           size_t (*base_offset)(size_t));
*/
static FailureOr<LLVM::LLVMFuncOp>
getScatterOrGatherFunc(OpBuilder &rewriter, ModuleOp moduleOp,
                       LLVMTypeConverter const *tyConverter, StringRef name) {
  auto ctx = moduleOp->getContext();
  auto ptrTy = untypedPtrType(ctx);
  auto sizeTy = tyConverter->getIndexType();
  auto funPtrTy = functionPtrTy(sizeTy, {sizeTy});
  return LLVM::lookupOrCreateFn(
      rewriter, moduleOp, name,
      {ptrTy, ptrTy, sizeTy, sizeTy, sizeTy, sizeTy, ptrTy, funPtrTy},
      LLVM::LLVMVoidType::get(ctx));
}

static FailureOr<LLVM::LLVMFuncOp>
appendOrGetFuncOp(OpBuilder &rewriter, StringRef funcName, Type resultType,
                  ArrayRef<Type> paramTypes, Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  return LLVM::lookupOrCreateFn(rewriter, module, funcName, paramTypes,
                                resultType);
}

struct FreeDPUsOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::FreeDPUsOp> {
public:
  explicit FreeDPUsOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::FreeDPUsOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::FreeDPUsOp op,
                  typename upmem::FreeDPUsOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = LLVM::LLVMVoidType::get(rewriter.getContext());

    // void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set) {

    auto funcOp = appendOrGetFuncOp(
        rewriter, "upmemrt_dpu_free", resultType,
        {getTypeConverter()->convertType(op.getHierarchy().getType())}, op);
    if (llvm::failed(funcOp))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, *funcOp,
                                              adaptor.getHierarchy());

    return success();
  }
};

struct AllocDPUOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::AllocDPUsOp> {
public:
  explicit AllocDPUOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::AllocDPUsOp>(lowering) {}

  FailureOr<Value>
  createConstantForDpuProgramName(ConversionPatternRewriter &rewriter,
                                  upmem::AllocDPUsOp op) const {
    std::string dpuProgramName;
    dpuProgramName.append(op.getDpuProgramRef().getRootReference().strref());
    for (auto segment : op.getDpuProgramRef().getNestedReferences()) {
      dpuProgramName.append(segment.getValue());
    }

    LLVM::GlobalOp constant =
        declareStringConstant(op->getParentOfType<ModuleOp>(), op->getLoc(),
                              dpuProgramName, true, "dpu_program");
    Value result = rewriter.create<LLVM::AddressOfOp>(op->getLoc(), constant);
    return success(result);
  }

  LogicalResult
  matchAndRewrite(upmem::AllocDPUsOp op, typename upmem::AllocDPUsOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const DeviceHierarchyType hierarchyShape = op.getResult().getType();
    const Value rankCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(hierarchyShape.getNumRanks()));
    const Value dpuCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(hierarchyShape.getNumDpusPerRank()));

    const auto maybeFailed = createConstantForDpuProgramName(rewriter, op);
    if (failed(maybeFailed))
      return failure();
    const Value dpuProgramPath = *maybeFailed;

    // struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t
    // num_dpus);
    Type resultType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    auto funcOp =
        appendOrGetFuncOp(rewriter, "upmemrt_dpu_alloc", resultType,
                          {rewriter.getI32Type(), rewriter.getI32Type(),
                           untypedPtrType(getContext())},
                          op);

    if (llvm::failed(funcOp))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, *funcOp, ValueRange{rankCount, dpuCount, dpuProgramPath});
    return success();
  }
};

static FailureOr<LLVM::LLVMFuncOp>
outlineAffineMap(ImplicitLocOpBuilder &rewriter,
                 LLVMTypeConverter const *tyConverter, ModuleOp moduleOp,
                 AffineMap map, DeviceHierarchyType hierarchyTy,
                 MemRefType bufferTy) {

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto sizeTy =
      tyConverter->convertType(IndexType::get(moduleOp->getContext()));
  auto shape = hierarchyTy.getWgShape();
  auto linearMap =
      linearizeAffineMap(map, ArrayRef<int64_t>(shape).drop_back(), bufferTy);
  if (failed(linearMap)) {
    emitError(rewriter.getLoc(), "Unsupported layout map for ") << bufferTy;
    return failure();
  }

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
  auto funName = getUniqueFunctionName(moduleOp, "scatter_map");
  rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  auto affineMapFun = rewriter.create<LLVM::LLVMFuncOp>(
      rewriter.getStringAttr(funName), affineFunTy, LLVM::Linkage::Private);

  // to find it later
  affineMapFun->setAttr("upmem.generated_from", AffineMapAttr::get(*linearMap));

  rewriter = ImplicitLocOpBuilder::atBlockBegin(
      rewriter.getLoc(), affineMapFun.addEntryBlock(rewriter));
  Value arg = affineMapFun.getArgument(0);
  // affine expects to deal with index type only
  arg = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                             rewriter.getIndexType(), arg);

  if (auto resOpt = affine::expandAffineMap(rewriter, rewriter.getLoc(),
                                            *linearMap, ValueRange{arg})) {
    auto result = (*resOpt)[0];
    result = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                                  sizeTy, result);
    rewriter.create<LLVM::ReturnOp>(ValueRange{result});
    return affineMapFun;
  }
  return failure();
}

template <class Op>
static LogicalResult lowerScatterOrGather(Op op, typename Op::Adaptor adaptor,
                                          LLVMTypeConverter const *tyConverter,
                                          ConversionPatternRewriter &rewriter0,
                                          bool isGather) {
  auto loc = op->getLoc();
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
  if (failed(affineMapFunOpt)) {
    return emitError(op->getLoc(), "Cannot emit affine map");
  }

  /*
  void upmemrt_scatter_dpu(struct dpu_set_t *dpu_set, void *A, size_t
  input_size, size_t copy_bytes, size_t offset_in_dpu, size_t
  (*base_offset)(size_t));
  */
  auto runtimeScatterFun = getScatterOrGatherFunc(
      rewriter, moduleOp, tyConverter,
      isGather ? "upmemrt_dpu_gather" : "upmemrt_dpu_scatter");

  if (llvm::failed(runtimeScatterFun))
    return failure();
  auto funPtrOp = rewriter0.create<LLVM::AddressOfOp>(loc, *affineMapFunOpt);
  auto bufferId = reifyAsString(rewriter, moduleOp, op.getDpuBufRef());
  auto numBytesCopied = op.getDpuBufferSizeInBytes();

  // Transfer count must be 8-byte aligned
  // TODO probably means we must pad the input
  if (numBytesCopied % 8 != 0) {
    numBytesCopied += 8 - (numBytesCopied % 8);
  }

  Value bareHostBuf = adaptor.getHostBuffer();
  if (isa<LLVM::LLVMStructType>(adaptor.getHostBuffer().getType())) {
    // Here we compute the pointer to the start of the memref
    // converted memref
    Value basePtr =
        rewriter0.create<LLVM::ExtractValueOp>(loc, adaptor.getHostBuffer(), 1);
    Value offset =
        rewriter0.create<LLVM::ExtractValueOp>(loc, adaptor.getHostBuffer(), 2);
    // need to do our own pointer arithmetic here
    bareHostBuf = rewriter0.create<LLVM::GEPOp>(
        loc, basePtr.getType(), op.getHostBuffer().getType().getElementType(),
        basePtr, ValueRange{offset});
  } else {
    return emitError(op->getLoc(), "Unhandled buffer type: ")
           << adaptor.getHostBuffer().getType();
  }

  const size_t elementSize =
      op.getHostBuffer().getType().getElementTypeBitWidth() / 8;
  const size_t numTasklets = op.getHierarchy().getType().getNumElements();
  const size_t numElements =
      computeProduct(op.getHostBuffer().getType().getShape());
  const size_t numElementsPerTasklet = numElements / numTasklets;

  rewriter0.create<LLVM::CallOp>(
      loc, *runtimeScatterFun,
      ValueRange{adaptor.getHierarchy(), bareHostBuf,
                 reifyAsIndex(rewriter, tyConverter, elementSize),
                 reifyAsIndex(rewriter, tyConverter, numElements),
                 reifyAsIndex(rewriter, tyConverter, numElementsPerTasklet),
                 reifyAsIndex(rewriter, tyConverter, numBytesCopied), bufferId,
                 funPtrOp.getRes()});

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

struct WaitForOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::WaitForOp> {
  using ConvertOpToLLVMPattern<upmem::WaitForOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(upmem::WaitForOp op,
                  typename upmem::WaitForOp ::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = LLVM::LLVMVoidType::get(rewriter.getContext());

    // void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set) {
    auto funcOp = appendOrGetFuncOp(
        rewriter, "upmemrt_dpu_launch", resultType,
        {getTypeConverter()->convertType(op.getDpuSet().getType())}, op);

    if (llvm::failed(funcOp))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, *funcOp, adaptor.getDpuSet());
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

struct EraseDpuProgram : public ConvertOpToLLVMPattern<upmem::DpuProgramOp> {
  using ConvertOpToLLVMPattern<upmem::DpuProgramOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(upmem::DpuProgramOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

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
  patterns.add<WaitForOpToFuncCallLowering>(typeConverter);
  patterns.add<FreeDPUsOpToFuncCallLowering>(typeConverter);
  patterns.add<EraseDpuProgram>(typeConverter);
  patterns.add<BaseDPUMemOffsetOpLowering>(&typeConverter.getContext());
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
      // if (isa<BaseMemRefType>(type) && inputs.size() == 1 &&
      //     isa<RankedTensorType>(inputs[0].getType())) {
      //   return builder.create<bufferization::ToMemrefOp>(loc, type, inputs)
      //       .getResult();
      // }
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateUPMEMToLLVMConversionPatterns(converter, patterns);

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
