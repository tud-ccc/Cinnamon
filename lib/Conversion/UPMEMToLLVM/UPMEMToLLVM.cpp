#include "cinm-mlir/Conversion/UPMEMToLLVM/UPMEMToLLVM.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Conversion/UPMEMPasses.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMTypes.h"
#include <cinm-mlir/Utils/CinmUtils.h>

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
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
  return LLVM::ConstantOp::create(builder, converter->getIndexType(), value);
}

static LLVM::GlobalOp
declareStringConstant(ModuleOp moduleOp, Location loc, StringRef value,
                      bool zeroTerminated = true,
                      std::optional<StringRef> globalName = std::nullopt) {
  llvm::SmallString<20> str(value);
  if (zeroTerminated)
    str.push_back('\0'); // Null terminate for C

  OpBuilder builder(moduleOp->getContext());
  auto globalType =
      LLVM::LLVMArrayType::get(builder.getI8Type(), str.size_in_bytes());
  auto valueAttr = builder.getStringAttr(std::move(str));

  // Try to find an existing identical constant
  LLVM::GlobalOp found;
  moduleOp->walk([&](LLVM::GlobalOp global) {
    if (global.getConstant() &&
        global.getLinkage() == LLVM::linkage::Linkage::Private &&
        global.getValue() == valueAttr &&
        global.getGlobalType() == globalType) {
      found = global;
      return WalkResult::interrupt();
    }
    return WalkResult::skip();
  });
  if (found) {
    return found;
  }
  // Otherwise create one.

  auto twine = llvm::Twine("const", value).str();
  // str is reused to store the name
  str = getUniqueFunctionName(moduleOp, globalName.value_or(twine));

  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  return LLVM::GlobalOp::create(builder, loc, globalType,
                                /*isConstant=*/true, LLVM::Linkage::Private,
                                builder.getStringAttr(std::move(str)),
                                valueAttr);
}

static Value reifyAsString(ImplicitLocOpBuilder &builder, ModuleOp container,
                           StringRef value, StringRef nameHint) {
  LLVM::GlobalOp global =
      declareStringConstant(container, builder.getLoc(), value, true, nameHint);
  return LLVM::AddressOfOp::create(builder, global);
}

// Name of the discardable string attribute an upmem.scatter/gather/broadcast/
// scatter_blocks op may carry to label its transfer's stats rows (see
// timers.h/upmemrt_record_scatter's `tag` parameter). Absent means untagged.
constexpr StringLiteral kTimingTagAttrName = "upmem.timing_tag";

/// Reifies the op's `upmem.timing_tag` attribute (if present) as a string
/// constant, or a null pointer otherwise, for use as the `tag` argument of
/// the runtime transfer functions.
static Value reifyTimingTag(ImplicitLocOpBuilder &builder, ModuleOp container,
                            Operation *op) {
  if (auto tagAttr = op->getAttrOfType<StringAttr>(kTimingTagAttrName))
    return reifyAsString(builder, container, tagAttr.getValue(), "timing_tag");
  return LLVM::ZeroOp::create(builder, untypedPtrType(builder.getContext()));
}

/// Composes `inflateMap.compose(map)`'s results with `bufferTy`'s layout to
/// produce a single result expressing a byte offset into `bufferTy`, and
/// converts the (element) result of that composition to bytes. Shared tail of
/// linearizeAffineMap and linearizeAffineMapForTasklets.
static FailureOr<AffineMap>
composeWithBufferLayoutToBytes(AffineMap map, AffineMap inflateMap,
                               MemRefType bufferTy) {
  auto ctx = bufferTy.getContext();
  auto outputShape = bufferTy.getShape();
  auto layoutMap = bufferTy.getLayout().getAffineMap();
  if (isa<StridedLayoutAttr>(bufferTy.getLayout())) {
    // Replace offsets with 0 to delete the symbols.
    // Offset is calculated outside of the affine map.
    layoutMap = layoutMap.replaceDimsAndSymbols(
        {}, {getAffineConstantExpr(0, ctx)}, layoutMap.getNumDims(), 0);
  } else if (bufferTy.getLayout().isIdentity()) {
    auto linearIndex = mlir::linearizeIndices(ctx, outputShape);
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
  LLVM_DEBUG(llvm::errs() << "- before simplification " << result.getAffineMap()
                          << '\n');
  result.simplify();
  LLVM_DEBUG(llvm::errs() << "- after simplification " << result.getAffineMap()
                          << '\n');

  assert(result.getNumResults() == 1);

  // last step is making sure this map operates on bytes and not on elements
  auto resExpr = result.getResult(0);
  result.setResult(0, resExpr * (bufferTy.getElementTypeBitWidth() / 8));
  result.simplify();
  LLVM_DEBUG(llvm::errs() << "- result " << result.getAffineMap() << '\n');
  return success(result.getAffineMap());
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
  mlir::structureIndex(getAffineDimExpr(0, ctx), inputShape, inflatedIndices);
  AffineMap inflateMap = AffineMap::get(1, 0, inflatedIndices, ctx);

  auto result = composeWithBufferLayoutToBytes(map, inflateMap, bufferTy);
  if (failed(result))
    return failure();
  assert(result->getNumDims() == 1);
  return result;
}

/// Linearize the (rank, dpu, tasklet) scatter map used by the UPMEM SDK
/// scatter transfer API form of upmem.scatter. Unlike linearizeAffineMap,
/// the resulting function of two arguments (dpu index, tasklet index) is not
/// further inflated on the tasklet dim: it is passed straight through to
/// `map`, since the runtime calls it once per (dpu, tasklet) pair (see
/// upmemrt_dpu_scatter_to_tasklets / get_block_func_t).
static FailureOr<AffineMap>
linearizeAffineMapForTasklets(AffineMap map, ArrayRef<int64_t> dpuShape,
                              MemRefType bufferTy) {
  auto ctx = map.getContext();
  SmallVector<AffineExpr> inflatedIndices;
  mlir::structureIndex(getAffineDimExpr(0, ctx), dpuShape, inflatedIndices);
  inflatedIndices.push_back(getAffineDimExpr(1, ctx));
  AffineMap inflateMap = AffineMap::get(2, 0, inflatedIndices, ctx);

  auto result = composeWithBufferLayoutToBytes(map, inflateMap, bufferTy);
  if (failed(result))
    return failure();
  assert(result->getNumDims() == 2);
  return result;
}

/*
size_t upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t element_size, size_t num_elements,
                           size_t num_elements_per_tasklet, size_t copy_bytes,
                           char* buffer_id,
                           size_t (*base_offset)(size_t),
                           const char *tag);
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
      {ptrTy, ptrTy, sizeTy, sizeTy, sizeTy, sizeTy, ptrTy, funPtrTy, ptrTy},
      LLVM::LLVMVoidType::get(ctx));
}

/*
void upmemrt_dpu_scatter_blocks(struct dpu_set_t *dpu_set,
                                void *host_buffer, size_t element_size,
                                size_t num_blocks,
                                size_t block_num_elements,
                                const char *buffer_id,
                                size_t (*base_offset)(size_t, size_t),
                                const char *tag);
-- and upmemrt_dpu_gather_blocks, with the same signature.
*/
static FailureOr<LLVM::LLVMFuncOp>
getBlockTransferFunc(OpBuilder &rewriter, ModuleOp moduleOp,
                     LLVMTypeConverter const *tyConverter, StringRef name) {
  auto ctx = moduleOp->getContext();
  auto ptrTy = untypedPtrType(ctx);
  auto sizeTy = tyConverter->getIndexType();
  auto funPtrTy = functionPtrTy(sizeTy, {sizeTy, sizeTy});
  return LLVM::lookupOrCreateFn(
      rewriter, moduleOp, name,
      {ptrTy, ptrTy, sizeTy, sizeTy, sizeTy, ptrTy, funPtrTy, ptrTy},
      LLVM::LLVMVoidType::get(ctx));
}

/*
void upmemrt_dpu_broadcast(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t copy_bytes, const char *buffer_id,
                           const char *tag);
*/
static FailureOr<LLVM::LLVMFuncOp>
getBroadcastFunc(OpBuilder &rewriter, ModuleOp moduleOp,
                 LLVMTypeConverter const *tyConverter) {
  auto ctx = moduleOp->getContext();
  auto ptrTy = untypedPtrType(ctx);
  auto sizeTy = tyConverter->getIndexType();
  return LLVM::lookupOrCreateFn(rewriter, moduleOp, "upmemrt_dpu_broadcast",
                                {ptrTy, ptrTy, sizeTy, ptrTy, ptrTy},
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

// Set by ConvertUPMEMToLLVMPass on each upmem.alloc_dpus before conversion
// starts (see computeMaxBlocksPerDpu): the largest numBlocksPerDpu among the
// upmem.scatter_blocks/gather_blocks ops using that hierarchy, or absent if
// none use it. AllocDPUOpToFuncCallLowering reads it back to size the UPMEM
// SDK's sgXferMaxBlocksPerDpu profile option -- this can't be recomputed from
// inside the conversion pattern itself, since by the time an individual op is
// legalized, its users may already have been converted away.
constexpr StringLiteral kMaxBlocksPerDpuAttrName = "upmem.max_blocks_per_dpu";

struct AllocDPUOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::AllocDPUsOp> {
public:
  explicit AllocDPUOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::AllocDPUsOp>(lowering) {}

  FailureOr<Value>
  createConstantForDpuProgramName(ConversionPatternRewriter &rewriter,
                                  upmem::AllocDPUsOp op) const {
    auto leafName = op.getDpuProgramRef().getLeafReference().getValue();

    LLVM::GlobalOp constant =
        declareStringConstant(op->getParentOfType<ModuleOp>(), op->getLoc(),
                              leafName, true, "dpu_program");
    Value result = LLVM::AddressOfOp::create(rewriter, op->getLoc(), constant);
    return success(result);
  }

  LogicalResult
  matchAndRewrite(upmem::AllocDPUsOp op, typename upmem::AllocDPUsOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const DeviceHierarchyType hierarchyShape = op.getResult().getType();
    // The SDK allocates DPU counts, not rank layouts: one number suffices.
    const Value dpuCount = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getI32IntegerAttr(hierarchyShape.getNumDpus()));

    const auto maybeFailed = createConstantForDpuProgramName(rewriter, op);
    if (failed(maybeFailed))
      return failure();
    const Value dpuProgramPath = *maybeFailed;

    // Computed by ConvertUPMEMToLLVMPass before conversion started (see
    // kMaxBlocksPerDpuAttrName): 0 if no upmem.scatter using this hierarchy
    // needs the UPMEM SDK's scatter transfer API.
    int64_t maxBlocksPerDpu = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>(kMaxBlocksPerDpuAttrName))
      maxBlocksPerDpu = attr.getInt();
    Type sizeTy = getTypeConverter()->getIndexType();
    Value maxBlocksPerDpuVal = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), sizeTy,
        rewriter.getIntegerAttr(sizeTy, maxBlocksPerDpu));

    // struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_dpus,
    //     const char *dpu_binary_path, size_t max_blocks_per_dpu);
    Type resultType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    auto funcOp = appendOrGetFuncOp(
        rewriter, "upmemrt_dpu_alloc", resultType,
        {rewriter.getI32Type(), untypedPtrType(getContext()), sizeTy}, op);

    if (llvm::failed(funcOp))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, *funcOp, ValueRange{dpuCount, dpuProgramPath, maxBlocksPerDpuVal});
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
  auto affineMapFun =
      LLVM::LLVMFuncOp::create(rewriter, rewriter.getStringAttr(funName),
                               affineFunTy, LLVM::Linkage::Private);

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
    LLVM::ReturnOp::create(rewriter, ValueRange{result});
    return affineMapFun;
  }
  return failure();
}

/// Same as outlineAffineMap, but for the (rank, dpu, tasklet) upmem.scatter
/// form: emits a function of two arguments (dpu index, tasklet index)
/// matching the runtime's base_offset(size_t, size_t) callback (see
/// upmemrt_dpu_scatter_to_tasklets).
static FailureOr<LLVM::LLVMFuncOp> outlineAffineMapForTasklets(
    ImplicitLocOpBuilder &rewriter, LLVMTypeConverter const *tyConverter,
    ModuleOp moduleOp, AffineMap map, DeviceHierarchyType hierarchyTy,
    MemRefType bufferTy) {

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto sizeTy =
      tyConverter->convertType(IndexType::get(moduleOp->getContext()));
  auto shape = hierarchyTy.getWgShape();
  auto linearMap = linearizeAffineMapForTasklets(
      map, ArrayRef<int64_t>(shape).drop_back(), bufferTy);
  if (failed(linearMap)) {
    emitError(rewriter.getLoc(), "Unsupported layout map for ") << bufferTy;
    return failure();
  }

  auto affineFunTy = LLVM::LLVMFunctionType::get(sizeTy, {sizeTy, sizeTy});
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
  auto funName = getUniqueFunctionName(moduleOp, "sg_scatter_map");
  rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  auto affineMapFun =
      LLVM::LLVMFuncOp::create(rewriter, rewriter.getStringAttr(funName),
                               affineFunTy, LLVM::Linkage::Private);

  // to find it later
  affineMapFun->setAttr("upmem.generated_from", AffineMapAttr::get(*linearMap));

  rewriter = ImplicitLocOpBuilder::atBlockBegin(
      rewriter.getLoc(), affineMapFun.addEntryBlock(rewriter));
  Value arg0 = affineMapFun.getArgument(0);
  Value arg1 = affineMapFun.getArgument(1);
  // affine expects to deal with index type only
  arg0 = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                              rewriter.getIndexType(), arg0);
  arg1 = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                              rewriter.getIndexType(), arg1);

  if (auto resOpt = affine::expandAffineMap(
          rewriter, rewriter.getLoc(), *linearMap, ValueRange{arg0, arg1})) {
    auto result = (*resOpt)[0];
    result = createOrFoldUnrealizedConversionCast(rewriter.getLoc(), rewriter,
                                                  sizeTy, result);
    LLVM::ReturnOp::create(rewriter, ValueRange{result});
    return affineMapFun;
  }
  return failure();
}

/// Computes the untyped pointer to the start of a lowered memref operand, and
/// the buffer-name string constant used by both the block and tasklet
/// scatter/gather lowerings.
static FailureOr<std::pair<Value, Value>> computeBareHostBufAndBufferId(
    Operation *op, Value hostBufferAdaptor, Type hostBufferElementTy,
    StringRef dpuBufRef, ImplicitLocOpBuilder &rewriter,
    ConversionPatternRewriter &rewriter0, ModuleOp moduleOp) {
  auto loc = op->getLoc();
  Value bareHostBuf = hostBufferAdaptor;
  if (isa<LLVM::LLVMStructType>(hostBufferAdaptor.getType())) {
    // Here we compute the pointer to the start of the memref
    // converted memref
    Value basePtr =
        LLVM::ExtractValueOp::create(rewriter0, loc, hostBufferAdaptor, 1);
    Value offset =
        LLVM::ExtractValueOp::create(rewriter0, loc, hostBufferAdaptor, 2);
    // need to do our own pointer arithmetic here
    bareHostBuf =
        LLVM::GEPOp::create(rewriter0, loc, basePtr.getType(),
                            hostBufferElementTy, basePtr, ValueRange{offset});
  } else {
    return emitError(loc, "Unhandled buffer type: ")
           << hostBufferAdaptor.getType();
  }
  Value bufferId = reifyAsString(rewriter, moduleOp, dpuBufRef, "buffer_name");
  return std::make_pair(bareHostBuf, bufferId);
}

template <class Op>
static LogicalResult lowerBlockTransfer(Op op, typename Op::Adaptor adaptor,
                                        LLVMTypeConverter const *tyConverter,
                                        ConversionPatternRewriter &rewriter0,
                                        bool isGather) {
  auto loc = op->getLoc();
  ImplicitLocOpBuilder rewriter(loc, rewriter0);
  auto moduleOp = op->template getParentOfType<ModuleOp>();

  auto bufsOrFailure = computeBareHostBufAndBufferId(
      op, adaptor.getHostBuffer(),
      op.getHostBuffer().getType().getElementType(), op.getDpuBufRef(),
      rewriter, rewriter0, moduleOp);
  if (failed(bufsOrFailure))
    return failure();
  auto [bareHostBuf, bufferId] = *bufsOrFailure;

  // Use the UPMEM SDK's scatter/gather transfer API (dpu_push_sg_xfer) so each
  // block can sit at a location in the host buffer that isn't contiguous with
  // the other blocks'.
  auto affineMapFunOpt = outlineAffineMapForTasklets(
      rewriter, tyConverter, moduleOp, op.getScatterMap(),
      op.getHierarchy().getType(), op.getHostBuffer().getType());
  if (failed(affineMapFunOpt))
    return emitError(loc, "Cannot emit affine map");

  auto runtimeFun = getBlockTransferFunc(
      rewriter, moduleOp, tyConverter,
      isGather ? "upmemrt_dpu_gather_blocks" : "upmemrt_dpu_scatter_blocks");
  if (llvm::failed(runtimeFun))
    return failure();
  auto funPtrOp = LLVM::AddressOfOp::create(rewriter0, loc, *affineMapFunOpt);
  Value tag = reifyTimingTag(rewriter, moduleOp, op);

  // Size of elements in bytes
  const size_t elementSize =
      op.getHostBuffer().getType().getElementTypeBitWidth() / 8;
  // Number of blocks per DPU. This is independent of the hierarchy's
  // declared tasklet count -- blocks are just UPMEM SDK transfer units and
  // need not correspond 1:1 to actual DPU tasklets (see the op
  // description) -- so it must come from the required numBlocksPerDpu
  // attribute, not from op.getHierarchy().
  const size_t numBlocksPerDpu = op.getNumBlocksPerDpu();
  // transferCount is the size of a single block, in elements (see the op
  // description)
  const size_t blockNumElements = op.getTransferCount();

  /*
  void upmemrt_dpu_scatter_blocks(struct dpu_set_t *dpu_set,
                                  void *host_buffer,
                                  size_t element_size,
                                  size_t num_blocks,
                                  size_t block_num_elements,
                                  const char *buffer_id,
                                  size_t (*base_offset)(size_t, size_t),
                                  const char *tag)
  */
  LLVM::CallOp::create(
      rewriter0, loc, *runtimeFun,
      ValueRange{adaptor.getHierarchy(), bareHostBuf,
                 reifyAsIndex(rewriter, tyConverter, elementSize),
                 reifyAsIndex(rewriter, tyConverter, numBlocksPerDpu),
                 reifyAsIndex(rewriter, tyConverter, blockNumElements),
                 bufferId, funPtrOp.getRes(), tag});

  rewriter0.eraseOp(op);
  return success();
}

static LogicalResult lowerBroadcast(upmem::BroadcastOp op,
                                    upmem::BroadcastOp::Adaptor adaptor,
                                    LLVMTypeConverter const *tyConverter,
                                    ConversionPatternRewriter &rewriter0) {
  auto loc = op->getLoc();
  ImplicitLocOpBuilder rewriter(loc, rewriter0);
  auto moduleOp = op->getParentOfType<ModuleOp>();

  auto bufsOrFailure = computeBareHostBufAndBufferId(
      op, adaptor.getHostBuffer(),
      op.getHostBuffer().getType().getElementType(), op.getDpuBufRef(),
      rewriter, rewriter0, moduleOp);
  if (failed(bufsOrFailure))
    return failure();
  auto [bareHostBuf, bufferId] = *bufsOrFailure;

  auto runtimeFun = getBroadcastFunc(rewriter, moduleOp, tyConverter);
  if (llvm::failed(runtimeFun))
    return failure();
  Value tag = reifyTimingTag(rewriter, moduleOp, op);

  // Transfer size must be 8-byte aligned, like the classic scatter/gather
  // block form.
  auto numBytesCopied = op.getDpuBufferSizeInBytes();
  numBytesCopied = llvm::alignTo(numBytesCopied, 8);

  /*
  void upmemrt_dpu_broadcast(struct dpu_set_t *dpu_set, void *host_buffer,
                             size_t copy_bytes, const char *buffer_id,
                             const char *tag)
  */
  LLVM::CallOp::create(
      rewriter0, loc, *runtimeFun,
      ValueRange{adaptor.getHierarchy(), bareHostBuf,
                 reifyAsIndex(rewriter, tyConverter, numBytesCopied), bufferId,
                 tag});

  rewriter0.eraseOp(op);
  return success();
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

  auto bufsOrFailure = computeBareHostBufAndBufferId(
      op, adaptor.getHostBuffer(),
      op.getHostBuffer().getType().getElementType(), op.getDpuBufRef(),
      rewriter, rewriter0, moduleOp);
  if (failed(bufsOrFailure))
    return failure();
  auto [bareHostBuf, bufferId] = *bufsOrFailure;

  auto affineMapFunOpt = outlineAffineMap(
      rewriter, tyConverter, moduleOp, op.getScatterMap(),
      op.getHierarchy().getType(), op.getHostBuffer().getType());
  if (failed(affineMapFunOpt)) {
    return emitError(op->getLoc(), "Cannot emit affine map");
  }

  auto runtimeScatterFun = getScatterOrGatherFunc(
      rewriter, moduleOp, tyConverter,
      isGather ? "upmemrt_dpu_gather" : "upmemrt_dpu_scatter");

  if (llvm::failed(runtimeScatterFun))
    return failure();
  auto funPtrOp = LLVM::AddressOfOp::create(rewriter0, loc, *affineMapFunOpt);
  Value tag = reifyTimingTag(rewriter, moduleOp, op);
  // Transfer count must be 8-byte aligned
  auto numBytesCopied = op.getDpuBufferSizeInBytes();
  numBytesCopied = llvm::alignTo(numBytesCopied, 8);

  // Size of elements in bytes
  const size_t elementSize =
      op.getHostBuffer().getType().getElementTypeBitWidth() / 8;
  // Total number of concurrent tasklets in the array
  const size_t numTasklets = op.getHierarchy().getType().getNumElements();
  // Total number of elements in the containing buffer, used for in-bounds check
  const size_t numElements =
      computeProduct(op.getHostBuffer().getType().getShape());
  // Number of elements for each tasklet
  const size_t numElementsPerTasklet = numElements / numTasklets;

  /*
  void upmemrt_dpu_scatter( struct dpu_set_t *dpu_set,
                            void *hostBuffer,
                            size_t element_size,
                            size_t num_elements,
                            size_t num_elements_per_tasklet,
                            size_t copy_bytes,
                            const char *bufId,
                            size_t (*base_offset)(size_t),
                            const char *tag)
  */
  LLVM::CallOp::create(
      rewriter0, loc, *runtimeScatterFun,
      ValueRange{adaptor.getHierarchy(), bareHostBuf,
                 reifyAsIndex(rewriter, tyConverter, elementSize),
                 reifyAsIndex(rewriter, tyConverter, numElements),
                 reifyAsIndex(rewriter, tyConverter, numElementsPerTasklet),
                 reifyAsIndex(rewriter, tyConverter, numBytesCopied), bufferId,
                 funPtrOp.getRes(), tag});

  rewriter0.eraseOp(op);
  return success();
}

struct ScatterOnArrayOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::ScatterOnArrayOp> {
public:
  explicit ScatterOnArrayOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::ScatterOnArrayOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::ScatterOnArrayOp op,
                  typename upmem::ScatterOnArrayOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerScatterOrGather(op, adaptor, getTypeConverter(), rewriter0,
                                false);
  }
};

struct GatherFromArrayOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::GatherFromArrayOp> {
public:
  explicit GatherFromArrayOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::GatherFromArrayOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::GatherFromArrayOp op,
                  typename upmem::GatherFromArrayOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerScatterOrGather(op, adaptor, getTypeConverter(), rewriter0,
                                true);
  }
};

struct ScatterBlocksOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::ScatterBlocksOp> {
public:
  explicit ScatterBlocksOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::ScatterBlocksOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::ScatterBlocksOp op,
                  typename upmem::ScatterBlocksOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerBlockTransfer(op, adaptor, getTypeConverter(), rewriter0,
                              false);
  }
};

struct GatherBlocksOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::GatherBlocksOp> {
public:
  explicit GatherBlocksOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::GatherBlocksOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::GatherBlocksOp op,
                  typename upmem::GatherBlocksOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerBlockTransfer(op, adaptor, getTypeConverter(), rewriter0, true);
  }
};

struct BroadcastOpToFuncCallLowering
    : public ConvertOpToLLVMPattern<upmem::BroadcastOp> {
public:
  explicit BroadcastOpToFuncCallLowering(LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<upmem::BroadcastOp>(lowering) {}

  LogicalResult
  matchAndRewrite(upmem::BroadcastOp op,
                  typename upmem::BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter0) const override {
    return lowerBroadcast(op, adaptor, getTypeConverter(), rewriter0);
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
  patterns.add<ScatterOnArrayOpToFuncCallLowering>(typeConverter);
  patterns.add<ScatterBlocksOpToFuncCallLowering>(typeConverter);
  patterns.add<BroadcastOpToFuncCallLowering>(typeConverter);
  patterns.add<GatherFromArrayOpToFuncCallLowering>(typeConverter);
  patterns.add<GatherBlocksOpToFuncCallLowering>(typeConverter);
  patterns.add<WaitForOpToFuncCallLowering>(typeConverter);
  patterns.add<FreeDPUsOpToFuncCallLowering>(typeConverter);
  patterns.add<EraseDpuProgram>(typeConverter);
}

struct ConvertUPMEMToLLVMPass
    : public impl::ConvertUPMEMToLLVMPassBase<ConvertUPMEMToLLVMPass> {
  void runOnOperation() final {
    // Stash, on each upmem.alloc_dpus, the largest numBlocksPerDpu among the
    // upmem.scatter_blocks/gather_blocks ops using it (see
    // kMaxBlocksPerDpuAttrName). This must happen as a plain IR walk before
    // conversion starts: once conversion is under way, a transfer op may
    // already have been legalized (and erased) by the time alloc_dpus's own
    // pattern runs, so it can no longer be found by scanning the hierarchy
    // value's users from inside a pattern.
    getOperation()->walk([&](upmem::AllocDPUsOp allocOp) {
      uint64_t maxBlocks = 0;
      for (Operation *user : allocOp.getResult().getUsers())
        maxBlocks = std::max(
            maxBlocks, llvm::TypeSwitch<Operation *, uint64_t>(user)
                           .Case<upmem::ScatterBlocksOp, upmem::GatherBlocksOp>(
                               [](auto op) { return op.getNumBlocksPerDpu(); })
                           .Default(uint64_t{0}));
      if (maxBlocks > 0)
        allocOp->setAttr(
            kMaxBlocksPerDpuAttrName,
            IntegerAttr::get(IntegerType::get(&getContext(), 64), maxBlocks));
    });

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
      //   return bufferization::ToMemrefOp::create(builder, loc, type, inputs)
      //       .getResult();
      // }
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
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
