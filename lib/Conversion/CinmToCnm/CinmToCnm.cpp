
#include "cinm-mlir/Conversion/CinmPasses.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmBase.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmInterfaces.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Utils/CinmUtils.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>

namespace mlir {
#define GEN_PASS_DEF_CONVERTTILEDCINMTOCNM
#include <cinm-mlir/Conversion/CinmPasses.h.inc>
} // namespace mlir
using namespace mlir;

namespace {

/// Resolve the pass's `cnm-buffer-level` option against the accelerator's
/// platform, yielding the attribute that goes in the `cnm.buffer` type's level
/// field and in the memory space of the launch body's memrefs.
///
/// An empty name yields a null level, which is what every buffer got before
/// this option existed: the backend conversion then picks the staging itself.
FailureOr<cinm::CinmLevelAttrInterface>
resolveBufferLevel(StringRef levelName, cnm::CnmAcceleratorAttrInterface acc,
                   Operation *op) {
  if (levelName.empty())
    return cinm::CinmLevelAttrInterface{};

  auto platform = acc.getPlatform();
  if (!platform)
    return op->emitOpError("cannot resolve memory level '")
           << levelName << "': the accelerator declares no platform";

  cinm::CinmLevelDefAttr def = platform.getLevel(levelName);
  if (!def) {
    auto diag = op->emitOpError("unknown memory level '")
                << levelName << "' for platform '" << platform.getName()
                << "'; known levels are ";
    llvm::interleaveComma(platform.getLevels(), diag,
                          [&](cinm::CinmLevelDefAttr l) {
                            diag << "'" << l.getName().getValue() << "'";
                          });
    return diag;
  }

  cinm::CinmLevelAttrInterface space = platform.getMemrefMemspace(def);
  if (!space)
    return op->emitOpError("platform '")
           << platform.getName()
           << "' does not provide a memref memory space for level '"
           << levelName << "'";
  return space;
}

LogicalResult
computeShapeOfTensors(Location loc, llvm::ArrayRef<int64_t> shape,
                      cnm::WorkgroupType wgTy, int64_t maxBlockSize,
                      // if empty then all dims are parallel
                      // otherwise those dims are reductions. They are
                      // used to select the size of the buffer. The rest of
                      // the dimensions are used to create a scattermap
                      llvm::ArrayRef<int64_t> reductionDims,
                      AffineMap &scatterMap,
                      llvm::SmallVectorImpl<int64_t> &shapeOfBuffer,

                      std::optional<llvm::SmallVector<int64_t>> &reshapeInputTo,
                      bool scatterScalar) {
  auto wgShape = wgTy.getShape();

  auto numWgItems =
      std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
  auto numBufItems =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());

  llvm::SmallVector<int64_t> parallelDims;
  int64_t numParallelElts = 1;
  int64_t numReductionElts = 1;

  for (size_t i = 0, j = 0; i < shape.size(); i++) {
    if (j >= reductionDims.size() ||
        i != static_cast<size_t>(reductionDims[j])) {
      // this is a parallel dim
      parallelDims.push_back(shape[i]);
      numParallelElts *= shape[i];
      if (numReductionElts > 1) {
        // This means the reduction dims are not all at the end
        // Todo revisit, needs a transpose
        return failure();
      }
    } else {
      // this is a reduction dim
      j++;
      shapeOfBuffer.push_back(shape[i]);
      numReductionElts *= shape[i];
    }
  }

  if (numReductionElts > maxBlockSize) {
    emitError(loc, "can't compute shape of tensors: numReductionElts (" +
                       std::to_string(numReductionElts) + ") > maxBlockSize (" +
                       std::to_string(maxBlockSize) + ")");
    return failure();
  }

  // Now we support 3 cases: either
  // 0. scattering a single element, or broadcasting
  if (scatterScalar || numReductionElts == numBufItems) {
    // The buffer may end up with fewer dims than the (possibly wrapped)
    // input tensor, e.g. when scatterScalar wraps a bare scalar into
    // tensor<numTasklets x ElementTy> but the buffer stays a plain scalar
    // (shapeOfBuffer empty), or when leading unit parallel dims aren't
    // classified as reduction dims. The scatter map must produce one result
    // per truncated dim; since every value being scattered is identical
    // (broadcast to all workgroup members), a constant 0 index is always
    // valid.
    size_t numTruncatedDims = shape.size() - shapeOfBuffer.size();
    scatterMap = AffineMap::get(
        wgShape.size(), 0,
        SmallVector<AffineExpr>(numTruncatedDims,
                                getAffineConstantExpr(0, wgTy.getContext())),
        wgTy.getContext());
    return success();
  }

  // 1. tensor has shape of WG
  if (parallelDims == wgShape) {
    scatterMap =
        AffineMap::getMultiDimIdentityMap(wgShape.size(), wgTy.getContext());
    return success();
  }

  // or 2. numParallelItems == k * numWgItems
  if (numParallelElts % numWgItems != 0) {
    return emitError(loc, "can't compute shape of tensors: numParallelElts (" +
                              std::to_string(numParallelElts) +
                              ") % numWgItems (" + std::to_string(numWgItems) +
                              ") != 0");
  }

  // Say we have p parallel dims, m working group dimensions, and r reduction
  // dimensions. Affine map has the form

  // In case we have 0 reduction dimensions:
  // Affine map has the form
  //   F: (W1, ..., Wm) -> (T1, ..., Tp)
  // For all w=(w1, ..., wm), F(w) must be in range,
  // that is to say, for each dim i, 0 <= F(w)_i < |T_i|,
  // and

  // For example consider
  //    gemm: (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
  //    WG: <1x128>
  //    bufsize: 32
  //
  // The first tensor is <1x32>. Num par elements is 1. Broadcast:
  //    (w0, w1) -> (0, 0)
  // The second tensor is <32x128>. Num par elements is 128 = |WG|.
  //    (w0, w1) -> (0, lin(w0, w1))
  // Here we see that the elements are not contiguous though.
  //    Transpose: <32x128> -> <128x32>
  //    (w0, w1) -> (lin(w0, w1), 0)
  // The output tensor is <1x128>. It matches WG shape:
  //    (w0, w1) -> (w0, w1)

  // IMPLNOTE: the transpose is not done in this routine. It must be
  //   done in the caller. This rountine will fail if the reduction
  //   dims are not at the end.

  // Another example. Elementwise operators have no reduction dims.
  //    add: tensor<16384xi32>
  //    WG: <8x128>
  //    bufsize: 16
  //
  // Both input tensors and output have same shape.
  // Num par elements is 16384. Break this down into
  // 16384/16=1024 buffers of 16 elements and expand shape:
  //   T': <16384> -> <1024x16>
  // Then scatter map is
  //   (w0, w1) -> (lin(w0, w1), 0)

  // What if I want to do this without expand_shape?
  // (w0, w1) -> (lin(w0, w1) * 16)

  /*
  Summary:
  We need to support the following cases:
  - Tensor has exactly 1 parallel element. Then broadcast. (only if it is an
  input)
  - Tensor is flat and needs to be chunked.
  - Tensor has parallel elts = |WG| but chunks are not contiguous.
    Need a linalg.transpose.

  */

  // if k = 1:
  //     (t, R1,..., Rn) -> (W1,...,Wm,R1,...,Rn)
  // if k * numReductionItems <= maxBlockSize:
  //     (t, R1,..., Rn) -> (W1,...,Wm,ki,R1,...,Rn)
  //    where ki ranges from 0 to k
  //

  int64_t k = numParallelElts / numWgItems;
  if (k != 1) {
    if (k * numReductionElts <= maxBlockSize) {
      // In this branch we handle the case where there are no reduction
      // dimensions, in that case we do some parallel work on the DPU, and
      // therefore push these extra parallel elts into the buffer.
      shapeOfBuffer.insert(shapeOfBuffer.begin(), k);

      // expand dimension
      int trailing = 1;
      int numFlattened = 0;
      for (auto it = parallelDims.rbegin(); it != parallelDims.rend(); it++) {
        trailing *= *it;
        numFlattened++;
        if (trailing >= k && trailing % k == 0) {
          break;
        }
      }
      SmallVector<int64_t> newShape;
      for (auto [i, dim] : llvm::enumerate(parallelDims)) {
        if (i < parallelDims.size() - numFlattened) {
          // not flattened
          newShape.push_back(dim);
        } else {
          newShape.push_back(trailing / k);
          newShape.push_back(k);
          break;
        }
      }
      parallelDims = newShape; // our parallel dims have changed
      parallelDims.pop_back();

      for (auto dim : reductionDims)
        newShape.push_back(shape[dim]);

      reshapeInputTo = std::make_optional(std::move(newShape));

    } else {
      // probably the op hasn't been tiled properly
      emitError(loc, "can't compute shape of tensors: k (" + std::to_string(k) +
                         ") * numReductionElts (" +
                         std::to_string(numReductionElts) +
                         ") > "
                         "maxBlockSize (" +
                         std::to_string(maxBlockSize) + ")");
      return failure();
    }
  }

  AffineExpr index = mlir::linearizeIndices(wgTy.getContext(), wgShape);

  llvm::SmallVector<AffineExpr> scatterResults;
  scatterResults.reserve(parallelDims.size());
  mlir::structureIndex(index, parallelDims, scatterResults);

  scatterMap =
      AffineMap::get(wgShape.size(), 0, scatterResults, wgTy.getContext());
  return success();
}

LogicalResult convertInputIntoAlloc(Value &inputBuf, Value workGroup,
                                    cnm::WorkgroupType wgTy,
                                    int64_t maxBlockSizeBytes,
                                    ArrayRef<int64_t> reduceDims,
                                    cinm::CinmLevelAttrInterface level,
                                    AffineMap &scatterMap, Value &result,
                                    ImplicitLocOpBuilder &rewriter) {
  // For each input of the reduce, we need to

  // convert single element to tensor<numTasklets x leafSize x ElementTy>
  bool scatterScalar = false;
  if (!llvm::isa<ShapedType>(inputBuf.getType())) {
    scatterScalar = true;
    inputBuf = tensor::FromElementsOp::create(
        rewriter,
        RankedTensorType::get({wgTy.getShape()[2]}, inputBuf.getType()),
        SmallVector<Value>(wgTy.getShape()[2], inputBuf));
  }

  auto inputType = cast<ShapedType>(inputBuf.getType());

  llvm::SmallVector<int64_t, 1> shapeOfBuffer;
  std::optional<SmallVector<int64_t>> reshapeInto;
  auto maxBlockSizeItems =
      maxBlockSizeBytes * 8 / inputType.getElementTypeBitWidth();
  if (computeShapeOfTensors(inputBuf.getLoc(), inputType.getShape(), wgTy,
                            maxBlockSizeItems, reduceDims, scatterMap,
                            shapeOfBuffer, reshapeInto, scatterScalar)
          .failed())
    return failure();

  // If the tensor was just allocated it is assumed empty, therefore we don't
  // need to scatter as its contents are undefined.
  const bool needScatter = !inputBuf.getDefiningOp() ||
                           (!isa<tensor::EmptyOp>(inputBuf.getDefiningOp()) &&
                            !isa<memref::AllocOp>(inputBuf.getDefiningOp()));
  if (reshapeInto) {
    inputBuf = mlir::reshapeStatic(rewriter, rewriter.getLoc(), inputBuf,
                                   cast<ShapedType>(inputType), *reshapeInto);
  }

  // Allocate a cinm buffer
  cnm::BufferType bufTy =
      cnm::BufferType::get(shapeOfBuffer, inputType.getElementType(),
                           wgTy.getAccelerator(), level);

  Value alloc = cnm::AllocOp::create(rewriter, bufTy, workGroup);

  // Scatter into buffer
  if (needScatter) {
    cnm::ScatterOp::create(rewriter, inputBuf, alloc, workGroup, scatterMap);
  }
  result = alloc;

  return success();
}

cnm::LaunchOp createLaunchOp(
    ImplicitLocOpBuilder &builder, Value workgroup, ValueRange inputs,
    ValueRange outputs,
    function_ref<void(ImplicitLocOpBuilder &, ValueRange, ValueRange)>
        createCnmLaunchBlock) {

  cnm::LaunchOp launchOp =
      cnm::LaunchOp::create(builder, workgroup, inputs, outputs);

  {
    auto &launchBlock = launchOp.getBody().emplaceBlock();
    // arguments are memrefs with same shape as inputs
    for (auto input : launchOp.getParams()) {
      if (auto inputTy = dyn_cast<cnm::BufferType>(input.getType())) {
        // The buffer's level becomes the memref's memory space -- this is what
        // LaunchOp::verify requires, and it is how the launch body learns
        // which memory it is computing on.
        auto mappedTy = MemRefType::get(inputTy.getShape(),
                                        inputTy.getElementType(),
                                        MemRefLayoutAttrInterface{},
                                        inputTy.getLevel());
        launchBlock.addArgument(mappedTy, input.getLoc());
      } else {
        launchBlock.addArgument(input.getType(), input.getLoc());
      }
    }

    auto args = launchBlock.getArguments();
    auto firstOutput = args.begin() + inputs.size();
    llvm::SmallVector<Value, 2> reduceInpts(args.begin(), firstOutput);
    llvm::SmallVector<Value, 1> reduceInits(firstOutput, args.end());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&launchBlock);
    createCnmLaunchBlock(builder, reduceInpts, reduceInits);
    cnm::TerminatorOp::create(builder);
  }
  return launchOp;
}

LogicalResult convertCinmToCnm(
    ImplicitLocOpBuilder builder, Operation *operation,
    TypedValue<cnm::WorkgroupType> workgroup,
    ArrayRef<ArrayRef<int64_t>> reductionDimensionsSorted, ValueRange operands,
    ValueRange outputInitializers,
    ValueRange /*optional elements*/ gatherBuffers, ValueRange results,
    cinm::CinmLevelAttrInterface level,
    llvm::SmallVectorImpl<Value> &resultValues,
    function_ref<void(ImplicitLocOpBuilder &, ValueRange, ValueRange)>
        createCnmLaunchBlock) {

  auto wgTy = workgroup.getType();

  llvm::SmallVector<Value, 3> launchInputs;
  llvm::SmallVector<Value, 3> launchOutputs;
  llvm::SmallVector<AffineMap, 3> gatherMaps;
  llvm::SmallVector<Type, 3> mappedArgTypes;

  auto cnmAccelerator = workgroup.getType().getAccelerator();
  // Is there a way to generically know and where to place a buffer in
  // accelerator memory?
  // TODO for now we assume all levels of the accelerator memory are used.
  //  In fact for upmem, using WRAM should be a conscious decision.
  //  It should be possible to express scattering from host memory to MRAM, then
  //  do another scatter for MRAM to WRAM. It sounds like we should split the
  //  CINM->CNM transformation into more steps, that would each be configurable.
  //  For instance, the current strategy is to use WRAM size to determine buffer
  //  sizes. Then when lowering from CNM to upmem another layer is added. In the
  //  future what we should be doing is let the target platform choose its own
  //  lowering strategy. That means there probably wouldn't be a simple CINM ->
  //  CNM pass. We could add some methods to the CnmAcceleratorAttrInterface to
  //  support our current flow though. It seems what we need is:
  //  - How much memory can each leaf element use?
  //  - In what memory space? (string identifier)
  //  Using this data it is already possible to
  //  - Infer a mapping between parallel dimensions and leaf elements (scatter
  //  maps)
  //  - Produce a scatter/launch/gather program that targets the leaves
  //  There is not much

  // Could we have something like
  // scatter memref (host) onto MRAM
  // cnm.launch (%A, %B, %C) { // host
  // ^bb0(%a, %b, %c): // MRAM buffers
  //    Here we have another "accelerator" that allows scattering on
  //    %2 = cnm.workgroup #upmem.on_dpu<8 tasklets>
  //    %awram = cnm.alloc() for %2: !cnm.buffer<128xi32 on 8, "wram">
  //    %bwram = cnm.alloc() for %2: !cnm.buffer<128xi32 on 8, "wram">
  //    cnm.scatter %a into %awram[(tid) -> (tid)] of %2 :  // each tasklet gets
  //    its own buffer cnm.scatter %b into %bwram[(tid) -> ()] of %2 :     //
  //    all tasklets share the same buffer cnm.launch (%awram, %bwram) {
  //      .. kernel on wram
  //    }
  // }

  int maxBlockSizeBytes = cnmAccelerator.bufferSizeOfLeaf() / operands.size();

  builder.setInsertionPointAfter(operation);

  for (auto [input, redDims] : llvm::zip(operands, reductionDimensionsSorted)) {
    if (convertInputIntoAlloc(input, workgroup, wgTy, maxBlockSizeBytes,
                              redDims, level, gatherMaps.emplace_back(),
                              launchInputs.emplace_back(), builder)
            .failed()) {
      return failure();
    }
  }

  // output values, may have been reshaped
  llvm::SmallVector<Value, 1> reshapedOutputs;
  for (auto output : outputInitializers) {
    if (convertInputIntoAlloc(output, workgroup, wgTy, maxBlockSizeBytes, {},
                              level, gatherMaps.emplace_back(),
                              launchOutputs.emplace_back(), builder)
            .failed()) {
      return failure();
    }
    reshapedOutputs.push_back(output);
  }

  createLaunchOp(builder, workgroup, launchInputs, launchOutputs,
                 createCnmLaunchBlock);

  // gather the results (only the out buffers)

  // Gather tensor results
  for (auto [i, reshaped, cnmAlloc, gatherBuf] :
       llvm::enumerate(reshapedOutputs, launchOutputs, gatherBuffers)) {
    auto map = gatherMaps[launchInputs.size() + i];
    Value outBuf;
    if (isa<TensorType>(reshaped.getType())) {
      // The gather output must have the reshaped (post-convertInputIntoAlloc)
      // type so that buffer dims match what the scatter used.  gatherBuf may
      // carry the original un-reshaped shape and would fail verification.
      outBuf =
          tensor::EmptyOp::create(builder, reshaped.getType(), ValueRange{});
    } else if (gatherBuf) {
      outBuf = gatherBuf;
    } else {
      outBuf = reshaped;
    }
    auto res = cnm::GatherOp::create(builder, cnmAlloc, workgroup, map, outBuf);
    if (isa<TensorType>(reshaped.getType())) {
      auto correspondingResult = results[i];
      if (auto resultShapedTy =
              dyn_cast<ShapedType>(correspondingResult.getType())) {
        Value shapedBack = mlir::reshapeStatic(
            builder, builder.getLoc(),
            cast<TypedValue<ShapedType>>(res.getOutput()),
            resultShapedTy.getShape());
        // If an explicit destination was provided, tell the bufferizer that
        // the result should alias it so the copy can be folded away.
        if (gatherBuf && !matchPattern(gatherBuf, m_Constant()))
          shapedBack = bufferization::MaterializeInDestinationOp::create(
                           builder, shapedBack, gatherBuf)
                           .getResult();
        resultValues.push_back(shapedBack);
      } else {
        // The op's result is a scalar (e.g. cinm.op.reduce fully reducing
        // its input to a single value): `reshaped` was only wrapped into a
        // tensor to make it scatterable (see convertInputIntoAlloc's
        // scatterScalar handling), so unwrap it back to the scalar the op
        // actually returns.
        auto outTy = cast<ShapedType>(res.getOutput().getType());
        SmallVector<Value> indices(
            outTy.getRank(), arith::ConstantIndexOp::create(builder, 0));
        Value scalar =
            tensor::ExtractOp::create(builder, res.getOutput(), indices);
        resultValues.push_back(scalar);
      }
    }
  }

  cnm::FreeWorkgroupOp::create(builder, workgroup);
  return success();
}

/// Base for the CINM->CNM patterns, carrying the pass's `cnm-buffer-level`
/// option. The name is resolved against each op's own accelerator rather than
/// once for the pass, since the mapping from level name to memory-space
/// attribute is the platform's business.
template <typename OpT>
struct CinmToCnmPattern : public OpConversionPattern<OpT> {
  CinmToCnmPattern(MLIRContext *ctx, StringRef bufferLevel)
      : OpConversionPattern<OpT>(ctx), bufferLevel(bufferLevel) {}

  std::string bufferLevel;
};

// todo change that into
struct ConvertLinalgReduceIntoLaunch
    : public CinmToCnmPattern<linalg::ReduceOp> {
  using CinmToCnmPattern::CinmToCnmPattern;

  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto cnmAccelerator =
        mlir::cinm::getEnclosingAcceleratorAs<cnm::CnmAcceleratorAttrInterface>(
            op);

    if (!cnmAccelerator)
      return failure();

    auto level = resolveBufferLevel(bufferLevel, cnmAccelerator, op);
    if (failed(level))
      return failure();

    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, cnmAccelerator.getWorkgroupType());

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(
            builder, op, workgroup.getResult(), {op.getDimensions()},
            adaptor.getInputs(), adaptor.getInits(), adaptor.getInits(),
            op->getResults(), *level, newResults,
            [&](ImplicitLocOpBuilder &builder, ValueRange memrefInputs,
                ValueRange memrefOutputs) {
              // Here we are copying the original reduce into the launch,
              // except it's now operating on memrefs provided by cinm.
              // This can be lowered to affine or whatever afterwards.
              auto innerReduce = linalg::ReduceOp::create(
                  builder,
                  // no results bc memref
                  TypeRange{}, memrefInputs, memrefOutputs,
                  // todo we are hardcoding the dimensions
                  // This is because we flatten everything. This does not
                  // support custom reduction dimensions.
                  ArrayRef<int64_t>{0});

              IRMapping irMapping;
              op.getRegion().cloneInto(&innerReduce.getRegion(), irMapping);
            })
            .failed())
      return failure();
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

struct ConvertElementwiseOpToCnm : CinmToCnmPattern<cinm::ElementwiseOp> {
  ConvertElementwiseOpToCnm(MLIRContext *ctx, StringRef bufferLevel)
      : CinmToCnmPattern(ctx, bufferLevel) {
    this->setHasBoundedRewriteRecursion();
  }

  static std::optional<linalg::ElementwiseKind>
  toLinalgUnaryFn(cinm::ElementwiseKind kind) {
    switch (kind) {
    case cinm::ElementwiseKind::Neg:
      return linalg::ElementwiseKind::negf;
    case cinm::ElementwiseKind::Abs:
      return linalg::ElementwiseKind::abs;
    case cinm::ElementwiseKind::Ceil:
      return linalg::ElementwiseKind::ceil;
    case cinm::ElementwiseKind::Erf:
      return linalg::ElementwiseKind::erf;
    case cinm::ElementwiseKind::Exp:
      return linalg::ElementwiseKind::exp;
    case cinm::ElementwiseKind::Floor:
      return linalg::ElementwiseKind::floor;
    case cinm::ElementwiseKind::Log:
      return linalg::ElementwiseKind::log;
    case cinm::ElementwiseKind::Reciprocal:
      return linalg::ElementwiseKind::reciprocal;
    case cinm::ElementwiseKind::Round:
      return linalg::ElementwiseKind::round;
    case cinm::ElementwiseKind::Rsqrt:
      return linalg::ElementwiseKind::rsqrt;
    case cinm::ElementwiseKind::Sqrt:
      return linalg::ElementwiseKind::sqrt;
    case cinm::ElementwiseKind::Square:
      return linalg::ElementwiseKind::square;
    case cinm::ElementwiseKind::Tanh:
      return linalg::ElementwiseKind::tanh;
    default:
      return std::nullopt;
    }
  }

  LogicalResult
  matchAndRewrite(cinm::ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto cnmAccelerator =
        mlir::cinm::getEnclosingAcceleratorAs<cnm::CnmAcceleratorAttrInterface>(
            op);
    if (!cnmAccelerator)
      return failure();

    auto level = resolveBufferLevel(bufferLevel, cnmAccelerator, op);
    if (failed(level))
      return failure();

    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, cnmAccelerator.getWorkgroupType());

    // Initialize output for linalg.generic
    auto outputInit = tensor::EmptyOp::create(builder, op.getResult().getType(),
                                              ValueRange{});
    const Type elementType =
        dyn_cast_or_null<ShapedType>(op.getLhs().getType()).getElementType();
    bool isScalarOp = op.getRhs() && op.getRhs().getType() == elementType;

    SmallVector<ArrayRef<int64_t>> reductionDims(adaptor.getOperands().size(),
                                                 ArrayRef<int64_t>{});

    SmallVector<Value, 1> newResults;
    const auto conversionResult = convertCinmToCnm(
        builder, op, workgroup.getResult(), reductionDims,
        adaptor.getOperands(), ValueRange{outputInit}, ValueRange{op.getOut()},
        op->getResults(), *level, newResults,
        [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
            ValueRange outputs) {
          SmallVector<AffineMap> affineMaps;
          for (const auto &i : inputs) {
            MemRefType t = cast<MemRefType>(i.getType());
            affineMaps.push_back(AffineMap::getMultiDimIdentityMap(
                t.getRank(), op.getContext()));

            if (isScalarOp) {
              // for scalar ops only the first parameter is
              // passed to the linalg::generic op
              break;
            }
          }

          affineMaps.push_back(AffineMap::getMultiDimIdentityMap(
              cast<MemRefType>(outputs[0u].getType()).getRank(),
              op.getContext()));

          SmallVector<utils::IteratorType> iteratorTypes(
              cast<MemRefType>(inputs[0u].getType()).getRank(),
              utils::IteratorType::parallel);

          if (auto fn = toLinalgUnaryFn(op.getKind())) {
            linalg::ElementwiseOp::create(
                builder, ValueRange(inputs), ValueRange(outputs),
                builder.getAttr<linalg::ElementwiseKindAttr>(*fn),
                builder.getAffineMapArrayAttr(affineMaps));
          } else {
            linalg::GenericOp::create(
                builder, isScalarOp ? inputs.drop_back() : inputs, outputs,
                affineMaps, iteratorTypes,
                [&](OpBuilder &builder, Location loc, ValueRange args) {
                  Value lhs = args[0u];
                  Value rhs = op.getRhs() ? (isScalarOp ? inputs[1u] : args[1u])
                                          : Value();
                  if (isScalarOp) {
                    if (const auto memrefType =
                            dyn_cast<MemRefType>(rhs.getType())) {
                      const Value zero =
                          arith::ConstantIndexOp::create(builder, loc, 0);
                      rhs = memref::LoadOp::create(
                          builder, loc, rhs,
                          SmallVector<Value>(memrefType.getRank(), zero));
                    }
                  }

                  Value result;
                  bool isFloatOp = dyn_cast<FloatType>(elementType) != nullptr;
                  switch (op.getKind()) {
                  case cinm::ElementwiseKind::Add:
                    result = isFloatOp
                                 ? arith::AddFOp::create(builder, loc, lhs, rhs)
                                       .getResult()
                                 : arith::AddIOp::create(builder, loc, lhs, rhs)
                                       .getResult();
                    break;
                  case cinm::ElementwiseKind::Sub:
                    result = isFloatOp
                                 ? arith::SubFOp::create(builder, loc, lhs, rhs)
                                       .getResult()
                                 : arith::SubIOp::create(builder, loc, lhs, rhs)
                                       .getResult();
                    break;
                  case cinm::ElementwiseKind::Mul:
                    result = isFloatOp
                                 ? arith::MulFOp::create(builder, loc, lhs, rhs)
                                       .getResult()
                                 : arith::MulIOp::create(builder, loc, lhs, rhs)
                                       .getResult();
                    break;
                  case cinm::ElementwiseKind::Div:
                    result =
                        isFloatOp
                            ? arith::DivFOp::create(builder, loc, lhs, rhs)
                                  .getResult()
                            : arith::DivSIOp::create(builder, loc, lhs, rhs)
                                  .getResult();
                    break;
                  case cinm::ElementwiseKind::Mod:
                    result =
                        isFloatOp
                            ? arith::RemFOp::create(builder, loc, lhs, rhs)
                                  .getResult()
                            : arith::RemSIOp::create(builder, loc, lhs, rhs)
                                  .getResult();
                    break;
                  case cinm::ElementwiseKind::And:
                    result = arith::AndIOp::create(builder, loc, lhs, rhs)
                                 .getResult();
                    break;
                  case cinm::ElementwiseKind::Or:
                    result = arith::OrIOp::create(builder, loc, lhs, rhs)
                                 .getResult();
                    break;
                  case cinm::ElementwiseKind::Xor:
                    result = arith::XOrIOp::create(builder, loc, lhs, rhs)
                                 .getResult();
                    break;
                  case cinm::ElementwiseKind::Not: // ~a = a xor 0b111111111
                    result = arith::XOrIOp::create(
                                 builder, loc, lhs,
                                 arith::ConstantOp::create(
                                     builder, loc, lhs.getType(),
                                     builder.getIntegerAttr(lhs.getType(), -1)))
                                 .getResult();
                    break;
                  default:
                    break;
                  }

                  linalg::YieldOp::create(builder, loc, result);
                });
          }
        });

    if (conversionResult.failed()) {
      return failure();
    }

    rewriter.replaceOp(op, newResults);

    return success();
  }
};

LogicalResult computeScatterMapForGemm(cnm::BufferType bufferTyAB,
                                       int64_t rowsA, int64_t colsB,
                                       AffineMap &scatterA, AffineMap &scatterB,
                                       AffineMap &scatterGatherC) {

  auto wgElts = 1;
  SmallVector<int64_t, 4> wgShapeWithoutUnits; // only non unit dims
  for (auto dim : bufferTyAB.getWorkgroupShape()) {
    if (dim != 1) {
      wgElts *= dim;
      wgShapeWithoutUnits.push_back(dim);
    }
  }

  // this assumption relies on the tiling phase
  if (wgElts != rowsA * colsB)
    return failure();

  // couple of situations we know work
  if (wgShapeWithoutUnits == ArrayRef<int64_t>{rowsA, colsB} ||
      wgShapeWithoutUnits == ArrayRef<int64_t>{colsB, rowsA} ||
      wgElts == rowsA * colsB) {

    auto ctx = bufferTyAB.getContext();
    auto numInputs = bufferTyAB.getWorkgroupShape().size();
    const auto linearInput =
        mlir::linearizeIndices(ctx, bufferTyAB.getWorkgroupShape());

    scatterA = mlir::simplifyAffineMapWithBounds(
        AffineMap::get(numInputs, 0, linearInput % rowsA),
        bufferTyAB.getWorkgroupShape());

    scatterB = mlir::simplifyAffineMapWithBounds(
        AffineMap::get(numInputs, 0, linearInput % colsB),
        bufferTyAB.getWorkgroupShape());

    SmallVector<AffineExpr> results;
    mlir::structureIndex(linearInput, {rowsA, colsB}, results);
    scatterGatherC = mlir::simplifyAffineMapWithBounds(
        AffineMap::get(numInputs, 0, std::move(results), ctx),
        bufferTyAB.getWorkgroupShape());

    return success();
  }

  return failure();
}
template <class Op>
static Value getOutputInitForGemmLike(Op op, ImplicitLocOpBuilder &builder) {

  // Build the scatter init: bias (or zero) copied into the gather target so
  // the kernel accumulates bias + A*x.  If bias == out the copy is a no-op
  // and canonicalizes away.
  Value outputInit = op.getOut();
  if (outputInit) {
    // if bias need to copy it into the output
    if (op.getBias()) {
      return linalg::CopyOp::create(builder, op.getBias(), outputInit)
          .getResult(0);
    }
    // no bias: zero out the output, unless it already folds to a zero splat.
    if (isZeroSplatFoldable(outputInit))
      return outputInit;
    // not a zero: fill output with zero.
    auto resultTy = cast<ShapedType>(outputInit.getType()).getElementType();
    auto fillOp = linalg::FillOp::create(
        builder,
        arith::ConstantOp::create(builder, resultTy,
                                  builder.getZeroAttr(resultTy))
            .getResult(),
        outputInit);
    if (fillOp->getNumResults() > 0)
      return fillOp.getResult(0);
    return outputInit;
  }
  // without an output buffer, we need a tensor result, and the parameters are
  // tensors too
  assert(op.getResult() && "Need a tensor result");
  if (op.getBias())
    return op.getBias();
  // return a zero tensor
  return arith::ConstantOp::create(
      builder, op.getResult().getType(),
      builder.getZeroAttr(op.getResult().getType()));
}
struct ConvertCinmGemmToCnm : public CinmToCnmPattern<cinm::GemmOp> {
  using CinmToCnmPattern::CinmToCnmPattern;

  static Value transpose(ImplicitLocOpBuilder &builder, Value tensor) {
    auto inTy = cast<ShapedType>(tensor.getType());
    auto shape = inTy.getShape();

    SmallVector<int64_t, 2> newShape{shape[1], shape[0]};
    SmallVector<int64_t, 2> perms{1, 0};
    Value output;
    bool tensorOutput;
    if (llvm::isa<TensorType>(inTy)) {
      output =
          tensor::EmptyOp::create(builder, newShape, inTy.getElementType());
      tensorOutput = true;
    } else {
      output = memref::AllocOp::create(
          builder, MemRefType::get(newShape, inTy.getElementType()));
      tensorOutput = false;
    }
    auto transposeRight =
        linalg::TransposeOp::create(builder, tensor, output, perms);

    if (tensorOutput)
      return transposeRight->getResult(0);
    else
      return output;
  }

  LogicalResult
  matchAndRewrite(cinm::GemmOp op,
                  OpConversionPattern<cinm::GemmOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto lhs = llvm::cast<TypedValue<ShapedType>>(adaptor.getLhs());
    auto rhs = llvm::cast<TypedValue<ShapedType>>(adaptor.getRhs());

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto cnmAccelerator =
        mlir::cinm::getEnclosingAcceleratorAs<cnm::CnmAcceleratorAttrInterface>(
            op);
    if (!cnmAccelerator || !cnmAccelerator.bufferSizeOfLeaf())
      return failure();

    auto level = resolveBufferLevel(bufferLevel, cnmAccelerator, op);
    if (failed(level))
      return failure();

    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, cnmAccelerator.getWorkgroupType());
    auto wgShape = cnmAccelerator.getWorkgroupShape();

    auto transposeRight = transpose(builder, rhs);

    auto elTyBytes = lhs.getType().getElementTypeBitWidth() / 8;

    // Check that the tiling pass chose a fitting reduction size.
    auto reductionSize = lhs.getType().getDimSize(1);
    if (reductionSize * 2 * elTyBytes >
        cnmAccelerator.bufferSizeOfLeaf() - elTyBytes) {
      return op->emitOpError(
          "cannot be converted to CINM, reduction size is too large");
    }
    auto eltTy = lhs.getType().getElementType();
    // buffer type for A and B
    cnm::BufferType bufferType =
        cnm::BufferType::get({reductionSize}, eltTy, cnmAccelerator, *level);
    Value bufferA = cnm::AllocOp::create(builder, bufferType, workgroup);
    Value bufferB = cnm::AllocOp::create(builder, bufferType, workgroup);

    // C has a single element and no dimensions
    cnm::BufferType bufferCType =
        cnm::BufferType::get({}, eltTy, cnmAccelerator, *level);
    Value bufferC = cnm::AllocOp::create(builder, bufferCType, workgroup);

    //::mlir::Value input, ::mlir::Value buffer, ::mlir::Value wg,
    //:::mlir::AffineMap scatterMap);
    AffineMap scatterA;
    AffineMap scatterB;
    AffineMap scatterGatherC;
    if (computeScatterMapForGemm(bufferType, lhs.getType().getDimSize(0),
                                 rhs.getType().getDimSize(1), scatterA,
                                 scatterB, scatterGatherC)
            .failed()) {
      return op->emitOpError("Cannot be converted to CINM, parallel dims "
                             "cannot be mapped onto workgroup (")
             << wgShape << ")";
    }
    cnm::ScatterOp::create(builder, op.getLhs(), bufferA, workgroup,
                           std::move(scatterA));
    cnm::ScatterOp::create(builder, transposeRight, bufferB, workgroup,
                           std::move(scatterB));

    // Scatter init: bias (if any) or zero.  The `out` operand is only the
    // gather target; its contents are not read by the kernel.
    Value outputInit = getOutputInitForGemmLike(op, builder);
    assert(outputInit);
    cnm::ScatterOp::create(builder, outputInit, bufferC, workgroup,
                           scatterGatherC);

    SmallVector<AffineMap, 2> indexingMaps{
        AffineMap::getMultiDimIdentityMap(1, getContext()),
        AffineMap::getMultiDimIdentityMap(1, getContext()),
        AffineMap::get(1, 0, getContext()),
    };

    createLaunchOp(
        builder, workgroup, ValueRange{bufferA, bufferB}, ValueRange{bufferC},
        [&](ImplicitLocOpBuilder &builder, ValueRange ins, ValueRange outs) {
          linalg::ContractOp::create(
              builder, TypeRange{}, ins, outs,
              builder.getAffineMapArrayAttr(indexingMaps));
        });

    // Gather target: explicit `out` if provided, otherwise a fresh tensor.
    Value outbuf;
    if (op.getOut()) {
      outbuf = op.getOut();
    } else {
      assert(op.getResult() &&
             "cinm.gemm needs either an out buffer or a tensor result");
      outbuf = tensor::EmptyOp::create(builder, op.getResult().getType(),
                                       ValueRange{});
    }

    auto gather = cnm::GatherOp::create(builder, bufferC, workgroup,
                                        scatterGatherC, outbuf);

    if (op.getResult()) {
      Value result = gather.getOutput();
      if (!matchPattern(outputInit, m_Constant())) {
        // Add a materialization guard to relate the output of the gather with
        // the input of the scatter, in case they're a loop accumulator and we
        // need them to bufferize to the same buffer.

        // If it is a constant then we don't do that as that would create a copy
        // from the constant to the actual output buffer.
        result = bufferization::MaterializeInDestinationOp::create(
                     builder, result, outputInit)
                     .getResult();
      }

      rewriter.replaceOp(op, result);
    } else {
      rewriter.eraseOp(op);
    }

    // todo workgroup sharing.
    cnm::FreeWorkgroupOp::create(builder, workgroup);
    return success();
  }
};

struct ConvertCinmGemvToCnm : public CinmToCnmPattern<cinm::GemvOp> {
  ConvertCinmGemvToCnm(MLIRContext *ctx, StringRef bufferLevel)
      : CinmToCnmPattern(ctx, bufferLevel) {
    this->setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(cinm::GemvOp op,
                  OpConversionPattern<cinm::GemvOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto cnmAccelerator =
        mlir::cinm::getEnclosingAcceleratorAs<cnm::CnmAcceleratorAttrInterface>(
            op);
    if (!cnmAccelerator)
      return failure();

    auto level = resolveBufferLevel(bufferLevel, cnmAccelerator, op);
    if (failed(level))
      return failure();

    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, cnmAccelerator.getWorkgroupType());

    // Build the scatter init: bias (or zero) copied into the gather target so
    // the kernel accumulates bias + A*x.  If bias == out the copy is a no-op
    // and canonicalizes away.
    Value outputInit = getOutputInitForGemmLike(op, builder);

    llvm::SmallVector<Value, 1> newResults;
    // Only lhs/rhs are scattered as reduction inputs here (matching the two
    // entries in reductionDimensionsSorted below); bias/out are folded into
    // outputInit above and allocated once via outputInitializers. Passing
    // the full adaptor.getOperands() (4 segments: lhs, rhs, bias, out) would
    // inflate convertCinmToCnm's per-tasklet WRAM budget divisor with two
    // operands that never get their own buffer.
    if (convertCinmToCnm(
            builder, op, workgroup.getResult(), {{1}, {0}},
            ValueRange{adaptor.getLhs(), adaptor.getRhs()},
            ValueRange{outputInit},
            ValueRange{op.getOut()}, op->getResults(), *level, newResults,
            [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
                ValueRange outputs) {
              int outputRank =
                  dyn_cast<ShapedType>(outputs[0].getType()).getRank();
              auto ctx = builder.getContext();
              int numLoops = 1 + outputRank;

              // (m, k) -> (m, k)
              auto aMap = AffineMap::getMinorIdentityMap(numLoops, numLoops,
                                                         builder.getContext());

              // (m, k) -> (k)
              auto vMap =
                  AffineMap::getMinorIdentityMap(numLoops, 1, ctx);

              // (m, k) -> (m)
              auto resMap = aMap.dropResult(numLoops - 1);

              auto indexingMaps = builder.getAffineMapArrayAttr({
                  aMap,
                  vMap,
                  resMap,
              });
              linalg::ContractOp::create(builder, inputs, outputs,
                                         indexingMaps);
            })
            .failed()) {
      return failure();
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct ConvertCinmReduceToCnm : public CinmToCnmPattern<cinm::ReduceOp> {
  ConvertCinmReduceToCnm(MLIRContext *ctx, StringRef bufferLevel)
      : CinmToCnmPattern(ctx, bufferLevel) {
    this->setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(cinm::ReduceOp op,
                  OpConversionPattern<cinm::ReduceOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode is not handled here, same as the gemm-like patterns.
    if (!op.getResult())
      return failure();

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto cnmAccelerator =
        mlir::cinm::getEnclosingAcceleratorAs<cnm::CnmAcceleratorAttrInterface>(
            op);
    if (!cnmAccelerator)
      return failure();

    auto level = resolveBufferLevel(bufferLevel, cnmAccelerator, op);
    if (failed(level))
      return failure();

    cnm::WorkgroupOp workgroup =
        cnm::WorkgroupOp::create(builder, cnmAccelerator.getWorkgroupType());
    auto outputInit = arith::ConstantOp::create(
        builder, op.getResult().getType(),
        builder.getZeroAttr(op.getResult().getType()));

    SmallVector<int64_t> redDim = {static_cast<int64_t>(op.getDimension())};

    llvm::SmallVector<Value, 1> newResults;
    if (convertCinmToCnm(
            builder, op, workgroup.getResult(), {redDim}, adaptor.getOperands(),
            ValueRange{outputInit}, ValueRange{nullptr}, op->getResults(),
            *level, newResults,
            [&](ImplicitLocOpBuilder &builder, ValueRange inputs,
                ValueRange outputs) {
              linalg::ReduceOp::create(
                  builder, inputs, outputs, ArrayRef<int64_t>{0},
                  [&](OpBuilder &builder, Location loc,
                      ValueRange inputs) -> void {
                    arith::AtomicRMWKind arithMethod = cinm::getArithConstant(
                        op.getMethod(),
                        op.getInput().getType().getElementType());
                    Value result = arith::getReductionOp(
                        arithMethod, builder, loc, inputs[0], inputs[1]);
                    linalg::YieldOp::create(builder, loc, result);
                  });
            })
            .failed()) {
      return failure();
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

void populateCinmRewritePatterns(RewritePatternSet &patterns, MLIRContext *ctx,
                                 StringRef bufferLevel) {
  patterns.insert<ConvertLinalgReduceIntoLaunch>(ctx, bufferLevel);
  // elementwise
  patterns.insert<ConvertElementwiseOpToCnm>(ctx, bufferLevel);
  // matmul
  patterns.insert<ConvertCinmGemmToCnm>(ctx, bufferLevel);
  patterns.insert<ConvertCinmGemvToCnm>(ctx, bufferLevel);
  // reduce
  patterns.insert<ConvertCinmReduceToCnm>(ctx, bufferLevel);
}

struct ConvertTiledCinmToCnm
    : public impl::ConvertTiledCinmToCnmBase<ConvertTiledCinmToCnm> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCinmRewritePatterns(patterns, &getContext(), bufferLevel);
    ConversionTarget target(getContext());

    //  target.addIllegalDialect<linalg::ReduceOp>();
    //    target.addDynamicallyLegalOp<linalg::ReduceOp>(
    //      [](linalg::ReduceOp op) -> bool {
    //          return
    //          !WorkGroupMakerStrategy::determineWorkGroupTypeForRewrite(op);
    //      });
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addIllegalDialect<cinm::CinmDialect>();
    target.addLegalDialect<cnm::CnmDialect>();
    target.addLegalOp<cnm::LaunchOp>();
    target.addLegalOp<cinm::ComputeBlockOp>();
    target.addLegalOp<cinm::ComputeOp>();
    target.addLegalOp<cinm::YieldOp>();
    target.markOpRecursivelyLegal<cnm::LaunchOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cinm::createConvertTiledCinmToCnmPass() {
  return std::make_unique<ConvertTiledCinmToCnm>();
}

std::unique_ptr<Pass> mlir::cinm::createConvertTiledCinmToCnmPass(
    ConvertTiledCinmToCnmOptions options) {
  return std::make_unique<ConvertTiledCinmToCnm>(std::move(options));
}

void mlir::cinm::registerCinmToCnmPipeline() {
  // todo
}
