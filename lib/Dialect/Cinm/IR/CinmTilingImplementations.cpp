#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmBase.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmUtils.h>
#include <cinm-mlir/Dialect/Cinm/IR/TilingInterface.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <cstdint>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/TilingInterface.h>

using namespace mlir;
using namespace mlir::cinm;

#include <cinm-mlir/Dialect/Cinm/IR/TilingInterface.cpp.inc>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static SmallVector<Value> createNestedScfForLoops(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> tripCounts,
    ArrayRef<int64_t> steps, ValueRange iterArgs,
    std::function<SmallVector<Value>(OpBuilder &, Location, ValueRange,
                                     ValueRange)>
        bodyBuilder) {
  assert(tripCounts.size() == steps.size());
  const unsigned rank = tripCounts.size();

  SmallVector<Value> lbs(rank), ubs(rank), stepVals(rank);
  for (unsigned d = 0; d < rank; ++d) {
    lbs[d] = arith::ConstantIndexOp::create(builder, loc, 0);
    ubs[d] = arith::ConstantIndexOp::create(builder, loc, tripCounts[d]);
    stepVals[d] = arith::ConstantIndexOp::create(builder, loc, steps[d]);
  }

  SmallVector<Value> ivs;
  ivs.reserve(rank);
  std::function<SmallVector<Value>(unsigned, ValueRange)> build;
  build = [&](unsigned depth, ValueRange carried) -> SmallVector<Value> {
    if (depth == rank) {
      return bodyBuilder(builder, loc, ivs, carried);
    }
    auto loop = scf::ForOp::create(builder, loc, lbs[depth], ubs[depth],
                                   stepVals[depth], carried);
    builder.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
    SmallVector<Value> yielded = build(depth + 1, loop.getRegionIterArgs());
    scf::YieldOp::create(builder, loc, yielded);
    ivs.pop_back();
    builder.setInsertionPointAfter(loop);
    return SmallVector<Value>(loop.getResults().begin(),
                              loop.getResults().end());
  };

  return build(0, iterArgs);
}

static Value extractSliceND(OpBuilder &builder, Location loc,
                            TypedValue<ShapedType> tensorOrMemref,
                            ArrayRef<int64_t> sizes, ValueRange offsets) {
  assert(offsets.size() == sizes.size());

  const ShapedType sliceTy = tensorOrMemref.getType().clone(sizes);
  if (llvm::isa<RankedTensorType>(tensorOrMemref.getType())) {
    llvm::SmallVector<int64_t> unitStrides(offsets.size(), 1);
    llvm::SmallVector<int64_t> noStaticOffsets(offsets.size(),
                                               ShapedType::kDynamic);
    return tensor::ExtractSliceOp::create(
        builder, loc, sliceTy, tensorOrMemref, offsets, ValueRange{},
        ValueRange{}, noStaticOffsets, sliceTy.getShape(), unitStrides);
  } else if (llvm::isa<MemRefType>(tensorOrMemref.getType())) {
    llvm::SmallVector<OpFoldResult> offsetsFoldRes(offsets.begin(),
                                                   offsets.end());
    auto one = builder.getI64IntegerAttr(1);
    llvm::SmallVector<OpFoldResult> stridesFoldRes(offsets.size(), one);
    llvm::SmallVector<OpFoldResult> sizesFoldRes;
    sizesFoldRes.reserve(offsets.size());
    for (auto i : sizes) {
      auto attr = builder.getI64IntegerAttr(i);
      sizesFoldRes.push_back(attr);
    }

    return memref::SubViewOp::create(builder, loc, tensorOrMemref,
                                     offsetsFoldRes, sizesFoldRes,
                                     stridesFoldRes);
  }
  assert(false && "type not handled");
}

static Value extractSlice1D(OpBuilder &builder, Location loc,
                            TypedValue<ShapedType> tensorOrMemref, int64_t size,
                            Value offset) {
  return extractSliceND(builder, loc, tensorOrMemref, {size}, {offset});
}

static Value insertSliceND(OpBuilder &builder, Location loc, Value slice,
                           TypedValue<ShapedType> tensorOrMemref,
                           ArrayRef<int64_t> sizes, ValueRange offsets) {
  assert(offsets.size() == sizes.size());

  const ShapedType sliceTy = tensorOrMemref.getType().clone(sizes);
  if (llvm::isa<RankedTensorType>(tensorOrMemref.getType())) {
    llvm::SmallVector<int64_t> unitStrides(offsets.size(), 1);
    llvm::SmallVector<int64_t> noStaticOffsets(offsets.size(),
                                               ShapedType::kDynamic);
    return tensor::InsertSliceOp::create(
        builder, loc, slice, tensorOrMemref, offsets, ValueRange{},
        ValueRange{}, noStaticOffsets, sliceTy.getShape(), unitStrides);
  } else if (llvm::isa<MemRefType>(tensorOrMemref.getType())) {
    auto dest = extractSliceND(builder, loc, tensorOrMemref, sizes, offsets);
    memref::CopyOp::create(builder, loc, slice, dest);
    return {};
  }
  assert(false && "type not handled");
}

// Shared tiling kernel for GemmOp, GemvOp, BatchGemmOp, BatchGemvOp.
//
// All four ops share the layout pattern:
//   lhs:    [...batch, M, K]
//   rhs:    [...batch, K, ...N]   (N dims follow the reduction dim)
//   result: [...batch, M, ...N]
//
// nBatch and nN are derived from the ranks:
//   nBatch = lhs.rank - 2,   nN = result.rank - (nBatch + 1)
//
// tilingFactors = [batch_tiles..., mTile, nTiles..., rTile]
//                 ↑ nPar = tilingFactors.size() - 1 parallel factors ↑
//
// buildTileOp(b, loc, lhsSlice, rhsSlice, acc, out) creates the inner tile op
// and returns its tensor result (or a null Value for the memref path).
static DiagnosedSilenceableFailure convertGemmlikeToTiledOps(
    RewriterBase &rewriter, Location loc, TypedValue<ShapedType> lhs,
    TypedValue<ShapedType> rhs, TypedValue<ShapedType> bias,
    TypedValue<ShapedType> out, ShapedType resultType,
    ArrayRef<int64_t> tilingFactors, SmallVectorImpl<Value> &results,
    std::function<Value(OpBuilder &, Location, Value, Value, Value, Value)>
        buildTileOp) {
  // Rank consistency:
  //   lhs    [...batch, M, K]       → rank = nBatch + 2
  //   rhs    [...batch, K, ...N]    → rank = nBatch + 1 + nN = result.rank
  //   result [...batch, M, ...N]    → rank = nBatch + 1 + nN
  const int64_t lhsRank = lhs.getType().getRank();
  const int64_t nPar = resultType.getRank();
  if (lhsRank < 2)
    return emitDefiniteFailure(loc)
           << "gemm-like op: lhs rank must be >= 2, got " << lhsRank;
  if (resultType.getRank() < lhsRank - 1)
    return emitDefiniteFailure(loc)
           << "gemm-like op: result rank too small for lhs rank " << lhsRank;
  if (rhs.getType().getRank() != resultType.getRank())
    return DiagnosedSilenceableFailure::definiteFailure();

  if (static_cast<int64_t>(tilingFactors.size()) != nPar + 1)
    return emitSilenceableFailure(loc)
           << "gemm-like op: expected " << nPar + 1 << " tiling factors, got "
           << tilingFactors.size();

  for (int64_t i = 0; i < lhsRank; ++i)
    if (ShapedType::isDynamic(lhs.getType().getDimSize(i)))
      return emitSilenceableFailure(loc)
             << "Unsupported: tiling on dynamic dimensions";
  for (int64_t i = 0; i < rhs.getType().getRank(); ++i)
    if (ShapedType::isDynamic(rhs.getType().getDimSize(i)))
      return emitSilenceableFailure(loc)
             << "Unsupported: tiling on dynamic dimensions";

  const int64_t nBatch = lhsRank - 2;
  const int64_t K = lhs.getType().getDimSize(nBatch + 1);
  const int64_t r = tilingFactors.back();
  const SmallVector<int64_t> parTiles(tilingFactors.drop_back(1));
  const Type eltTy = resultType.getElementType();

  ValueRange initArgs{};
  if (!out) {
    Value resultInit =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), eltTy);
    initArgs = resultInit;
  }

  SmallVector<Value> finals = createNestedAffineForLoops(
      rewriter, loc, resultType.getShape(), parTiles, initArgs,
      [&, nBatch, nPar](OpBuilder &b, Location loc, ValueRange indices,
                        ValueRange iterArgs) -> SmallVector<Value> {
        const ValueRange parIndices = indices;

        Value biasSlice;
        if (bias)
          biasSlice = extractSliceND(b, loc, bias, parTiles, parIndices);

        // If the output is a memref, slice it and accumulate bias before the
        // inner loop.
        Value outBuf;
        if (out && isa<MemRefType>(out.getType())) {
          outBuf = extractSliceND(b, loc, out, parTiles, parIndices);
          if (biasSlice)
            linalg::AddOp::create(b, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

        // Tensor case: seed the tile with bias or zeros, then carry the full
        // result tensor through the reduction loop. Keeping extract/insert
        // slice outside the inner loop improves bufferization results.
        ValueRange innerIterArgInit{};
        if (!outBuf) {
          Value initTile = biasSlice;
          if (!initTile) {
            auto tileTy = RankedTensorType::get(parTiles, eltTy);
            initTile = arith::ConstantOp::create(
                           b, loc,
                           DenseElementsAttr::get(tileTy, b.getZeroAttr(eltTy)))
                           .getResult();
          }
          innerIterArgInit = insertSliceND(
              b, loc, initTile, cast<TypedValue<ShapedType>>(iterArgs[0]),
              parTiles, parIndices);
        }

        SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
            b, loc, {K}, {r}, innerIterArgInit,
            [&, nBatch, nPar](OpBuilder &b, Location loc, ValueRange indices,
                              ValueRange innerIterArgs) -> SmallVector<Value> {
              const Value k = indices[0];

              // lhs: [...batch, M, K] → [...batch_idx, M_idx, k]
              SmallVector<Value> lhsOffsets(parIndices.begin(),
                                            parIndices.begin() + nBatch + 1);
              lhsOffsets.push_back(k);
              SmallVector<int64_t> lhsSizes(parTiles.begin(),
                                            parTiles.begin() + nBatch + 1);
              lhsSizes.push_back(r);

              // rhs: [...batch, K, ...N] → [...batch_idx, k, ...N_idx]
              SmallVector<Value> rhsOffsets(parIndices.begin(),
                                            parIndices.begin() + nBatch);
              rhsOffsets.push_back(k);
              SmallVector<int64_t> rhsSizes(parTiles.begin(),
                                            parTiles.begin() + nBatch);
              rhsSizes.push_back(r);
              for (int64_t i = nBatch + 1; i < nPar; ++i) {
                rhsOffsets.push_back(parIndices[i]);
                rhsSizes.push_back(parTiles[i]);
              }

              Value lhsSlice =
                  extractSliceND(b, loc, lhs, lhsSizes, lhsOffsets);
              Value rhsSlice =
                  extractSliceND(b, loc, rhs, rhsSizes, rhsOffsets);

              if (innerIterArgs.empty()) {
                // Memref path: accumulate in place.
                buildTileOp(b, loc, lhsSlice, rhsSlice, Value{}, outBuf);
                return {};
              }
              // Tensor path.
              Value accSlice = extractSliceND(
                  b, loc, cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                  parTiles, parIndices);
              Value tileResult =
                  buildTileOp(b, loc, lhsSlice, rhsSlice, accSlice, accSlice);
              Value updatedTensor =
                  insertSliceND(b, loc, tileResult,
                                cast<TypedValue<ShapedType>>(innerIterArgs[0]),
                                parTiles, parIndices);
              return {updatedTensor};
            });

        if (out)
          return {};
        return {reductionResult[0]};
      });

  results.append(finals.begin(), finals.end());
  return DiagnosedSilenceableFailure::success();
}

// ---------------------------------------------------------------------------
// getTilableDimSizes implementations
// ---------------------------------------------------------------------------

void ReduceOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto inputType = cast<ShapedType>(getOperand().getType());
  auto shape = inputType.getShape();
  dimSizes.append(shape.begin(), shape.end());
}

void ElementwiseOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  dimSizes.push_back(cast<ShapedType>(getLhs().getType()).getNumElements());
}

void GemmOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  auto rhsType = cast<ShapedType>(getRhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // M
  dimSizes.push_back(rhsType.getDimSize(1)); // N
  dimSizes.push_back(lhsType.getDimSize(1)); // K
}

void BatchGemmOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  auto rhsType = cast<ShapedType>(getRhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // B
  dimSizes.push_back(lhsType.getDimSize(1)); // M
  dimSizes.push_back(rhsType.getDimSize(2)); // N
  dimSizes.push_back(lhsType.getDimSize(2)); // K
}

void BatchGemvOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // B
  dimSizes.push_back(lhsType.getDimSize(1)); // M
  dimSizes.push_back(lhsType.getDimSize(2)); // K
}

void GemvOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  dimSizes.push_back(lhsType.getDimSize(0)); // M
  dimSizes.push_back(lhsType.getDimSize(1)); // K
}

void ActivateOp::getTilableDimSizes(SmallVectorImpl<int64_t> &dimSizes) {
  dimSizes.push_back(cast<ShapedType>(getInput().getType()).getNumElements());
}

// ---------------------------------------------------------------------------
// convertToTiledOps implementations
// ---------------------------------------------------------------------------

static TypedAttr getNeutralElement(ReduceMethod r, Type ty, OpBuilder &builder,
                                   Location loc) {
  return arith::getIdentityValueAttr(getArithConstant(r, ty), ty, builder, loc);
}

static Value materializeReduction(OpBuilder &builder, Location loc,
                                  ReduceMethod method, Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType());
  return arith::getReductionOp(getArithConstant(method, lhs.getType()), builder,
                               loc, lhs, rhs);
}

DiagnosedSilenceableFailure
ReduceOp::convertToTiledOps(RewriterBase &builder, ArrayRef<int64_t> tileSizes,
                            SmallVectorImpl<Value> &results) {
  auto inputType = getInput().getType();
  if (static_cast<int64_t>(tileSizes.size()) != inputType.getRank())
    return emitSilenceableFailure(getLoc())
           << "expected " << inputType.getRank()
           << " tiling factors for reduce, got " << tileSizes.size();

  auto method = getMethod();

  // To tile the reduction two different templates may be used:
  //   tensor<NxM> -> tensor<N>
  // - tile the parallel part (N) and concatenate the results
  // - tile the reduction part (M) and add the partial results

  // Assume you have been given tile sizes for both (for all dimensions
  // basically). Then:

  int64_t reductionDim = getDimensionAttr().getInt();
  if (reductionDim < 0)
    reductionDim += inputType.getRank();
  // auto reductionExtent = inputType.getDimSize(dim);

  auto neutral =
      getNeutralElement(method, inputType.getElementType(), builder, getLoc());

  auto resultType = getResult().getType();

  Value result;
  if (isa<TensorType>(resultType))
    result = tensor::EmptyOp::create(builder, getLoc(), resultType, {});
  else if (resultType.isIntOrFloat())
    result = arith::ConstantOp::create(builder, getLoc(), neutral);
  else
    // memref not supported
    return emitSilenceableFailure(getLoc(), "Cannot tile reduction on type ")
           << resultType;

  SmallVector<Value> loopResult = createNestedAffineForLoops(
      builder, getLoc(), inputType.getShape(), tileSizes, {result},
      [&](OpBuilder &b, Location loc, ValueRange tileIndex,
          ValueRange iterArgs) -> SmallVector<Value> {
        auto acc = iterArgs[0];
        Value sliceIn =
            extractSliceND(b, loc, getInput(), tileSizes, tileIndex);

        SmallVector<int64_t> resultTileSize(tileSizes);
        resultTileSize.erase(resultTileSize.begin() + reductionDim);
        SmallVector<Value> resultTileIndex(tileIndex);
        resultTileIndex.erase(resultTileIndex.begin() + reductionDim);

        Type resultTy = resultTileSize.size() == 0
                            ? inputType.getElementType()
                            : inputType.cloneWith(resultTileSize,
                                                  inputType.getElementType());

        auto smaller =
            ReduceOp::create(b, loc, resultTy, method, sliceIn, reductionDim);

        auto shapedResultTile =
            llvm::dyn_cast_or_null<TypedValue<ShapedType>>(smaller.getResult());
        auto shapedResult = llvm::dyn_cast_or_null<TypedValue<ShapedType>>(acc);

        if (shapedResult && shapedResultTile) {
          return {insertSliceND(b, loc, shapedResultTile, shapedResult,
                                resultTileSize, resultTileIndex)};
        } else if (smaller.getResult().getType().isIntOrFloat()) {
          if (isa<TensorType>(acc.getType())) {
            auto accElt =
                tensor::ExtractOp::create(b, loc, acc, resultTileIndex);
            auto red = materializeReduction(b, loc, method, accElt,
                                            smaller.getResult());
            return {
                tensor::InsertOp::create(b, loc, red, acc, resultTileIndex)};
          } else if (acc.getType().isIntOrFloat()) {
            // both are scalars
            return {
                materializeReduction(b, loc, method, acc, smaller.getResult())};
          }
        }
        assert(false && "unhandled type");
      });

  results.append(loopResult.begin(), loopResult.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
ElementwiseOp::convertToTiledOps(RewriterBase &rewriter,
                                 ArrayRef<int64_t> tilingFactors,
                                 SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 1)
    return emitSilenceableFailure(getLoc())
           << "expected 1 tiling factor for elementwise, got "
           << tilingFactors.size();

  ImplicitLocOpBuilder builder(getLoc(), rewriter);

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();
  const bool isUnaryOp = !rhs;

  ShapedType tensorTy = cast<ShapedType>(lhs.getType());
  auto shape = tensorTy.getShape();
  const ShapedType originalType = tensorTy;
  Value originalShapeValue;

  TypedValue<ShapedType> memrefOut =
      llvm::dyn_cast_or_null<TypedValue<ShapedType>>(getOut());
  if (shape.size() > 1) {
    originalShapeValue = arith::ConstantOp::create(
        builder,
        RankedTensorType::get({static_cast<int64_t>(shape.size())},
                              builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    lhs = mlir::reshapeStatic(builder, builder.getLoc(), lhs,
                              {tensorTy.getNumElements()});
    if (!isUnaryOp) {
      rhs = mlir::reshapeStatic(builder, builder.getLoc(), rhs,
                                {tensorTy.getNumElements()});
    }
    if (memrefOut) {
      memrefOut = mlir::reshapeStatic(builder, builder.getLoc(), memrefOut,
                                      {tensorTy.getNumElements()});
    }
    tensorTy = lhs.getType();
  }

  const int64_t numElements = tensorTy.getNumElements();
  int64_t tileSize =
      std::max<int64_t>(1, std::min<int64_t>(tilingFactors[0], numElements));

  ValueRange resultInit{};
  if (getResult()) {
    resultInit =
        tensor::EmptyOp::create(builder, tensorTy, ValueRange{})->getResults();
  } else {
    assert(memrefOut);
  }

  SmallVector<Value> loopResult = createNestedAffineForLoops(
      builder, getLoc(), {numElements}, {tileSize}, resultInit,
      [&](OpBuilder &b, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value base = indices[0];
        SmallVector<OpFoldResult, 1> off{base};

        Value lhsSlice = extractSlice1D(b, loc, lhs, tileSize, base);

        Value rhsSlice;
        if (!isUnaryOp)
          rhsSlice = extractSlice1D(b, loc, rhs, tileSize, base);

        Value sliceOut;
        if (memrefOut)
          sliceOut = extractSlice1D(b, loc, memrefOut, tileSize, base);

        ElementwiseOp smaller = ElementwiseOp::create(
            b, loc, getKind(), lhsSlice, rhsSlice, sliceOut);

        if (smaller.getResult()) {
          SmallVector<OpFoldResult, 1> siz{b.getIndexAttr(tileSize)};
          SmallVector<OpFoldResult, 1> str{b.getI64IntegerAttr(1)};
          Value subResult = tensor::InsertSliceOp::create(
              b, loc, smaller.getResult(), iterArgs[0], off, siz, str);
          return {subResult};
        }
        return {};
      });

  if (originalType.getRank() > 1) {
    loopResult[0] = tensor::ReshapeOp::create(
        builder, originalType, loopResult[0], originalShapeValue);
  }
  results.append(loopResult.begin(), loopResult.end());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
GemmOp::convertToTiledOps(RewriterBase &rewriter,
                          ArrayRef<int64_t> tilingFactors,
                          SmallVectorImpl<Value> &results) {
  ShapedType resultType = getResult() ? getResult().getType()
                                      : cast<ShapedType>(getOut().getType());
  return convertGemmlikeToTiledOps(
      rewriter, getLoc(), getLhs(), getRhs(), getBias(), getOut(), resultType,
      tilingFactors, results,
      [](OpBuilder &b, Location loc, Value lhs, Value rhs, Value acc,
         Value out) -> Value {
        return cinm::GemmOp::create(b, loc, lhs, rhs, acc, out).getResult();
      });
}

DiagnosedSilenceableFailure
BatchGemmOp::convertToTiledOps(RewriterBase &rewriter,
                               ArrayRef<int64_t> tilingFactors,
                               SmallVectorImpl<Value> &results) {
  ShapedType resultType = getResult() ? getResult().getType()
                                      : cast<ShapedType>(getOut().getType());
  return convertGemmlikeToTiledOps(
      rewriter, getLoc(), getLhs(), getRhs(), getBias(), getOut(), resultType,
      tilingFactors, results,
      [](OpBuilder &b, Location loc, Value lhs, Value rhs, Value acc,
         Value out) -> Value {
        return cinm::BatchGemmOp::create(b, loc, lhs, rhs, acc, out)
            .getResult();
      });
}

DiagnosedSilenceableFailure
BatchGemvOp::convertToTiledOps(RewriterBase &rewriter,
                               ArrayRef<int64_t> tilingFactors,
                               SmallVectorImpl<Value> &results) {
  ShapedType resultType = getResult() ? getResult().getType()
                                      : cast<ShapedType>(getOut().getType());
  return convertGemmlikeToTiledOps(
      rewriter, getLoc(), getLhs(), getRhs(), getBias(), getOut(), resultType,
      tilingFactors, results,
      [](OpBuilder &b, Location loc, Value lhs, Value rhs, Value acc,
         Value out) -> Value {
        return cinm::BatchGemvOp::create(b, loc, lhs, rhs, acc, out)
            .getResult();
      });
}

DiagnosedSilenceableFailure
GemvOp::convertToTiledOps(RewriterBase &rewriter,
                          ArrayRef<int64_t> tilingFactors,
                          SmallVectorImpl<Value> &results) {
  ShapedType resultType = getResult() ? getResult().getType()
                                      : cast<ShapedType>(getOut().getType());
  return convertGemmlikeToTiledOps(
      rewriter, getLoc(), getLhs(), getRhs(), getBias(), getOut(), resultType,
      tilingFactors, results,
      [](OpBuilder &b, Location loc, Value lhs, Value rhs, Value acc,
         Value out) -> Value {
        return cinm::GemvOp::create(b, loc, lhs, rhs, acc, out).getResult();
      });
}

DiagnosedSilenceableFailure
ActivateOp::convertToTiledOps(RewriterBase &rewriter,
                              ArrayRef<int64_t> tilingFactors,
                              SmallVectorImpl<Value> &results) {
  if (tilingFactors.size() != 1)
    return emitSilenceableFailure(getLoc())
           << "expected 1 tiling factor for activate, got "
           << tilingFactors.size();

  ImplicitLocOpBuilder builder(getLoc(), rewriter);
  auto inputT = getInput();
  auto inTy = inputT.getType();
  Type elt = inTy.getElementType();

  const ShapedType originalTy = inTy;
  Value originalShapeValue;
  if (inTy.getRank() > 1) {
    originalShapeValue = arith::ConstantOp::create(
        builder, RankedTensorType::get({inTy.getRank()}, builder.getI64Type()),
        builder.getI64TensorAttr(inTy.getShape()));
    inputT = mlir::reshapeStatic(builder, builder.getLoc(), inputT,
                                 ArrayRef<int64_t>{inTy.getNumElements()});
    inTy = inputT.getType();
  }

  const int64_t total = inTy.getNumElements();
  const int64_t p = tilingFactors[0];

  Value init = tensor::EmptyOp::create(builder, inTy.getShape(), elt);
  Value totalC = arith::ConstantIndexOp::create(builder, total);
  Value pC = arith::ConstantIndexOp::create(builder, p);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, getLoc(), ArrayRef<int64_t>{total}, ArrayRef<int64_t>{p},
      ValueRange{init},
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange iters) -> SmallVector<Value> {
        Value i = ivs[0];

        Value rem = arith::SubIOp::create(b, loc, totalC, i);
        Value useP =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, rem, pC);
        Value thisTile = arith::SelectOp::create(b, loc, useP, pC, rem);

        SmallVector<OpFoldResult> off{i};
        SmallVector<OpFoldResult> siz{thisTile};
        SmallVector<OpFoldResult> str{b.getI64IntegerAttr(1)};

        auto inSlice =
            tensor::ExtractSliceOp::create(b, loc, inputT, off, siz, str);
        auto tile =
            cinm::ActivateOp::create(b, loc, getKind(), inSlice, Value());

        Value out = tensor::InsertSliceOp::create(b, loc, tile.getResult(),
                                                  iters[0], off, siz, str);
        return {out};
      });

  if (originalTy.getRank() > 1) {
    finals[0] = tensor::ReshapeOp::create(builder, originalTy, finals[0],
                                          originalShapeValue);
  }
  results.append(finals.begin(), finals.end());
  return DiagnosedSilenceableFailure::success();
}
