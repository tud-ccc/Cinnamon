#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmDialect.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"
#include "cinm-mlir/Utils/CinmUtils.h"

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
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/TilingInterface.h>

using namespace mlir;
using namespace mlir::cinm;

// ---------------------------------------------------------------------------
// Helpers (identical to the old CinmTilingImplementations.cpp)
// ---------------------------------------------------------------------------

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

static OpFoldResult getDimOfr(OpBuilder &b, Location loc,
                              TypedValue<ShapedType> shaped, int64_t dimIdx) {
  const int64_t size = shaped.getType().getDimSize(dimIdx);
  if (!ShapedType::isDynamic(size))
    return b.getIndexAttr(size);
  Value idx = arith::ConstantIndexOp::create(b, loc, dimIdx);
  if (isa<RankedTensorType>(shaped.getType()))
    return tensor::DimOp::create(b, loc, shaped, idx).getResult();
  return memref::DimOp::create(b, loc, shaped, idx).getResult();
}

namespace {
DiagnosedSilenceableFailure convertGemmlikeToTiledOps(
    RewriterBase &rewriter, cinm::GemmlikeOpInterface op,
    ArrayRef<int64_t> tilingFactors, SmallVectorImpl<Value> &results,
    std::function<Value(OpBuilder &, Location, Value, Value, Value, Value)>
        buildTileOp) {
  Location loc = op->getLoc();
  auto lhs = cast<TypedValue<ShapedType>>(op.getLhs());
  auto rhs = cast<TypedValue<ShapedType>>(op.getRhs());
  TypedValue<ShapedType> bias, out;
  if (Value v = op.getBias())
    bias = cast<TypedValue<ShapedType>>(v);
  if (Value v = op.getOut())
    out = cast<TypedValue<ShapedType>>(v);
  ShapedType resultType = op.isTensorVariant()
                              ? cast<ShapedType>(op.getGemmResult().getType())
                              : cast<ShapedType>(op.getOut().getType());
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

  const int64_t nBatch = lhsRank - 2;
  const int64_t r = tilingFactors.back();
  const SmallVector<int64_t> parTiles(tilingFactors.drop_back(1));
  const Type eltTy = resultType.getElementType();

  SmallVector<OpFoldResult> parBounds;
  for (int64_t i = 0; i <= nBatch; ++i)
    parBounds.push_back(getDimOfr(rewriter, loc, lhs, i));
  for (int64_t i = nBatch + 1; i < rhs.getType().getRank(); ++i)
    parBounds.push_back(getDimOfr(rewriter, loc, rhs, i));
  const OpFoldResult kBound = getDimOfr(rewriter, loc, lhs, nBatch + 1);

  ValueRange initArgs{};
  if (!out) {
    SmallVector<Value> dynamicDims;
    for (auto ofr : parBounds)
      if (auto v = ofr.dyn_cast<Value>())
        dynamicDims.push_back(v);
    Value resultInit = tensor::EmptyOp::create(
        rewriter, loc, cast<RankedTensorType>(resultType), dynamicDims);
    initArgs = resultInit;
  }

  SmallVector<Value> finals = createNestedAffineForLoops(
      rewriter, loc, ArrayRef<OpFoldResult>(parBounds), parTiles, initArgs,
      [&, nBatch, nPar](OpBuilder &b, Location loc, ValueRange indices,
                        ValueRange iterArgs) -> SmallVector<Value> {
        const ValueRange parIndices = indices;

        Value biasSlice;
        if (bias)
          biasSlice = extractSliceND(b, loc, bias, parTiles, parIndices);

        Value outBuf;
        if (out && isa<MemRefType>(out.getType())) {
          outBuf = extractSliceND(b, loc, out, parTiles, parIndices);
          if (biasSlice)
            linalg::AddOp::create(b, loc, ValueRange{biasSlice, outBuf},
                                  outBuf);
        }

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
            b, loc, ArrayRef<OpFoldResult>{kBound}, ArrayRef<int64_t>{r},
            innerIterArgInit,
            [&, nBatch, nPar](OpBuilder &b, Location loc, ValueRange indices,
                              ValueRange innerIterArgs) -> SmallVector<Value> {
              const Value k = indices[0];

              SmallVector<Value> lhsOffsets(parIndices.begin(),
                                            parIndices.begin() + nBatch + 1);
              lhsOffsets.push_back(k);
              SmallVector<int64_t> lhsSizes(parTiles.begin(),
                                            parTiles.begin() + nBatch + 1);
              lhsSizes.push_back(r);

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
                buildTileOp(b, loc, lhsSlice, rhsSlice, Value{}, outBuf);
                return {};
              }
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
// External model structs
// ---------------------------------------------------------------------------

struct ReduceTilingModel
    : public CinmTilingInterface::ExternalModel<ReduceTilingModel,
                                                cinm::ReduceOp> {
  void getTilableDimSizes(Operation *op,
                          SmallVectorImpl<int64_t> &dimSizes) const {
    auto reduce = cast<cinm::ReduceOp>(op);
    auto inputType = cast<ShapedType>(reduce.getOperand().getType());
    auto shape = inputType.getShape();
    dimSizes.append(shape.begin(), shape.end());
  }

  DiagnosedSilenceableFailure
  convertToTiledOps(Operation *op, RewriterBase &builder,
                    ArrayRef<int64_t> tileSizes,
                    SmallVectorImpl<Value> &results) const {
    auto reduce = cast<cinm::ReduceOp>(op);
    auto inputType = reduce.getInput().getType();
    if (static_cast<int64_t>(tileSizes.size()) != inputType.getRank())
      return emitSilenceableFailure(reduce.getLoc())
             << "expected " << inputType.getRank()
             << " tiling factors for reduce, got " << tileSizes.size();

    auto method = reduce.getMethod();

    int64_t reductionDim = reduce.getDimensionAttr().getInt();
    if (reductionDim < 0)
      reductionDim += inputType.getRank();

    auto neutral = arith::getIdentityValueAttr(
        getArithConstant(method, inputType.getElementType()),
        inputType.getElementType(), builder, reduce.getLoc());

    auto resultType = reduce.getResult().getType();

    Value result;
    if (isa<TensorType>(resultType))
      result =
          tensor::EmptyOp::create(builder, reduce.getLoc(), resultType, {});
    else if (resultType.isIntOrFloat())
      result = arith::ConstantOp::create(builder, reduce.getLoc(), neutral);
    else
      return emitSilenceableFailure(reduce.getLoc(),
                                    "Cannot tile reduction on type ")
             << resultType;

    SmallVector<Value> loopResult = createNestedAffineForLoops(
        builder, reduce.getLoc(), inputType.getShape(), tileSizes, {result},
        [&](OpBuilder &b, Location loc, ValueRange tileIndex,
            ValueRange iterArgs) -> SmallVector<Value> {
          auto acc = iterArgs[0];
          Value sliceIn =
              extractSliceND(b, loc, reduce.getInput(), tileSizes, tileIndex);

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
              llvm::dyn_cast_or_null<TypedValue<ShapedType>>(
                  smaller.getResult());
          auto shapedResult =
              llvm::dyn_cast_or_null<TypedValue<ShapedType>>(acc);

          if (shapedResult && shapedResultTile) {
            return {insertSliceND(b, loc, shapedResultTile, shapedResult,
                                  resultTileSize, resultTileIndex)};
          } else if (smaller.getResult().getType().isIntOrFloat()) {
            if (isa<TensorType>(acc.getType())) {
              auto accElt =
                  tensor::ExtractOp::create(b, loc, acc, resultTileIndex);
              auto red = arith::getReductionOp(
                  getArithConstant(method, accElt.getType()), b, loc, accElt,
                  smaller.getResult());
              return {
                  tensor::InsertOp::create(b, loc, red, acc, resultTileIndex)};
            } else if (acc.getType().isIntOrFloat()) {
              return {
                  arith::getReductionOp(getArithConstant(method, acc.getType()),
                                        b, loc, acc, smaller.getResult())};
            }
          }
          assert(false && "unhandled type");
        });

    results.append(loopResult.begin(), loopResult.end());
    return DiagnosedSilenceableFailure::success();
  }
};

struct ElementwiseTilingModel
    : public CinmTilingInterface::ExternalModel<ElementwiseTilingModel,
                                                cinm::ElementwiseOp> {
  void getTilableDimSizes(Operation *op,
                          SmallVectorImpl<int64_t> &dimSizes) const {
    auto ew = cast<cinm::ElementwiseOp>(op);
    dimSizes.push_back(
        cast<ShapedType>(ew.getLhs().getType()).getNumElements());
  }

  DiagnosedSilenceableFailure
  convertToTiledOps(Operation *op, RewriterBase &rewriter,
                    ArrayRef<int64_t> tilingFactors,
                    SmallVectorImpl<Value> &results) const {
    auto ew = cast<cinm::ElementwiseOp>(op);
    if (tilingFactors.size() != 1)
      return emitSilenceableFailure(ew.getLoc())
             << "expected 1 tiling factor for elementwise, got "
             << tilingFactors.size();

    ImplicitLocOpBuilder builder(ew.getLoc(), rewriter);

    TypedValue<ShapedType> lhs = ew.getLhs();
    TypedValue<ShapedType> rhs = ew.getRhs();
    const bool isUnaryOp = !rhs;

    ShapedType tensorTy = cast<ShapedType>(lhs.getType());
    auto shape = tensorTy.getShape();
    const ShapedType originalType = tensorTy;
    Value originalShapeValue;

    TypedValue<ShapedType> memrefOut =
        llvm::dyn_cast_or_null<TypedValue<ShapedType>>(ew.getOut());
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
    if (ew.getResult()) {
      resultInit = tensor::EmptyOp::create(builder, tensorTy, ValueRange{})
                       ->getResults();
    } else {
      assert(memrefOut);
    }

    SmallVector<Value> loopResult = createNestedAffineForLoops(
        builder, ew.getLoc(), {numElements}, {tileSize}, resultInit,
        [&](OpBuilder &b, Location loc, ValueRange indices,
            ValueRange iterArgs) -> SmallVector<Value> {
          Value base = indices[0];

          Value lhsSlice = extractSlice1D(b, loc, lhs, tileSize, base);

          Value rhsSlice;
          if (!isUnaryOp)
            rhsSlice = extractSlice1D(b, loc, rhs, tileSize, base);

          Value sliceOut;
          if (memrefOut)
            sliceOut = extractSlice1D(b, loc, memrefOut, tileSize, base);

          ElementwiseOp smaller = ElementwiseOp::create(
              b, loc, ew.getKind(), lhsSlice, rhsSlice, sliceOut);

          if (smaller.getResult()) {
            SmallVector<OpFoldResult, 1> siz{b.getIndexAttr(tileSize)};
            SmallVector<OpFoldResult, 1> str{b.getI64IntegerAttr(1)};
            SmallVector<OpFoldResult, 1> off{base};
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
};

template <class Self, class Op>
struct GemmLikeTilingModel
    : public CinmTilingInterface::ExternalModel<Self, Op> {
  DiagnosedSilenceableFailure
  convertToTiledOps(Operation *op, RewriterBase &rewriter,
                    ArrayRef<int64_t> tilingFactors,
                    SmallVectorImpl<Value> &results) const {
    return convertGemmlikeToTiledOps(
        rewriter, cast<Op>(op), tilingFactors, results,
        [](OpBuilder &b, Location loc, Value lhs, Value rhs, Value acc,
           Value out) -> Value {
          return Op::create(b, loc, lhs, rhs, acc, out).getResult();
        });
  }
};

struct GemmTilingModel
    : public GemmLikeTilingModel<GemmTilingModel, cinm::GemmOp> {
  void getTilableDimSizes(Operation *op,
                          SmallVectorImpl<int64_t> &dimSizes) const {
    auto gemm = cast<cinm::GemmOp>(op);
    auto lhsType = cast<ShapedType>(gemm.getLhs().getType());
    auto rhsType = cast<ShapedType>(gemm.getRhs().getType());
    dimSizes.push_back(lhsType.getDimSize(0)); // M
    dimSizes.push_back(rhsType.getDimSize(1)); // N
    dimSizes.push_back(lhsType.getDimSize(1)); // K
  }
};

struct GemvTilingModel
    : public GemmLikeTilingModel<GemvTilingModel, cinm::GemvOp> {
  void getTilableDimSizes(Operation *op,
                          SmallVectorImpl<int64_t> &dimSizes) const {
    auto lhsType = cast<ShapedType>(cast<cinm::GemvOp>(op).getLhs().getType());
    dimSizes.push_back(lhsType.getDimSize(0)); // M
    dimSizes.push_back(lhsType.getDimSize(1)); // K
  }
};

struct BatchGemmTilingModel
    : public GemmLikeTilingModel<BatchGemmTilingModel, cinm::BatchGemmOp> {
  void getTilableDimSizes(Operation *op,
                          SmallVectorImpl<int64_t> &dimSizes) const {
    auto gemm = cast<cinm::BatchGemmOp>(op);
    auto lhsType = cast<ShapedType>(gemm.getLhs().getType());
    auto rhsType = cast<ShapedType>(gemm.getRhs().getType());
    dimSizes.push_back(lhsType.getDimSize(0)); // B
    dimSizes.push_back(lhsType.getDimSize(1)); // M
    dimSizes.push_back(rhsType.getDimSize(2)); // N
    dimSizes.push_back(lhsType.getDimSize(2)); // K
  }
};

struct BatchGemvTilingModel
    : public GemmLikeTilingModel<BatchGemvTilingModel, cinm::BatchGemvOp> {
  void getTilableDimSizes(Operation *op,
                          SmallVectorImpl<int64_t> &dimSizes) const {
    auto lhsType =
        cast<ShapedType>(cast<cinm::BatchGemvOp>(op).getLhs().getType());
    dimSizes.push_back(lhsType.getDimSize(0)); // B
    dimSizes.push_back(lhsType.getDimSize(1)); // M
    dimSizes.push_back(lhsType.getDimSize(2)); // K
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void mlir::cinm::registerCinmTilingExternalModels(DialectRegistry &registry) {
  registry.addExtension<::mlir::cinm::CinmDialect>(
      +[](MLIRContext *ctx, ::mlir::cinm::CinmDialect *) {
        cinm::ReduceOp::attachInterface<ReduceTilingModel>(*ctx);
        cinm::ElementwiseOp::attachInterface<ElementwiseTilingModel>(*ctx);
        cinm::GemmOp::attachInterface<GemmTilingModel>(*ctx);
        cinm::GemvOp::attachInterface<GemvTilingModel>(*ctx);
        cinm::BatchGemmOp::attachInterface<BatchGemmTilingModel>(*ctx);
        cinm::BatchGemvOp::attachInterface<BatchGemvTilingModel>(*ctx);
      });
}
