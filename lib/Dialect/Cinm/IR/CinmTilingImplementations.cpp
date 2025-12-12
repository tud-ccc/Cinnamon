#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <array>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/ValueRange.h>

using namespace mlir;
using namespace mlir::cinm;

using TilingResult2 = FailureOr<SmallVector<Value>>;

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

TilingResult2 ReduceOp::convertToTiledOps(OpBuilder &builder,
                                          TilingParameters params) {
  auto ty = getInput().getType();
  auto reduceClusterSize =
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());

  auto method = getMethod();
  if (method == ReduceMethod::ADD) {
    return TilingResult2(
        {createVectorReduceAdd(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MUL) {
    return TilingResult2(
        {createVectorReduceMul(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MAX) {
    return TilingResult2(
        {createVectorReduceMax(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else if (method == ReduceMethod::MIN) {
    return TilingResult2(
        {createVectorReduceMin(builder, getLoc(), getOperand(),
                               getDimensionsAttr(), reduceClusterSize)});
  } else {
    abort();
  }
}

static constexpr std::array<int64_t, 2> noStaticOffsets2{ShapedType::kDynamic,
                                                         ShapedType::kDynamic};
static constexpr std::array<int64_t, 2> unitStrides2{1, 1};
static constexpr std::array<int64_t, 1> noStaticOffsets1{ShapedType::kDynamic};
static constexpr std::array<int64_t, 1> unitStrides1{1};
static constexpr std::array<int64_t, 3> noStaticOffsets3{
    ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};
static constexpr std::array<int64_t, 3> unitStrides3{1, 1, 1};

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getGemmTilesFromAttributes(const ShapedType &lhsType, const ShapedType &rhsType,
                           const mlir::cinm::TilingParameters &params) {
  int64_t p0 = 0, p1 = 0;
  if (auto providedPar = params.getProvidedParallelTiles()) {
    std::tie(p0, p1) = *providedPar;
  } else {
    return failure();
  }
  if (p0 <= 0 || p1 <= 0)
    return failure();

  int64_t r = 0;
  if (auto providedR = params.getProvidedReductionTile()) {
    r = *providedR;
  } else {
    return failure();
  }

  const int64_t K = lhsType.getDimSize(1);
  if (r <= 0 || r > K)
    return failure();

  const int64_t maxElems =
      params.maxNumElementsOfType(lhsType.getElementType());
  const int64_t maxSizePerBuffer = (maxElems - 1) / 2;
  if (r > maxSizePerBuffer)
    return failure();

  return std::make_tuple(p0, p1, r);
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t, int64_t>>
getBatchGemmTilesFromAttributes(const ShapedType &lhsType,
                                const ShapedType &rhsType,
                                const mlir::cinm::TilingParameters &params) {
  auto tiles = params.getTileSizes();
  if (!tiles || tiles->size() < 4)
    return failure();
  int64_t bTile = (*tiles)[0];
  int64_t mTile = (*tiles)[1];
  int64_t nTile = (*tiles)[2];
  int64_t rTile = (*tiles)[3];
  if (bTile <= 0 || mTile <= 0 || nTile <= 0 || rTile <= 0)
    return failure();

  const int64_t K = lhsType.getDimSize(2);
  if (!ShapedType::isDynamic(K) && rTile > K)
    return failure();

  const int64_t maxElems =
      params.maxNumElementsOfType(lhsType.getElementType());
  const int64_t maxPerBuffer = (maxElems - 1) / 2;
  if (rTile > maxPerBuffer)
    return failure();

  return std::make_tuple(bTile, mTile, nTile, rTile);
}

static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getBatchGemvTilesFromAttributes(const ShapedType &lhsType,
                                const ShapedType &rhsType,
                                const mlir::cinm::TilingParameters &params) {
  auto tiles = params.getTileSizes();
  if (!tiles || tiles->size() < 3)
    return failure();
  int64_t bTile = (*tiles)[0];
  int64_t mTile = (*tiles)[1];
  int64_t rTile = (*tiles)[2];
  if (bTile <= 0 || mTile <= 0 || rTile <= 0)
    return failure();
  const int64_t K = lhsType.getDimSize(2);
  if (!ShapedType::isDynamic(K) && rTile > K)
    return failure();
  const int64_t maxElems =
      params.maxNumElementsOfType(lhsType.getElementType());
  const int64_t maxPerBuffer = (maxElems - 1) / 2;
  if (rTile > maxPerBuffer)
    return failure();
  return std::make_tuple(bTile, mTile, rTile);
}

TilingResult2 ElementwiseOp::convertToTiledOps(OpBuilder &builder0,
                                               TilingParameters params) {
  ImplicitLocOpBuilder builder(getLoc(), builder0);
  const bool isUnaryOp = !getRhs();

  TypedValue<ShapedType> lhs = getLhs();
  TypedValue<ShapedType> rhs = getRhs();

  ShapedType tensorTy = cast<ShapedType>(lhs.getType());
  auto shape = tensorTy.getShape();
  const ShapedType originalType = tensorTy;
  Value originalShapeValue;
  if (shape.size() > 1) {
    originalShapeValue = arith::ConstantOp::create(
        builder,
        RankedTensorType::get({static_cast<int64_t>(shape.size())},
                              builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    lhs = cinm::reshapeStatic(builder, builder.getLoc(), lhs,
                              {tensorTy.getNumElements()});
    if (!isUnaryOp) {
      rhs = cinm::reshapeStatic(builder, builder.getLoc(), rhs,
                                {tensorTy.getNumElements()});
    }
    tensorTy = cast<RankedTensorType>(lhs.getType());
  }

  int64_t tileSize = 0;
  if (auto provided = params.getTileSizes()) {
    if (!provided->empty())
      tileSize = (*provided)[0];
  }
  if (tileSize <= 0) {
    emitError() << "elementwise tiling requires a positive tile size in "
                   "compute.tileSizes";
    return failure();
  }

  const int64_t numElements = tensorTy.getNumElements();
  tileSize = std::max<int64_t>(1, std::min<int64_t>(tileSize, numElements));

  Value resultInit = tensor::EmptyOp::create(builder, tensorTy, ValueRange{});
  Value totalC = arith::ConstantIndexOp::create(builder, numElements);
  Value tileC = arith::ConstantIndexOp::create(builder, tileSize);

  SmallVector<Value> result = createNestedScfForLoops(
      builder, getLoc(), {numElements}, {tileSize}, ValueRange{resultInit},
      [&](OpBuilder &b, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value base = indices[0];
        SmallVector<OpFoldResult, 1> off{base};

        Value rem = arith::SubIOp::create(b, loc, totalC, base);
        Value useTile = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                              rem, tileC);
        Value thisTile = arith::SelectOp::create(b, loc, useTile, tileC, rem);
        SmallVector<OpFoldResult, 1> siz{thisTile};
        SmallVector<OpFoldResult, 1> str{b.getI64IntegerAttr(1)};

        Value lhsSlice =
            tensor::ExtractSliceOp::create(b, loc, lhs, off, siz, str);

        Value rhsSlice = nullptr;
        if (!isUnaryOp) {
          rhsSlice = tensor::ExtractSliceOp::create(b, loc, rhs, off, siz, str);
        }

        ElementwiseOp smaller = ElementwiseOp::create(
            b, loc, getKind(), lhsSlice, rhsSlice, Value());
        markOpAsNoTile(smaller);

        if (smaller.getResult()) {
          Value subResult = tensor::InsertSliceOp::create(
              b, loc, smaller.getResult(), iterArgs[0], off, siz, str);
          return {subResult};
        } else {
          return {};
        }
      });

  if (originalType.getRank() > 1) {
    result[0] = tensor::ReshapeOp::create(builder, originalType, result[0],
                                          originalShapeValue);
  }
  return TilingResult2(result);
}

TilingResult2 GemmOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  Location loc = getLoc();

  Value lhs = getLhs();
  Value rhs = getRhs();

  auto lhsType = cast<ShapedType>(lhs.getType());
  auto rhsType = cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult()) {
    resultType = getResult().getType();
  } else {
    resultType = getOut().getType();
  }

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return failure();

  const int64_t M = lhsType.getDimSize(0);
  const int64_t K = lhsType.getDimSize(1);
  const int64_t N = rhsType.getDimSize(1);
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
      ShapedType::isDynamic(N))
    return failure();

  Value resultInit = tensor::EmptyOp::create(
      builder, loc, resultType.getShape(), resultType.getElementType());

  int64_t p0 = 0, p1 = 0, r = 0;
  if (auto tiles = getGemmTilesFromAttributes(lhsType, rhsType, params);
      succeeded(tiles)) {
    std::tie(p0, p1, r) = *tiles;
  } else {
    getOperation()->emitError()
        << "requires tileSizes attribute with [M, N, K] entries";
    return failure();
  }

  Type eltTy = resultType.getElementType();
  TypedAttr zeroElt = builder.getZeroAttr(eltTy);

  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Nc = arith::ConstantIndexOp::create(builder, loc, N);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value p0c = arith::ConstantIndexOp::create(builder, loc, p0);
  Value p1c = arith::ConstantIndexOp::create(builder, loc, p1);
  Value rc = arith::ConstantIndexOp::create(builder, loc, r);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{M, N}, ArrayRef<int64_t>{p0, p1},
      ValueRange{resultInit},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iters) -> SmallVector<Value> {
        Value iM = ivs[0];
        Value jN = ivs[1];

        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value remN = arith::SubIOp::create(b, loc2, Nc, jN);
        Value useP0 = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                            remM, p0c);
        Value useP1 = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                            remN, p1c);
        Value pMTile = arith::SelectOp::create(b, loc2, useP0, p0c, remM);
        Value pNTile = arith::SelectOp::create(b, loc2, useP1, p1c, remN);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroElt);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            eltTy, ValueRange{pMTile, pNTile});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{r},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange ivs2,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = ivs2[0];
              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rc);
              Value kTile = arith::SelectOp::create(b2, loc3, useR, rc, remK);

              auto lhsTileTy = RankedTensorType::get(
                  {ShapedType::kDynamic, ShapedType::kDynamic}, eltTy);
              Value lhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, lhs, ValueRange{iM, k},
                  ValueRange{pMTile, kTile}, ValueRange{},

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),
                  ArrayRef<int64_t>({1, 1}));

              auto rhsTileTy = RankedTensorType::get(
                  {ShapedType::kDynamic, ShapedType::kDynamic}, eltTy);
              Value rhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, rhsTileTy, rhs, ValueRange{k, jN},
                  ValueRange{kTile, pNTile}, ValueRange{},

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),
                  ArrayRef<int64_t>({1, 1}));

              auto tileGemm = cinm::GemmOp::create(b2, loc3, lhsSlice, rhsSlice,
                                                   accArgs[0]);
              cinm::markOpAsNoTile(tileGemm);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemm.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iters[0], ValueRange{iM, jN},
            ValueRange{pMTile, pNTile}, ValueRange{},

            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),

            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            ArrayRef<int64_t>({1, 1}));

        return {out};
      });

  return TilingResult2(finals);
}

TilingResult2 BatchGemmOp::convertToTiledOps(OpBuilder &builder,
                                             TilingParameters params) {
  Location loc = getLoc();

  Value lhs = getLhs();
  Value rhs = getRhs();
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult()) {
    resultType = getResult().getType();
  } else {
    resultType = getOut().getType();
  }
  if (!lhsType || !rhsType || !resultType)
    return failure();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3 ||
      resultType.getRank() != 3)
    return failure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  const int64_t N = rhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K) || ShapedType::isDynamic(N))
    return failure();

  Type elementTy = lhsType.getElementType();

  int64_t bTile = 0, mTile = 0, nTile = 0, rTile = 0;
  if (auto provided = getBatchGemmTilesFromAttributes(lhsType, rhsType, params);
      succeeded(provided)) {
    std::tie(bTile, mTile, nTile, rTile) = *provided;
  } else {
    getOperation()->emitError()
        << "requires tileSizes attribute with [batch, M, N, K] entries";
    return failure();
  }

  Value resultInit =
      tensor::EmptyOp::create(builder, loc, resultType.getShape(), elementTy);

  TypedAttr zeroAttr = builder.getZeroAttr(elementTy);

  Value Bc = arith::ConstantIndexOp::create(builder, loc, B);
  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Nc = arith::ConstantIndexOp::create(builder, loc, N);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value bTileC = arith::ConstantIndexOp::create(builder, loc, bTile);
  Value mTileC = arith::ConstantIndexOp::create(builder, loc, mTile);
  Value nTileC = arith::ConstantIndexOp::create(builder, loc, nTile);
  Value rTileC = arith::ConstantIndexOp::create(builder, loc, rTile);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{B, M, N},
      ArrayRef<int64_t>{bTile, mTile, nTile}, ValueRange{resultInit},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value iB = ivs[0];
        Value iM = ivs[1];
        Value jN = ivs[2];

        Value remB = arith::SubIOp::create(b, loc2, Bc, iB);
        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value remN = arith::SubIOp::create(b, loc2, Nc, jN);

        Value useB = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remB, bTileC);
        Value useM = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remM, mTileC);
        Value useN = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remN, nTileC);

        Value bTileDyn = arith::SelectOp::create(b, loc2, useB, bTileC, remB);
        Value mTileDyn = arith::SelectOp::create(b, loc2, useM, mTileC, remM);
        Value nTileDyn = arith::SelectOp::create(b, loc2, useN, nTileC, remN);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroAttr);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                               ShapedType::kDynamic}),
            elementTy, ValueRange{bTileDyn, mTileDyn, nTileDyn});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rTile},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange redIvs,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = redIvs[0];
              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rTileC);
              Value kTileDyn =
                  arith::SelectOp::create(b2, loc3, useR, rTileC, remK);

              auto lhsTileTy = RankedTensorType::get({ShapedType::kDynamic,
                                                      ShapedType::kDynamic,
                                                      ShapedType::kDynamic},
                                                     elementTy);
              Value lhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, lhs, ValueRange{iB, iM, k},
                  ValueRange{bTileDyn, mTileDyn, kTileDyn}, ValueRange{},
                  noStaticOffsets3,
                  ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                                     ShapedType::kDynamic}),
                  unitStrides3);

              Value rhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, rhs, ValueRange{iB, k, jN},
                  ValueRange{bTileDyn, kTileDyn, nTileDyn}, ValueRange{},
                  noStaticOffsets3,
                  ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                                     ShapedType::kDynamic}),
                  unitStrides3);

              auto tileGemm = cinm::BatchGemmOp::create(b2, loc3, lhsSlice,
                                                        rhsSlice, accArgs[0]);
              cinm::markOpAsNoTile(tileGemm);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemm.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iterArgs[0], ValueRange{iB, iM, jN},
            ValueRange{bTileDyn, mTileDyn, nTileDyn}, ValueRange{},
            noStaticOffsets3,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                               ShapedType::kDynamic}),
            unitStrides3);

        return SmallVector<Value>{out};
      });

  return TilingResult2(finals);
}

TilingResult2 BatchGemvOp::convertToTiledOps(OpBuilder &builder,
                                             TilingParameters params) {
  Location loc = getLoc();

  Value lhs = getLhs();
  Value rhs = getRhs();
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  ShapedType resultType;
  if (getResult()) {
    resultType = getResult().getType();
  } else {
    resultType = getOut().getType();
  }
  if (!lhsType || !rhsType || !resultType)
    return failure();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return failure();

  const int64_t B = lhsType.getDimSize(0);
  const int64_t M = lhsType.getDimSize(1);
  const int64_t K = lhsType.getDimSize(2);
  if (ShapedType::isDynamic(B) || ShapedType::isDynamic(M) ||
      ShapedType::isDynamic(K))
    return failure();

  Type elementTy = lhsType.getElementType();

  int64_t bTile = 0, mTile = 0, rTile = 0;
  if (auto provided = getBatchGemvTilesFromAttributes(lhsType, rhsType, params);
      succeeded(provided)) {
    std::tie(bTile, mTile, rTile) = *provided;
  } else {
    getOperation()->emitError()
        << "requires tileSizes attribute with [batch, M, K] entries";
    return failure();
  }

  Value resultInit =
      tensor::EmptyOp::create(builder, loc, resultType.getShape(), elementTy);

  TypedAttr zeroAttr = builder.getZeroAttr(elementTy);

  Value Bc = arith::ConstantIndexOp::create(builder, loc, B);
  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value bTileC = arith::ConstantIndexOp::create(builder, loc, bTile);
  Value mTileC = arith::ConstantIndexOp::create(builder, loc, mTile);
  Value rTileC = arith::ConstantIndexOp::create(builder, loc, rTile);

  SmallVector<Value> finals = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{B, M}, ArrayRef<int64_t>{bTile, mTile},
      ValueRange{resultInit},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iterArgs) -> SmallVector<Value> {
        Value iB = ivs[0];
        Value iM = ivs[1];

        Value remB = arith::SubIOp::create(b, loc2, Bc, iB);
        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value useB = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remB, bTileC);
        Value useM = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                           remM, mTileC);
        Value bTileDyn = arith::SelectOp::create(b, loc2, useB, bTileC, remB);
        Value mTileDyn = arith::SelectOp::create(b, loc2, useM, mTileC, remM);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroAttr);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            elementTy, ValueRange{bTileDyn, mTileDyn});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rTile},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange redIvs,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = redIvs[0];
              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rTileC);
              Value kTileDyn =
                  arith::SelectOp::create(b2, loc3, useR, rTileC, remK);

              auto lhsTileTy = RankedTensorType::get({ShapedType::kDynamic,
                                                      ShapedType::kDynamic,
                                                      ShapedType::kDynamic},
                                                     elementTy);
              Value lhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, lhsTileTy, lhs, ValueRange{iB, iM, k},
                  ValueRange{bTileDyn, mTileDyn, kTileDyn}, ValueRange{},
                  noStaticOffsets3,
                  ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic,
                                     ShapedType::kDynamic}),
                  unitStrides3);

              auto rhsTileTy = RankedTensorType::get(
                  {ShapedType::kDynamic, ShapedType::kDynamic}, elementTy);
              Value rhsSlice = tensor::ExtractSliceOp::create(
                  b2, loc3, rhsTileTy, rhs, ValueRange{iB, k},
                  ValueRange{bTileDyn, kTileDyn}, ValueRange{},
                  noStaticOffsets2,
                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),
                  unitStrides2);

              auto tileGemv = cinm::BatchGemvOp::create(b2, loc3, lhsSlice,
                                                        rhsSlice, accArgs[0]);
              cinm::markOpAsNoTile(tileGemv);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, tileGemv.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iterArgs[0], ValueRange{iB, iM},
            ValueRange{bTileDyn, mTileDyn}, ValueRange{}, noStaticOffsets2,
            ArrayRef<int64_t>({ShapedType::kDynamic, ShapedType::kDynamic}),
            unitStrides2);

        return SmallVector<Value>{out};
      });

  return TilingResult2(finals);
}

TilingResult2 GemvOp::convertToTiledOps(OpBuilder &builder,
                                        TilingParameters params) {
  Location loc = getLoc();

  Value A = getLhs();
  Value x = getRhs();

  auto aTy = cast<ShapedType>(A.getType());
  auto xTy = cast<ShapedType>(x.getType());
  ShapedType yTy;
  if (getResult()) {
    yTy = getResult().getType();
  } else {
    yTy = getOut().getType();
  }
  if (aTy.getRank() != 2 || xTy.getRank() != 1 || yTy.getRank() != 1)
    return failure();
  if (aTy.getElementType() != xTy.getElementType() ||
      aTy.getElementType() != yTy.getElementType())
    return failure();

  const int64_t M = aTy.getDimSize(0);
  const int64_t K = aTy.getDimSize(1);
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K))
    return failure();

  Type elt = aTy.getElementType();

  int64_t pM = 0, rK = 0;
  if (auto tiles = params.getTileSizes()) {
    if (tiles->size() >= 1)
      pM = (*tiles)[0];
    if (tiles->size() >= 2)
      rK = (*tiles)[1];
  }
  if (pM <= 0 || rK <= 0) {
    getOperation()->emitError()
        << "requires tileSizes attribute with [M, K] entries";
    return failure();
  }

  Value init = tensor::EmptyOp::create(builder, loc, yTy.getShape(), elt);

  TypedAttr zeroAttr = builder.getZeroAttr(elt);

  Value Mc = arith::ConstantIndexOp::create(builder, loc, M);
  Value Kc = arith::ConstantIndexOp::create(builder, loc, K);
  Value pMc = arith::ConstantIndexOp::create(builder, loc, pM);
  Value rKc = arith::ConstantIndexOp::create(builder, loc, rK);

  SmallVector<Value> results = createNestedScfForLoops(
      builder, loc, ArrayRef<int64_t>{M}, ArrayRef<int64_t>{pM},
      ValueRange{init},
      [&](OpBuilder &b, Location loc2, ValueRange ivs,
          ValueRange iters) -> SmallVector<Value> {
        Value iM = ivs[0];

        Value remM = arith::SubIOp::create(b, loc2, Mc, iM);
        Value usePM = arith::CmpIOp::create(b, loc2, arith::CmpIPredicate::ugt,
                                            remM, pMc);
        Value pMTile = arith::SelectOp::create(b, loc2, usePM, pMc, remM);

        Value zeroScalar = arith::ConstantOp::create(b, loc2, zeroAttr);
        Value accEmpty = tensor::EmptyOp::create(
            b, loc2, ArrayRef<int64_t>({ShapedType::kDynamic}), elt,
            ValueRange{pMTile});
        Value acc0 = linalg::FillOp::create(b, loc2, ValueRange{zeroScalar},
                                            ValueRange{accEmpty})
                         .getResult(0);

        SmallVector<Value> red = createNestedScfForLoops(
            b, loc2, ArrayRef<int64_t>{K}, ArrayRef<int64_t>{rK},
            ValueRange{acc0},
            [&](OpBuilder &b2, Location loc3, ValueRange kIvs,
                ValueRange accArgs) -> SmallVector<Value> {
              Value k = kIvs[0];

              Value remK = arith::SubIOp::create(b2, loc3, Kc, k);
              Value useR = arith::CmpIOp::create(
                  b2, loc3, arith::CmpIPredicate::ugt, remK, rKc);
              Value kTile = arith::SelectOp::create(b2, loc3, useR, rKc, remK);

              auto aTileTy = RankedTensorType::get(
                  {ShapedType::kDynamic, ShapedType::kDynamic}, elt);
              Value aTile = tensor::ExtractSliceOp::create(
                  b2, loc3, aTileTy, A, ValueRange{iM, k},
                  ValueRange{pMTile, kTile}, ValueRange{},

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),

                  ArrayRef<int64_t>(
                      {ShapedType::kDynamic, ShapedType::kDynamic}),
                  ArrayRef<int64_t>({1, 1}));

              auto xTileTy = RankedTensorType::get({ShapedType::kDynamic}, elt);
              Value xTile = tensor::ExtractSliceOp::create(
                  b2, loc3, xTileTy, x, ValueRange{k}, ValueRange{kTile},
                  ValueRange{}, ArrayRef<int64_t>({ShapedType::kDynamic}),
                  ArrayRef<int64_t>({ShapedType::kDynamic}),
                  ArrayRef<int64_t>({1}));

              auto yTileResTy =
                  RankedTensorType::get({ShapedType::kDynamic}, elt);
              auto gemv =
                  cinm::GemvOp::create(b2, loc3, aTile, xTile, accArgs[0]);
              cinm::markOpAsNoTile(gemv);
              auto mat = bufferization::MaterializeInDestinationOp::create(
                  b2, loc3, gemv.getResult(), accArgs[0]);
              Value updatedAcc = mat.getResult();
              return SmallVector<Value>{updatedAcc};
            });

        Value out = tensor::InsertSliceOp::create(
            b, loc2, red[0], iters[0], ValueRange{iM}, ValueRange{pMTile},
            ValueRange{}, ArrayRef<int64_t>({ShapedType::kDynamic}),
            ArrayRef<int64_t>({ShapedType::kDynamic}), ArrayRef<int64_t>({1}));

        return SmallVector<Value>{out};
      });

  return TilingResult2(results);
}

TilingResult2 ActivateOp::convertToTiledOps(OpBuilder &builder0,
                                            TilingParameters params) {
  ImplicitLocOpBuilder builder(getLoc(), builder0);
  auto inputT = getInput();
  auto inTy = inputT.getType();

  Type elt = inTy.getElementType();

  const ShapedType originalTy = inTy;
  Value originalShapeValue;
  if (inTy.getRank() > 1) {
    originalShapeValue = arith::ConstantOp::create(
        builder, RankedTensorType::get({inTy.getRank()}, builder.getI64Type()),
        builder.getI64TensorAttr(inTy.getShape()));
    inputT = reshapeStatic(builder, builder.getLoc(), inputT,
                           ArrayRef<int64_t>{inTy.getNumElements()});
    inTy = inputT.getType();
  }

  const int64_t total = inTy.getNumElements();
  int64_t p = 0;
  if (auto par = params.parallelClusterSize(total, 1))
    p = std::max<int64_t>(1, par->first);
  if (p <= 0)
    p = std::max<int64_t>(1, params.workingGroupSize());
  p = std::min<int64_t>(p, total);

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
        cinm::markOpAsNoTile(tile);

        Value out = tensor::InsertSliceOp::create(b, loc, tile.getResult(),
                                                  iters[0], off, siz, str);
        return {out};
      });

  if (originalTy.getRank() > 1) {
    finals[0] = tensor::ReshapeOp::create(builder, originalTy, finals[0],
                                          originalShapeValue);
  }
  return TilingResult2(finals);
}
