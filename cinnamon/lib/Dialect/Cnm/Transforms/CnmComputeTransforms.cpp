
#include <cinm-mlir/Dialect/Cnm/Transforms/CnmComputeTransforms.h>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;

namespace mlir::cnm {

FailureOr<cnm::ComputeOp> expandWorkshoupDim(OpBuilder &builder,
                                             cnm::ComputeOp compute,
                                             uint64_t dim, int64_t factor) {
  auto wg = compute.getWorkgroupShape();
  if (dim >= wg.size() || wg[dim] % factor != 0)
    return failure();
  auto tile = wg[dim] / factor;
  SmallVector<int64_t> newShape(wg);
  newShape[dim] = factor;
  newShape.insert(newShape.begin() + dim + 1, tile);

  auto ctx = builder.getContext();
  auto d0 = getAffineDimExpr(dim, ctx);
  auto d1 = getAffineDimExpr(dim + 1, ctx);
  auto newIx = d0 * tile + d1;

  // Build an identity map that looks like (a,b,d0,d1,c) -> (a,b, d0 * tile + d1, c)
  SmallVector<AffineExpr> exprs;
  auto offset = 0;
  for (uint64_t i = 0; i < wg.size(); i++) {
    if (i == dim) {
      exprs.push_back(newIx);
      offset = 1;
    } else {
      exprs.push_back(getAffineDimExpr(offset + i, ctx));
    }
  }
  AffineMap linearMap = AffineMap::get(newShape.size(), 0, exprs, ctx);

  // apply the linear map first, then the original map
  SmallVector<AffineMap> affineMaps(
      compute.getAffineMaps().getAsValueRange<AffineMapAttr>());
  for (auto &map : affineMaps) {
    map = map.compose(linearMap);
  }

  auto result = builder.create<cnm::ComputeOp>(
      compute->getLoc(), newShape, compute.getInBuffers(),
      compute.getOutBuffers(), affineMaps);
  return success(result);
}

} // namespace mlir::cnm