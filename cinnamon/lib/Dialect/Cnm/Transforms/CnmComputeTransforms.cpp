
#include <cinm-mlir/Dialect/Cnm/Transforms/CnmComputeTransforms.h>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

using namespace mlir;

namespace mlir::cnm {

LogicalResult expandWorkshoupDim(cnm::ComputeOp compute, uint64_t dim,
                                 int64_t factor) {
  auto wg = compute.getWorkgroupShape();
  if (dim >= wg.size() || wg[dim] % factor != 0)
    return failure();
  auto tile = wg[dim] / factor;
  SmallVector<int64_t> newShape(wg);
  newShape[dim] = factor;
  newShape.insert(newShape.begin() + dim + 1, tile);

  auto ctx = compute.getContext();
  auto d0 = getAffineDimExpr(dim, ctx);
  auto d1 = getAffineDimExpr(dim + 1, ctx);
  auto newIx = d0 * tile + d1;

  // Build an identity map that looks like (a,b,d0,d1,c) -> (a,b, d0 * tile +
  // d1, c)
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
  auto affineMaps = compute.getAffineMapsVec();
  for (auto &map : affineMaps) {
    map = map.compose(linearMap);
  }

  OpBuilder b(ctx);
  compute.setAffineMapsAttr(b.getAffineMapArrayAttr(affineMaps));
  compute.setWorkgroupShape(newShape);

  return success();
}

/// Turn the rightmost dimension of the workgroup into a
/// parallel loop within the kernel.
/// This transformation might delete the op if the workgroup
/// has only a single dimension.
/// ```
/// cnm.compute
///  ins(%a[(i,j)->(i*512+j)]: memref<1024xi32>)
///  outs(%o[(i,j)->(i*512+j)]: memref<1024xi32>)
///  on hierarchy<2x512>
///  do (%a1: memref<i32>, %o1: memref<i32>)  {
///       %x = memref.load %a1[]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[]
///   }
/// ```
/// into
/// ```
/// %as = memref.reshape %a: (mr<1024xi32>) ...
/// %os = memref.reshape %o: (mr<1024xi32>) ...
/// cnm.compute
///   ins(%as[(i)->(i)]: memref<2x512xi32>)
///   outs(%os[(i)->(i)]: memref<2x512xi32>)
///   on hierarchy<2>
///   do (%a1: memref<512xi32>,
///       %o1: memref<512xi32>)  {
///     affine.parallel (%i) = (0) to (512) {
///       %x = memref.load %a1[%i]
///       %t2 = arith.muli %x, 2
///       memref.store %t2, %o1[%i]
///     }
///   }
/// ```

static void remapUse(Operation *usage, BlockArgument newArg, Value indexVal,
                     OpBuilder &builder) {
  if (auto load = dyn_cast_or_null<memref::LoadOp>(usage)) {
    // logically this load does not verify
    load.getIndicesMutable().append(indexVal);
  }
  if (auto store = dyn_cast_or_null<memref::StoreOp>(usage)) {
    // logically this load does not verify
    store.getIndicesMutable().append(indexVal);
  }
}

void wrapLoopOverKernel(int loopSize, cnm::ComputeOp originalCompute,
                        ArrayRef<BlockArgument> originalArgs,
                        ArrayRef<BlockArgument> newArgs,
                        // affine maps that describe how the accesses to
                        // each original argument must be remapped to the
                        // new args
                        ArrayRef<AffineMap> argMap, Block &originalBlock,
                        Block &newBlock) {

  IRMapping mapping;
  for (auto [orig, newer] : llvm::zip(originalArgs, newArgs)) {
    mapping.map(orig, newer);
  }

  OpBuilder builder(originalCompute->getContext());
  builder.setInsertionPointToStart(&newBlock);
  auto loop = builder.create<affine::AffineParallelOp>(
      originalCompute->getLoc(), TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
      ArrayRef<int64_t>{loopSize});

  auto bodyBuilder = loop.getBodyBuilder();

  for (auto &op : originalBlock) {
    auto o = op.clone();
  }
}

// This transfo does peelright on a compute op where the
// buffer dims are already correct (do not need reshape)
LogicalResult peelRightPerfect(OpBuilder &builder, cnm::ComputeOp compute) {
  auto ctx = builder.getContext();
  auto wg = compute.getWorkgroupShape();
  assert(!wg.empty());
  if (wg.size() == 1) {
    return failure();
  }

  auto dimIx = wg.size() - 1;
  auto peelDim = wg[dimIx];

  SmallVector<AffineMap> newAffineMaps(compute.getAffineMapsVec());
  SmallVector<Type> newArguments(compute.getBody().getArgumentTypes());
  llvm::SmallBitVector changedArgs(compute.getBody().getNumArguments());

  for (auto [i, buf, arg, map] :
       llvm::enumerate(compute.getBuffers(), compute.getKernelArgs(),
                       compute.getAffineMapsVec())) {

    auto bufTy = buf.getType().cast<ShapedType>();
    auto bufShape = bufTy.getShape();
    auto argTy = arg.getType().cast<ShapedType>();
    auto argShape = argTy.getShape();

    auto scatterShape = bufTy.getShape().slice(0, bufTy.getShape().size() -
                                                      argTy.getShape().size());

    bool isBroadCast;
    if (map.getNumResults() > 0) {
      // all the first N-1 results have to not use the result.
      for (auto expr : map.getResults().slice(0, map.getNumResults() - 1)) {
        if (expr.isFunctionOfDim(dimIx))
          return failure();
      }
      auto lastRes = map.getResult(dimIx);
      if (lastRes.isFunctionOfDim(dimIx)) {
        if (lastRes != getAffineDimExpr(dimIx, ctx)) {
          // todo we can support non-identity last dims later
          return failure();
        }
        // drop last result
        map = map.dropResult(map.getNumResults() - 1);
      }
      isBroadCast = !lastRes.isFunctionOfDim(dimIx);
    } else {
      isBroadCast = true;
    }
    // drop last dim
    MutableAffineMap mut(map);
    mut.setNumDims(mut.getNumDims() - 1);
    newAffineMaps[i] = mut.getAffineMap();

    if (isBroadCast) {
      // This is a broadcast, argument is untouched
      continue;
    }

    SmallVector<int64_t> newShape;
    newShape.reserve(argShape.size() + 1);
    newShape.push_back(peelDim);
    newShape.append(argShape.begin(), argShape.end());
    newArguments[i] = MemRefType::get(newShape, argTy.getElementType());
    changedArgs.set(i);
  }

  // clone the region into a swap space to be able to perform modifications in
  // place
  std::unique_ptr<Region> swapRegion = std::make_unique<Region>();
  swapRegion->takeBody(compute.getBody());

  // update attributes
  compute.setWorkgroupShape(wg.slice(0, wg.size() - 1));
  compute.setAffineMapsAttr(builder.getAffineMapArrayAttr(newAffineMaps));

  // update types of kernel arguments
  auto &kernelRegion = compute.getBody();
  auto &entry = kernelRegion.emplaceBlock();
  IRMapping mapping;
  for (auto [newTy, arg] :
       llvm::zip(newArguments, swapRegion->getArguments())) {
    arg.setType(newTy);
    auto newArg = entry.addArgument(newTy, arg.getLoc());
    mapping.map(arg, newArg);
  }

  builder.setInsertionPointToStart(&kernelRegion.front());

  auto loop = builder.create<affine::AffineParallelOp>(
      compute->getLoc(), TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
      ArrayRef<int64_t>{peelDim});

  // clone the region, note that the output is invalid as the uses of the
  // original arguments that changed type must be adapted
  auto loopBuilder = loop.getBodyBuilder();
  loopBuilder.setListener(builder.getListener());

  auto &loopRegion = loop.getBodyRegion();
  swapRegion->cloneInto(&loopRegion, mapping);
  // now we need to cleanup after the clone
  loopRegion.getBlocks().pop_front();
  auto &loopBody = loopRegion.getBlocks().front();
  loopBody.addArgument(builder.getIndexType(), loop->getLoc());
  loopBody.getTerminator()->erase(); // that's the cnm.terminator
  builder.setInsertionPointToEnd(&loopBody);
  builder.create<affine::AffineYieldOp>(loop.getLoc());
  builder.setInsertionPointToEnd(loop->getBlock());
  builder.create<cnm::TerminatorOp>(loop.getLoc());

  for (auto changed : changedArgs.set_bits()) {
    auto arg = kernelRegion.getArgument(changed);
    for (auto user : arg.getUsers()) {
      remapUse(user, arg, loopBody.getArgument(0), builder);
    }
  }

  return success();
}

LogicalResult peelRight(OpBuilder &builder, cnm::ComputeOp compute) {
  return peelRightPerfect(builder, compute);
}

} // namespace mlir::cnm