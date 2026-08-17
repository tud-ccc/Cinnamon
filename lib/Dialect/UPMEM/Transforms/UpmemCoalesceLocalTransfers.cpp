//===- UpmemCoalesceLocalTransfers.cpp - Fewer, larger staging transfers
//---===//
//
// Rewrites the `cnm.local_transfer`s a launch body issues inside a loop into
// fewer and larger ones: hoisting a transfer whose tile does not change with
// the loop, and coalescing a run of adjacent tiles into a single transfer.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <numeric>

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMCOALESCELOCALTRANSFERSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {

/// One operand's staging inside a loop: the tile it moves, the buffer it moves
/// through, and the transfers doing the moving.
///
/// Read and write are two ops around the same buffer, and either may be
/// absent -- an input is only read in, an output only written back, an
/// accumulator both.
struct Staging {
  memref::SubViewOp tile;    // the far-level tile, addressed per iteration
  memref::AllocaOp buffer;   // the leaf-level buffer it is staged through
  cnm::LocalTransferOp read; // far -> leaf
  cnm::LocalTransferOp write;
};

bool definedOutside(ArrayRef<OpFoldResult> range, affine::AffineForOp loop) {
  return llvm::all_of(range, [&](OpFoldResult ofr) {
    auto v = dyn_cast<Value>(ofr);
    // `isAncestor` and not `isProperAncestor`: the induction variable is a
    // block argument of the loop's own region, so the loop is the parent op
    // of its defining block and has to count as containing it.
    return !v || !loop->isAncestor(v.getParentBlock()->getParentOp());
  });
}

bool definedOutside(ValueRange range, affine::AffineForOp loop) {
  SmallVector<OpFoldResult> ofrs(range.begin(), range.end());
  return definedOutside(ofrs, loop);
}

/// The stagings `loop` performs in its own body.
///
/// Only the immediate body: a transfer nested deeper belongs to the loop it
/// sits in, and is reached when the walk gets there. The buffer has to be this
/// body's own, since the point of the rewrite is to move the transfers away
/// from the ops that use it.
SmallVector<Staging> collectStagings(affine::AffineForOp loop) {
  llvm::MapVector<Operation *, Staging> byBuffer;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto transfer = dyn_cast<cnm::LocalTransferOp>(&op);
    if (!transfer)
      continue;
    // Which side is the leaf buffer is implied by the operands: the one this
    // body allocated. A transfer between two buffers it did not allocate is
    // not a staging of the kind this pass moves.
    Value src = transfer.getSource(), dst = transfer.getTarget();
    auto srcAlloca = src.getDefiningOp<memref::AllocaOp>();
    auto buffer = srcAlloca ? srcAlloca : dst.getDefiningOp<memref::AllocaOp>();
    if (!buffer || buffer->getParentOp() != loop.getOperation())
      continue;
    auto tile = (srcAlloca ? dst : src).getDefiningOp<memref::SubViewOp>();
    if (!tile)
      continue;
    Staging &staging = byBuffer[buffer.getOperation()];
    staging.buffer = buffer;
    staging.tile = tile;
    (srcAlloca ? staging.write : staging.read) = transfer;
  }

  SmallVector<Staging> stagings;
  for (auto &entry : byBuffer) {
    Staging staging = entry.second;
    // Both transfers must move the same tile, or moving them apart would move
    // two different ones.
    if (staging.read && staging.write &&
        staging.read.getSource() != staging.write.getTarget())
      continue;
    stagings.push_back(staging);
  }
  return stagings;
}

/// Hoist a staging whose tile does not depend on the loop.
///
/// The same bytes are read on every trip and written back unchanged in
/// between, so one read before the loop and one write after it move exactly
/// the same data. This is what makes an output tile cheap under a reduction:
/// the reduction updates it every trip, and only the final value has to reach
/// the far level.
///
/// Read and write must both be present. A tile only read is already hoistable
/// by loop-invariant code motion, and one only written would need the loop to
/// write every byte of it -- which the pass cannot see from here.
LogicalResult hoistInvariant(IRRewriter &rewriter, affine::AffineForOp loop,
                             Staging staging) {
  if (!staging.read || !staging.write)
    return failure();
  if (!definedOutside(staging.tile->getOperands(), loop))
    return failure();

  rewriter.moveOpBefore(staging.tile, loop);
  rewriter.moveOpBefore(staging.buffer, loop);
  rewriter.moveOpBefore(staging.read, loop);
  rewriter.moveOpAfter(staging.write, loop);
  return success();
}

/// The dimension of `tile` addressed by `loop`'s induction variable, when the
/// tile is one step along it and is otherwise fixed. Consecutive trips then
/// address adjacent tiles, which is what lets a run of them move together.
std::optional<unsigned> stripDim(memref::SubViewOp tile,
                                 affine::AffineForOp loop) {
  if (!llvm::all_of(tile.getMixedStrides(),
                    [](OpFoldResult s) { return isConstantIntValue(s, 1); }))
    return std::nullopt;
  // The widened tile is built outside the loop, so the memref it views has to
  // be reachable from there.
  if (!definedOutside(ValueRange{tile.getSource()}, loop))
    return std::nullopt;

  SmallVector<OpFoldResult> offsets = tile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = tile.getMixedSizes();
  std::optional<unsigned> found;
  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    if (dyn_cast<Value>(offset) != loop.getInductionVar())
      continue;
    if (found)
      return std::nullopt; // moves along two dimensions at once
    if (!isConstantIntValue(sizes[dim], 1))
      return std::nullopt; // steps of one tile are what makes them adjacent
    found = dim;
  }
  if (!found)
    return std::nullopt;

  // Everything else has to hold still, or the tiles are not adjacent.
  for (auto [dim, offset] : llvm::enumerate(offsets))
    if (dim != *found && !definedOutside({offset}, loop))
      return std::nullopt;
  return definedOutside(sizes, loop) ? found : std::nullopt;
}

/// How many tiles must move together for every transfer to start on a granule
/// boundary; 1 when they already do.
///
/// A far-level buffer is indexed per tile, so consecutive transfers start one
/// tile apart. When a tile is a fraction of a granule every other transfer
/// begins mid-granule, and an MRAM DMA drops the low bits of such an address:
/// it moves the right bytes to the wrong place rather than failing.
int64_t coalescingFactor(int64_t tileElems, int64_t granuleBits,
                         int64_t eltBits) {
  if (tileElems <= 0 || granuleBits <= 0 || eltBits <= 0)
    return 1;
  const int64_t elemsPerGranule = granuleBits / std::gcd(granuleBits, eltBits);
  return elemsPerGranule / std::gcd(tileElems, elemsPerGranule);
}

/// A staging the loop walks one tile at a time, and the dimension of its tile
/// the induction variable addresses.
struct Strip {
  Staging staging;
  unsigned dim;
};

/// Move `k` adjacent tiles per transfer instead of one, for every staging in
/// `strips`.
///
/// The loop is split in two -- an outer loop stepping `k` tiles, and the
/// original walking the strip -- and the stagings move to the outer body,
/// each widened to the whole strip. The inner body works on a slice of each
/// wider buffer, which is the only change it sees.
///
/// All of the loop's stagings are widened together because the split replaces
/// the loop: one left behind would sit in a loop whose bounds are no longer
/// constant, and could never be widened afterwards.
///
/// This costs leaf-level space: each buffer grows from one tile to `k`.
/// Bounded, and deliberately so -- `k` is the least strip length that puts
/// every one of these stagings on a granule boundary, which is at most one
/// granule's worth of elements, and the tiles the search chose stay the tiles
/// the kernel computes on.
LogicalResult coalesceRun(IRRewriter &rewriter, affine::AffineForOp loop,
                          ArrayRef<Strip> strips, int64_t k) {
  const int64_t lb = loop.getConstantLowerBound();
  const int64_t ub = loop.getConstantUpperBound();
  // A partial last strip would transfer tiles the loop never visits, writing
  // back whatever the buffer happened to hold for them.
  if ((ub - lb) % k != 0)
    return failure();
  // Checked for every strip before anything is rewritten: the split is one
  // transformation, and a failure partway through would leave the loop torn.
  for (Strip s : strips)
    if (s.dim >=
        unsigned(cast<MemRefType>(s.staging.buffer.getType()).getRank()))
      return failure();

  Location loc = loop.getLoc();
  rewriter.setInsertionPoint(loop);
  // The outer loop counts strips rather than stepping `k` tiles at a time, so
  // that a tile offset reads as `strip * k` and is a whole number of granules
  // in the expression itself. Stepping by `k` would leave `strip * 1` scaled
  // by the element size, and the emitter's alignment check is syntactic: it
  // cannot see that the loop skips the odd values.
  auto outer = affine::AffineForOp::create(rewriter, loc, 0, (ub - lb) / k);
  AffineExpr strip = rewriter.getAffineDimExpr(0);
  AffineMap stripStartMap = AffineMap::get(1, 0, lb + strip * k);

  rewriter.setInsertionPointToStart(outer.getBody());
  auto stripStart = affine::AffineApplyOp::create(
      rewriter, loc, stripStartMap, ValueRange{outer.getInductionVar()});

  SmallVector<memref::AllocaOp> stripBuffers;
  SmallVector<memref::SubViewOp> stripTiles;
  for (Strip s : strips) {
    auto bufferType = cast<MemRefType>(s.staging.buffer.getType());
    SmallVector<int64_t> stripShape(bufferType.getShape());
    stripShape[s.dim] *= k;
    stripBuffers.push_back(memref::AllocaOp::create(
        rewriter, loc,
        MemRefType::get(stripShape, bufferType.getElementType(),
                        MemRefLayoutAttrInterface{},
                        bufferType.getMemorySpace())));

    SmallVector<OpFoldResult> offsets = s.staging.tile.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = s.staging.tile.getMixedSizes();
    SmallVector<OpFoldResult> strides = s.staging.tile.getMixedStrides();
    offsets[s.dim] = stripStart.getResult();
    sizes[s.dim] = rewriter.getIndexAttr(k);
    stripTiles.push_back(memref::SubViewOp::create(
        rewriter, loc, s.staging.tile.getSource(), offsets, sizes, strides));

    if (s.staging.read)
      cnm::LocalTransferOp::create(rewriter, loc, stripTiles.back().getResult(),
                                   stripBuffers.back().getResult());
  }

  // The original loop becomes the walk over the strip.
  rewriter.moveOpBefore(loop, outer.getBody()->getTerminator());
  loop.setLowerBound(outer.getInductionVar(), stripStartMap);
  loop.setUpperBound(outer.getInductionVar(),
                     AffineMap::get(1, 0, lb + strip * k + k));

  rewriter.setInsertionPointAfter(loop);
  for (size_t i = 0; i < strips.size(); ++i)
    if (strips[i].staging.write)
      cnm::LocalTransferOp::create(rewriter, loc, stripBuffers[i].getResult(),
                                   stripTiles[i].getResult());

  // Inside, the body works on this trip's slice of each strip.
  rewriter.setInsertionPointToStart(loop.getBody());
  auto within = affine::AffineApplyOp::create(
      rewriter, loc,
      AffineMap::get(2, 0,
                     rewriter.getAffineDimExpr(0) - lb -
                         rewriter.getAffineDimExpr(1) * k),
      ValueRange{loop.getInductionVar(), outer.getInductionVar()});
  for (size_t i = 0; i < strips.size(); ++i) {
    Strip s = strips[i];
    auto bufferType = cast<MemRefType>(s.staging.buffer.getType());
    SmallVector<OpFoldResult> sliceOffsets(bufferType.getRank(),
                                           rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sliceStrides(bufferType.getRank(),
                                           rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sliceSizes;
    for (int64_t extent : bufferType.getShape())
      sliceSizes.push_back(rewriter.getIndexAttr(extent));
    sliceOffsets[s.dim] = within.getResult();
    // The result type is inferred rather than the buffer's: this trip's slice
    // starts at a dynamic offset into the strip, which the layout has to say.
    auto slice =
        memref::SubViewOp::create(rewriter, loc, stripBuffers[i].getResult(),
                                  sliceOffsets, sliceSizes, sliceStrides);

    if (s.staging.read)
      rewriter.eraseOp(s.staging.read);
    if (s.staging.write)
      rewriter.eraseOp(s.staging.write);
    rewriter.replaceAllUsesWith(s.staging.buffer.getResult(),
                                slice.getResult());
    rewriter.eraseOp(s.staging.buffer);
    if (s.staging.tile->use_empty())
      rewriter.eraseOp(s.staging.tile);
  }
  return success();
}

struct UpmemCoalesceLocalTransfersPass
    : public impl::UpmemCoalesceLocalTransfersPassBase<
          UpmemCoalesceLocalTransfersPass> {
  using Base::Base;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    getOperation()->walk(
        [&](cnm::LaunchOp launch) { runOnLaunch(rewriter, launch); });
  }

  void runOnLaunch(IRRewriter &rewriter, cnm::LaunchOp launch) {
    // The granule is a property of the level a tile is staged from. A launch
    // body stages from whatever level its operands live in, so the strictest
    // alignment the platform declares is the one every transfer must meet.
    auto platform = launch.getWg().getType().getAccelerator().getPlatform();
    if (!platform)
      return;
    int64_t granuleBits = 0;
    for (cinm::CinmLevelDefAttr level : platform.getLevels())
      granuleBits = std::max(granuleBits, level.getAlignment() * 8);

    // Each rewrite invalidates the walk, and hoisting a staging out of an
    // inner loop can make it invariant in the loop outside, so this runs to a
    // fixed point rather than in one sweep.
    bool changed = true;
    while (changed) {
      changed = false;
      launch.walk<WalkOrder::PreOrder>([&](affine::AffineForOp loop) {
        SmallVector<Staging> stagings = collectStagings(loop);
        for (Staging staging : stagings) {
          if (succeeded(hoistInvariant(rewriter, loop, staging))) {
            changed = true;
            return WalkResult::interrupt();
          }
        }
        if (!loop.hasConstantBounds() || loop.getStepAsInt() != 1)
          return WalkResult::advance();

        // Every fractional staging of this loop is widened in the same
        // rewrite. The split replaces the loop by a strip loop and a walk with
        // non-constant bounds, so a staging left behind would never come back
        // round: `k` is the least strip length that reaches a granule for all
        // of them at once.
        SmallVector<Strip> strips;
        int64_t k = 1;
        for (Staging staging : stagings) {
          std::optional<unsigned> dim = stripDim(staging.tile, loop);
          if (!dim)
            continue;
          auto bufferType = cast<MemRefType>(staging.buffer.getType());
          if (!bufferType.hasStaticShape())
            continue;
          int64_t factor =
              coalescingFactor(bufferType.getNumElements(), granuleBits,
                               bufferType.getElementTypeBitWidth());
          if (factor <= 1)
            continue;
          strips.push_back({staging, *dim});
          k = std::lcm(k, factor);
        }
        if (!strips.empty() &&
            succeeded(coalesceRun(rewriter, loop, strips, k))) {
          changed = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
  }
};

} // namespace
} // namespace mlir::upmem
