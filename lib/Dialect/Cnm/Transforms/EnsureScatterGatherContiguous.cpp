//===- EnsureScatterGatherContiguous.cpp - Pack non-contiguous transfers -===//
//
// Lowering cnm.scatter/cnm.gather to upmem.scatter/upmem.gather performs a
// single flat memcpy per DPU. This is only correct if the transferred
// elements are actually contiguous in the host memref (see
// upmem::ScatterOnArrayOp::verify / upmem::GatherFromArrayOp::verify and the
// runtime's do_dpu_transfer). This pass detects host memrefs that aren't
// contiguous (e.g. subviews of a larger tensor) and inserts an intermediate
// contiguous buffer, similar to what packATile does for the tiled
// GEMV/reduction templates, but generically for any cnm.scatter/cnm.gather.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/Cinm/IR/CinmBase.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmUtils.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmScatterMap.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMENSURESCATTERGATHERCONTIGUOUSPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

namespace {

// Allocate a memref with the same shape/element type as `src` but with a
// default (fully packed, row-major) layout, resolving any dynamic dimensions
// from `src` itself.
Value allocateContiguousLike(OpBuilder &b, Location loc, Value src) {
  auto ty = cast<MemRefType>(src.getType());
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < ty.getRank(); ++i)
    if (ty.isDynamicDim(i))
      dynSizes.push_back(memref::DimOp::create(b, loc, src, i));

  auto contiguousTy = MemRefType::get(ty.getShape(), ty.getElementType());
  return memref::AllocOp::create(b, loc, contiguousTy, dynSizes);
}

// `value` is contiguous enough for a flat per-DPU memcpy iff its whole shape
// (not just the per-workgroup-element buffer suffix) is packed row-major.
bool isFullyContiguous(TypedValue<ShapedType> value) {
  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (!memrefTy)
    return true; // not yet bufferized; nothing we can (or need to) do here.
  return mlir::scatteredMemrefIsContiguous(value, memrefTy.getShape());
}

/// Records that a repack moves data that is the same on every inference.
///
/// The tag goes on the repack, which is what decides how it is timed, and on
/// the allocation it fills, which is what makes the conclusion reachable
/// afterwards: whoever later asks whether the packed buffer is static (the
/// backend, deciding how to time the transfer *out* of it) sees only a
/// memref.alloc, and an allocation says nothing about its contents.
void markStatic(Operation *repack, Value packed, OpBuilder &b) {
  repack->setAttr(cinm::CinmDialect::STATIC_ATTR_NAME, b.getUnitAttr());
  packed.getDefiningOp()->setAttr(cinm::CinmDialect::STATIC_ATTR_NAME,
                                  b.getUnitAttr());
}

/// The constants `map` divides each of its dimensions by, which is what has to
/// fall on a dimension boundary for the map to be linear in them.
///
/// Fails when a floordiv or mod is applied to anything but a bare dimension:
/// no split makes that linear.
LogicalResult
collectSplitPoints(AffineMap map,
                   SmallVectorImpl<SmallVector<int64_t>> &points) {
  points.assign(map.getNumDims(), {});
  bool supported = true;
  for (AffineExpr result : map.getResults())
    result.walk([&](AffineExpr e) {
      auto binary = dyn_cast<AffineBinaryOpExpr>(e);
      if (!binary)
        return;
      AffineExprKind kind = binary.getKind();
      if (kind != AffineExprKind::FloorDiv && kind != AffineExprKind::Mod &&
          kind != AffineExprKind::CeilDiv)
        return;
      auto dim = dyn_cast<AffineDimExpr>(binary.getLHS());
      auto constant = dyn_cast<AffineConstantExpr>(binary.getRHS());
      if (!dim || !constant || kind == AffineExprKind::CeilDiv) {
        supported = false;
        return;
      }
      SmallVector<int64_t> &forDim = points[dim.getPosition()];
      if (!llvm::is_contained(forDim, constant.getValue()))
        forDim.push_back(constant.getValue());
    });
  return success(supported);
}

/// The extents a dimension of `extent` is split into so that each of `points`
/// falls on a boundary: `extent/c0, c0/c1, ..., ck`. Empty when nothing needs
/// splitting; fails when a point does not divide what encloses it.
FailureOr<SmallVector<int64_t>> splitBasis(int64_t extent,
                                           SmallVector<int64_t> points) {
  llvm::sort(points, std::greater<int64_t>());
  SmallVector<int64_t> basis;
  int64_t inner = extent;
  for (int64_t point : points) {
    // A point at or beyond the extent introduces no boundary: the quotient is
    // constant over the whole dimension and has already been folded away.
    if (point <= 0 || point >= inner)
      continue;
    if (inner % point != 0)
      return failure();
    basis.push_back(inner / point);
    inner = point;
  }
  if (basis.empty())
    return SmallVector<int64_t>{};
  basis.push_back(inner);
  return basis;
}

/// Reorders `op`'s host value into workgroup x buffer order, so that each
/// leaf's share becomes one whole-buffer block instead of several. Returns
/// false when no repack it could emit is expressible, leaving `op` alone.
///
/// The packed value's index space is the scatter's own -- workgroup
/// coordinates then buffer coordinates -- except that a dimension the map
/// divides is split at the divisor. The repack is a strided copy, which walks
/// its target with one constant stride per dimension, and `d floordiv c` is
/// not that: the source offset jumps every c steps of d. Making c a boundary
/// turns the jump into a stride of its own. The scatter is then left naming
/// nothing but the leaf it addresses, delinearized over those splits.
template <class Op>
bool packIntoOneBlockPerLeaf(Op op, OpBuilder &b, bool isStatic) {
  constexpr bool isScatter = std::is_same_v<Op, cnm::ScatterOp>;
  Location loc = op.getLoc();
  MLIRContext *ctx = b.getContext();
  auto hostTy = cast<MemRefType>(op.getHostValue().getType());
  cnm::BufferType bufferTy = op.getBuffer().getType();

  AffineMap map =
      cnm::inflateScatterMapToPointwise(op.getScatterMap(), bufferTy);
  SmallVector<int64_t> indexSpace = cnm::getScatterIndexSpace(bufferTy);

  SmallVector<SmallVector<int64_t>> points;
  if (failed(collectSplitPoints(map, points)))
    return false;
  SmallVector<SmallVector<int64_t>> bases;
  for (auto [extent, forDim] : llvm::zip_equal(indexSpace, points)) {
    FailureOr<SmallVector<int64_t>> basis = splitBasis(extent, forDim);
    if (failed(basis))
      return false;
    bases.push_back(*basis);
  }

  // The packed shape, and each old dimension written in terms of the new ones
  // it became (for the repack's map) and the reverse (for the scatter's).
  SmallVector<int64_t> packedShape;
  SmallVector<AffineExpr> toNew;
  SmallVector<AffineExpr> toOld;
  for (auto [old, extent, basis] : llvm::enumerate(indexSpace, bases)) {
    AffineExpr oldDim = getAffineDimExpr(old, ctx);
    if (basis.empty()) {
      toNew.push_back(getAffineDimExpr(packedShape.size(), ctx));
      toOld.push_back(oldDim);
      packedShape.push_back(extent);
      continue;
    }
    SmallVector<int64_t> strides = mlir::computeSuffixProduct(basis);
    SmallVector<AffineExpr> newDims;
    for (int64_t split : basis) {
      newDims.push_back(getAffineDimExpr(packedShape.size(), ctx));
      packedShape.push_back(split);
    }
    toNew.push_back(mlir::linearize(ctx, newDims, strides));
    llvm::append_range(toOld, mlir::delinearize(oldDim, strides));
  }

  AffineMap packedMap = simplifyAffineMapWithBounds(
      map.replaceDimsAndSymbols(toNew, {}, packedShape.size(), 0), packedShape);

  b.setInsertionPoint(op);
  Value host = op.getHostValue();
  Value packed = memref::AllocOp::create(
      b, loc, MemRefType::get(packedShape, hostTy.getElementType()));

  // A scatter reads the host value, so the repack fills the packed buffer
  // before the transfer. A gather writes it, so the packed buffer is what the
  // transfer fills and the repack writes it back out afterwards -- the same
  // map, walked in the other direction, which is why it is a different op and
  // not this one with its operands exchanged.
  if constexpr (isScatter) {
    auto compact =
        cnm::CompactBufferOp::create(b, loc, host, packed, packedMap);
    if (isStatic)
      markStatic(compact, packed, b);
  } else {
    b.setInsertionPointAfter(op);
    auto expand = cnm::ExpandBufferOp::create(b, loc, packed, host, packedMap);
    if (isStatic)
      markStatic(expand, packed, b);
  }

  op.getHostValueMutable().assign(packed);
  op.setScatterMap(simplifyAffineMapWithBounds(
      AffineMap::get(indexSpace.size(), 0, toOld, ctx), indexSpace));
  return true;
}

/// Whether a leaf's share arrives as several blocks rather than one.
///
/// The stored map is pointwise and names every element its own block, so this
/// asks about the widest block that map and the host layout allow -- the same
/// derivation the backend will make.
template <class Op> bool isFragmented(Op op) {
  cnm::BufferType bufferTy = op.getBuffer().getType();
  return cnm::getScatterBlocksPerLeaf(cnm::deflateScatterMap(op.getScatterMap(),
                                                             bufferTy,
                                                             op.getHostType()),
                                      bufferTy) > 1;
}

void ensureScatterContiguous(cnm::ScatterOp op, OpBuilder &b, bool staticOnly) {
  auto input = op.getInput();
  if (!isa<MemRefType>(input.getType()) || isFullyContiguous(input))
    return;
  bool isStatic = cinm::isStaticValue(input);
  // Packing a per-inference operand costs a copy on every call, which is
  // usually worse than letting the backend move it as several blocks per DPU.
  if (staticOnly && !isStatic)
    return;

  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  Value packed = allocateContiguousLike(b, loc, input);
  // The packed buffer has the input's shape, so each of its elements comes
  // from the same index of the input: only the layout changes.
  auto rank = cast<MemRefType>(input.getType()).getRank();
  auto compact = cnm::CompactBufferOp::create(
      b, loc, input, packed,
      AffineMap::getMultiDimIdentityMap(rank, b.getContext()));
  if (isStatic)
    markStatic(compact, packed, b);
  op.getInputMutable().assign(packed);

  // b.setInsertionPointAfter(op);
  // memref::DeallocOp::create(b, loc, packed);
}

// The gather side stays a memref.copy: it writes a packed buffer back out to
// a strided one, and cnm.compact_buffer only describes the packing direction.
// That copy has a strided target, so MemRefToLLVM routes it through the
// instrumented memrefCopy anyway and it is not lost from the accounting.
void ensureGatherContiguous(cnm::GatherOp op, OpBuilder &b, bool staticOnly) {
  auto outputBuf = op.getOutputBuf();
  if (!isa<MemRefType>(outputBuf.getType()) || isFullyContiguous(outputBuf))
    return;
  // A gather destination is written every inference by definition, so there
  // is nothing here that could amortize.
  if (staticOnly)
    return;

  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  Value packed = allocateContiguousLike(b, loc, outputBuf);
  op.getOutputBufMutable().assign(packed);

  b.setInsertionPointAfter(op);
  memref::CopyOp::create(b, loc, packed, outputBuf);
  memref::DeallocOp::create(b, loc, packed);
}

// A leaf's tile is addressed element by element when it leaves
// `--convert-linalg-to-cnm`. Turn as much of it as possible into whole blocks,
// which is what a DMA moves: a trailing buffer dimension can be left implicit
// when the map indexes its host dimension by nothing but that buffer
// dimension, the host dimension is exactly as long -- making the pair one
// whole sub-array -- and the host value stores that sub-array contiguously.
//
// When the host dimension is longer but a whole multiple, the tile is still a
// contiguous run; it just does not line up with a dimension boundary. Splitting
// the host dimension makes it line up, and costs nothing: an expand_shape of a
// contiguous memref is a view.
struct BlockForm {
  int64_t implicit = 0;
  // Host dimensions to split, innermost first, as (position, inner extent).
  SmallVector<std::pair<unsigned, int64_t>> splits;
  SmallVector<AffineExpr> results;
  SmallVector<int64_t> hostShape;
};

// Whether `expr` is `q * factor + dim`, and `q` if so.
std::optional<AffineExpr> matchScaledPlusDim(AffineExpr expr, unsigned dim,
                                             int64_t factor, unsigned numDims) {
  MLIRContext *ctx = expr.getContext();
  SmallVector<AffineExpr> substitution;
  for (unsigned i = 0; i < numDims; ++i)
    substitution.push_back(i == dim ? getAffineConstantExpr(0, ctx)
                                    : getAffineDimExpr(i, ctx));
  AffineExpr base = expr.replaceDims(substitution);
  if (simplifyAffineExpr(base + getAffineDimExpr(dim, ctx) - expr, numDims,
                         0) != getAffineConstantExpr(0, ctx))
    return std::nullopt;

  // The run starts at a multiple of the block, or splitting would cut it.
  AffineExpr quotient = base.floorDiv(factor);
  if (simplifyAffineExpr(quotient * factor - base, numDims, 0) !=
      getAffineConstantExpr(0, ctx))
    return std::nullopt;
  return quotient;
}

BlockForm computeBlockForm(AffineMap map, cnm::BufferType bufferTy,
                           ShapedType hostTy) {
  BlockForm form;
  form.results.assign(map.getResults().begin(), map.getResults().end());
  // The results describe the host dimensions the map still names; the ones it
  // leaves implicit are already part of the block.
  form.hostShape.assign(hostTy.getShape().begin(), hostTy.getShape().end());
  form.hostShape.truncate(form.results.size());
  ArrayRef<int64_t> bufShape = bufferTy.getShape();
  unsigned wgRank = bufferTy.getWorkgroupShape().size();

  // A memref may store a sub-array with gaps in it; a tensor may not.
  int64_t contiguous = std::numeric_limits<int64_t>::max();
  if (auto memrefTy = dyn_cast<MemRefType>(hostTy)) {
    contiguous = getContiguousSuffixSize(memrefTy);
    if (contiguous < 0)
      return form;
  }

  // Pick up wherever the map already is, so this is idempotent.
  form.implicit = cnm::getNumImplicitHostDims(map, bufferTy);
  int64_t blockElements =
      computeProduct(cnm::getScatterBlockShape(map, bufferTy));
  while (form.implicit < static_cast<int64_t>(bufShape.size())) {
    if (form.results.empty())
      break;
    unsigned position = form.results.size() - 1;
    unsigned bufDim = bufShape.size() - 1 - form.implicit;
    unsigned mapDim = wgRank + bufDim;
    int64_t extent = bufShape[bufDim];
    if (blockElements * extent > contiguous)
      break;

    AffineExpr expr = form.results[position];
    if (expr == getAffineDimExpr(mapDim, map.getContext()) &&
        form.hostShape[position] == extent) {
      form.results.pop_back();
      form.hostShape.erase(form.hostShape.begin() + position);
    } else if (form.hostShape[position] % extent == 0) {
      std::optional<AffineExpr> outer =
          matchScaledPlusDim(expr, mapDim, extent, map.getNumDims());
      if (!outer)
        break;
      form.splits.push_back({position, extent});
      form.results[position] = *outer;
      form.hostShape[position] /= extent;
    } else {
      break;
    }
    blockElements *= extent;
    ++form.implicit;
  }
  return form;
}

// Split `host`'s dimensions as `splits` asks, innermost first.
Value expandHost(OpBuilder &b, Location loc, Value host,
                 ArrayRef<std::pair<unsigned, int64_t>> splits) {
  for (auto [position, inner] : splits) {
    auto type = cast<MemRefType>(host.getType());
    SmallVector<int64_t> shape(type.getShape());
    shape[position] /= inner;
    shape.insert(shape.begin() + position + 1, inner);

    SmallVector<ReassociationIndices> reassociation;
    for (unsigned i = 0, out = 0; i < type.getRank(); ++i) {
      reassociation.push_back({out++});
      if (i == position)
        reassociation.back().push_back(out++);
    }
    host = memref::ExpandShapeOp::create(b, loc, shape, host, reassociation);
  }
  return host;
}

/// Reshapes the host value so that the widest block the map allows lines up
/// with a dimension boundary.
///
/// The map itself is left alone. A block is one run of memory, so what block a
/// transfer moves depends on the host layout as much as on the map -- it is a
/// derivation, and each backend makes it for itself (cnm::deflateScatterMap).
/// Recording it on the op as a shorthand map would be a second description of
/// the same transfer, which then has to be kept in agreement with the first.
/// What is *not* derivable is this reshape: it is a real change to the IR, and
/// making it here keeps it out of every backend that moves blocks.
template <class Op> void alignHostToBlocks(Op op, OpBuilder &b) {
  Value host = op.getHostValue();
  // Splitting rewrites the value, which only works once it is a memref.
  if (!isa<MemRefType>(host.getType()))
    return;
  BlockForm form = computeBlockForm(op.getScatterMap(),
                                    op.getBuffer().getType(), op.getHostType());
  if (form.splits.empty())
    return;

  b.setInsertionPoint(op);
  Value expanded = expandHost(b, op.getLoc(), host, form.splits);
  op.getHostValueMutable().assign(expanded);

  // The map indexed the host before the split and has to keep naming one index
  // per host dimension. Splitting dimension p into (outer, inner) splits its
  // index the same way, which follows from the reshape alone -- no need to
  // reuse how the block form arrived at the split. Positions decrease along
  // `splits`, so an insertion never moves one still to come.
  AffineMap map = op.getScatterMap();
  SmallVector<AffineExpr> results(map.getResults());
  for (auto [position, inner] : form.splits) {
    AffineExpr index = results[position];
    results[position] = index.floorDiv(inner);
    results.insert(results.begin() + position + 1, index % inner);
  }
  op.setScatterMap(simplifyAffineMapWithBounds(
      AffineMap::get(map.getNumDims(), map.getNumSymbols(), results,
                     map.getContext()),
      cnm::getScatterMapDomain(map, op.getBuffer().getType())));
}

} // namespace

struct CnmEnsureScatterGatherContiguousPass
    : public cnm::impl::CnmEnsureScatterGatherContiguousPassBase<
          CnmEnsureScatterGatherContiguousPass> {
  using Base::Base;

  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](Operation *op) {
      if (auto scatter = dyn_cast<cnm::ScatterOp>(op)) {
        ensureScatterContiguous(scatter, builder, staticOnly);
        alignHostToBlocks(scatter, builder);
        // Now that the widest block form is reachable, a transfer that still
        // needs several blocks per leaf can be traded for one repack.
        if (packFragmented && isFragmented(scatter) &&
            isa<MemRefType>(scatter.getInput().getType()) &&
            (!staticOnly || cinm::isStaticValue(scatter.getInput())))
          packIntoOneBlockPerLeaf(scatter, builder,
                                  cinm::isStaticValue(scatter.getInput()));
      } else if (auto gather = dyn_cast<cnm::GatherOp>(op)) {
        ensureGatherContiguous(gather, builder, staticOnly);
        alignHostToBlocks(gather, builder);
        // A gather's destination is written every inference, so there is
        // nothing here that could amortize and `staticOnly` would rule the
        // repack out entirely. It is offered anyway when packing is on: a
        // fragmented gather is the one case the backend has no correct
        // transfer for, so the repack is not a trade but the only way.
        if (packFragmented && isFragmented(gather) &&
            isa<MemRefType>(gather.getOutputBuf().getType()))
          packIntoOneBlockPerLeaf(gather, builder, /*isStatic=*/false);
      }
    });
  }
};
