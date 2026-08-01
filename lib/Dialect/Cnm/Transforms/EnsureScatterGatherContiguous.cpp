//===- EnsureScatterGatherContiguous.cpp - Pack non-contiguous transfers -===//
//
// Lowering cnm.scatter/cnm.gather to upmem.scatter/upmem.gather performs a
// single flat memcpy per DPU. This is only correct if the transferred
// elements are actually contiguous in the host memref (see
// upmem::ScatterOp::verify / upmem::GatherOp::verify and the runtime's
// do_dpu_transfer). This pass detects host memrefs that aren't contiguous
// (e.g. subviews of a larger tensor) and inserts an intermediate contiguous
// buffer, similar to what packATile does for the tiled GEMV/reduction
// templates, but generically for any cnm.scatter/cnm.gather.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmScatterMap.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <cinm-mlir/Utils/CinmUtils.h>

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
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

void ensureScatterContiguous(cnm::ScatterOp op, OpBuilder &b) {
  auto input = op.getInput();
  if (!isa<MemRefType>(input.getType()) || isFullyContiguous(input))
    return;

  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  Value packed = allocateContiguousLike(b, loc, input);
  memref::CopyOp::create(b, loc, input, packed);
  op.getInputMutable().assign(packed);

  // b.setInsertionPointAfter(op);
  // memref::DeallocOp::create(b, loc, packed);
}

void ensureGatherContiguous(cnm::GatherOp op, OpBuilder &b) {
  auto outputBuf = op.getOutputBuf();
  if (!isa<MemRefType>(outputBuf.getType()) || isFullyContiguous(outputBuf))
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
                                             int64_t factor,
                                             unsigned numDims) {
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

template <class Op> void deriveBlockForm(Op op, OpBuilder &b) {
  AffineMap map = op.getScatterMap();
  auto bufferTy = op.getBuffer().getType();
  Value host = op.getHostValue();
  int64_t before = cnm::getNumImplicitHostDims(map, bufferTy);
  BlockForm form = computeBlockForm(map, bufferTy, op.getHostType());
  if (form.implicit == before)
    return;
  // Splitting rewrites the value, which only works once it is a memref.
  if (!form.splits.empty() && !isa<MemRefType>(host.getType()))
    return;

  if (!form.splits.empty()) {
    b.setInsertionPoint(op);
    Value expanded = expandHost(b, op.getLoc(), host, form.splits);
    op.getHostValueMutable().assign(expanded);
  }
  op.setScatterMap(AffineMap::get(map.getNumDims() - form.implicit,
                                  map.getNumSymbols(), form.results,
                                  map.getContext()));
}

} // namespace

struct CnmEnsureScatterGatherContiguousPass
    : public cnm::impl::CnmEnsureScatterGatherContiguousPassBase<
          CnmEnsureScatterGatherContiguousPass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation()->walk([&](Operation *op) {
      if (auto scatter = dyn_cast<cnm::ScatterOp>(op)) {
        ensureScatterContiguous(scatter, builder);
        deriveBlockForm(scatter, builder);
      } else if (auto gather = dyn_cast<cnm::GatherOp>(op)) {
        ensureGatherContiguous(gather, builder);
        deriveBlockForm(gather, builder);
      }
    });
  }
};
