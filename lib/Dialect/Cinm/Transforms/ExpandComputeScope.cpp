//===- ExpandComputeScope.cpp - Grow cinm.compute_block regions -----------===//
//
// Widens cinm.compute_block regions by pulling in the ops that produce their
// operands (tensor.extract_slice, tensor.splat, linalg.fill, ...) and the
// tensor.insert_slice ops that consume their results, so that the shaping of
// the offloaded data happens inside the region.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMEXPANDCOMPUTESCOPEPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// How deep a chain of producers we are willing to pull into a block.
constexpr unsigned kMaxRematDepth = 8;

/// Whether `def` is cheap enough to be executed inside the compute block
/// instead of being computed outside and passed in.
bool isRematerializable(Operation *def) {
  if (!isMemoryEffectFree(def))
    return false;
  // Constants are kept inside blocks by --cinm-isolate-compute-blocks too.
  if (def->hasTrait<OpTrait::ConstantLike>() ||
      isa<tensor::SplatOp, tensor::ExtractSliceOp, linalg::FillOp>(def))
    return true;
  if (auto empty = dyn_cast_or_null<tensor::EmptyOp>(def)) {
    return empty.getType().getNumElements() == 1;
  }

  if (auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(def)) {
    return linalg::isaFillOpInterface(generic).has_value() ||
           linalg::isaBroadcastOpInterface(generic).has_value() ||
           linalg::isaTransposeOpInterface(generic);
  }
  return false;
}

/// The values `def` takes from the outside: its operands, plus the values its
/// regions capture. A `linalg.generic` in fill form for instance holds the fill
/// value in its body rather than in an operand, and a compute block is isolated
/// from above, so those have to cross the boundary as well.
void getExternalValues(Operation *def, SmallVectorImpl<Value> &out) {
  llvm::append_range(out, def->getOperands());
  if (def->getNumRegions() == 0)
    return;
  SetVector<Value> captured;
  getUsedValuesDefinedAbove(def->getRegions(), captured);
  llvm::append_range(out, captured);
}

/// Whether `v` can be used by an operation placed right before `target`.
/// `v` is assumed to be visible from some point later in `target`'s block.
bool isAvailableBefore(Value v, Operation *target) {
  Operation *def = v.getDefiningOp();
  // A block argument either belongs to `target`'s block, in which case it
  // dominates all of its operations, or to an enclosing region.
  if (!def)
    return true;
  // A null ancestor means `def` lives in an enclosing region.
  Operation *ancestor = target->getBlock()->findAncestorOpInBlock(*def);
  return !ancestor || ancestor->isBeforeInBlock(target);
}

/// Whether the op defining `v` can be cloned into a block placed at `target`,
/// which requires everything it takes from the outside to be available there,
/// recursively.
bool canRematerializeBefore(Value v, Operation *target,
                            unsigned depth = kMaxRematDepth) {
  Operation *def = v.getDefiningOp();
  if (depth == 0 || !def || !isRematerializable(def))
    return false;
  SmallVector<Value> externals;
  getExternalValues(def, externals);
  return llvm::all_of(externals, [&](Value external) {
    return isAvailableBefore(external, target) ||
           canRematerializeBefore(external, target, depth - 1);
  });
}

/// Collects the values a rewritten compute block needs from the outside and
/// hands out an id for each of them. Values are deduplicated, so that the same
/// value is never captured twice. A value whose producer is rematerializable is
/// not captured at all: the producer is cloned into the block instead, which is
/// how this pass widens the scope of a block.
class BlockInputs {
public:
  /// `target` is the op the rewritten block takes the place of. The values
  /// already captured by the block are passed as `initial`; they keep their
  /// position in the operand list, so that id `i` is block argument `i`.
  explicit BlockInputs(Operation *target, ValueRange initial = {})
      : target(target) {
    for (Value value : initial)
      capture(value);
  }

  /// Returns the id under which `v` is available inside the block, or failure
  /// if `v` cannot be made available there.
  FailureOr<unsigned> add(Value v) {
    auto it = ids.find(v);
    if (it != ids.end())
      return it->second;
    if (canRematerializeBefore(v, target))
      return remat(v);
    if (isAvailableBefore(v, target))
      return capture(v);
    // Defined after the block, like the fresh destination of a
    // tensor.insert_slice usually is. A pure producer can be moved before the
    // block instead -- moving an op earlier only widens what it dominates --
    // and captured like any other operand; moving is a plan here and IR only
    // once the candidate is accepted (applyHoists), so a rejected candidate
    // leaves everything in place.
    if (canHoistBefore(v))
      return hoist(v);
    return failure();
  }

  /// Moves the producers `add` planned to hoist before the block, dependency
  /// order preserved. Call once the rewrite is decided, before the block is
  /// rebuilt.
  void applyHoists(RewriterBase &rewriter) {
    for (Operation *def : hoists)
      rewriter.moveOpBefore(def, target);
  }

  /// Whether the value behind `id` is produced inside the block.
  bool isRemat(unsigned id) const { return entries[id].def; }

  /// The operands of the rewritten block.
  SmallVector<Value> getOperands() const {
    SmallVector<Value> operands;
    for (const Entry &entry : entries)
      if (!entry.def)
        operands.push_back(entry.value);
    return operands;
  }

  /// Makes every input available inside `body`, which must have one argument
  /// per captured value. Has to be called before `getValue`.
  void materialize(OpBuilder &builder, Block *body) {
    values.reserve(entries.size());
    unsigned argIdx = 0;
    for (const Entry &entry : entries) {
      if (!entry.def) {
        values.push_back(body->getArgument(argIdx++));
        continue;
      }
      IRMapping mapping;
      for (auto [external, id] : entry.externals)
        mapping.map(external, values[id]);
      values.push_back(
          builder.clone(*entry.def, mapping)->getResult(entry.resultIdx));
    }
  }

  Value getValue(unsigned id) const { return values[id]; }

private:
  struct Entry {
    Value value;
    Operation *def = nullptr; ///< producer to clone, null for a plain operand
    unsigned resultIdx = 0;
    /// What the producer takes from the outside, and the id it is available
    /// under inside the block.
    SmallVector<std::pair<Value, unsigned>> externals;
  };

  unsigned capture(Value v) {
    unsigned id = entries.size();
    entries.push_back({v});
    ids.try_emplace(v, id);
    return id;
  }

  /// Whether `v`'s producer can be moved before the block: pure, in the
  /// block's own block, and everything it takes from the outside either
  /// already available there or hoistable itself.
  bool canHoistBefore(Value v, unsigned depth = kMaxRematDepth) const {
    Operation *def = v.getDefiningOp();
    if (depth == 0 || !def || def->getBlock() != target->getBlock() ||
        !isMemoryEffectFree(def))
      return false;
    SmallVector<Value> externals;
    getExternalValues(def, externals);
    return llvm::all_of(externals, [&](Value external) {
      return isAvailableBefore(external, target) ||
             canHoistBefore(external, depth - 1);
    });
  }

  /// Plans `v`'s producer to be moved before the block and captures `v`. The
  /// producer's own late operands are planned first, so the recorded order is
  /// a valid program order; they stay outside the block with it, so only `v`
  /// itself becomes an input.
  unsigned hoist(Value v) {
    planHoist(v.getDefiningOp());
    return capture(v);
  }

  void planHoist(Operation *def) {
    if (!plannedHoists.insert(def).second)
      return;
    SmallVector<Value> externals;
    getExternalValues(def, externals);
    for (Value external : externals)
      if (!isAvailableBefore(external, target))
        planHoist(external.getDefiningOp());
    hoists.push_back(def);
  }

  /// Registers `v` as produced inside the block. What its producer takes from
  /// the outside is added first, so that entries are always in materialization
  /// order.
  unsigned remat(Value v) {
    auto result = cast<OpResult>(v);
    Entry entry{v, result.getOwner(), result.getResultNumber(), {}};
    SmallVector<Value> externals;
    getExternalValues(entry.def, externals);
    for (Value external : externals) {
      FailureOr<unsigned> id = add(external);
      assert(succeeded(id) && "checked by canRematerializeBefore");
      entry.externals.emplace_back(external, *id);
    }
    unsigned id = entries.size();
    entries.push_back(std::move(entry));
    ids.try_emplace(v, id);
    return id;
  }

  Operation *target;
  SmallVector<Entry> entries;
  SmallVector<Value> values;
  DenseMap<Value, unsigned> ids;
  /// Producers to move before the block, in a valid program order.
  SmallVector<Operation *> hoists;
  SmallPtrSet<Operation *, 4> plannedHoists;
};

/// Creates a copy of `op` with the given operands and result types, moving the
/// body of `op` into it. `prologue` is called on the empty body of the new op
/// and must fill `argRepl` with one value per block argument of the old body;
/// those values replace the old block arguments. `op` is left in place, without
/// a body, for the caller to replace and erase.
cinm::ComputeBlockOp rebuildComputeBlock(
    RewriterBase &rewriter, cinm::ComputeBlockOp op, ValueRange newOperands,
    TypeRange newResultTypes,
    llvm::function_ref<void(RewriterBase &, Block *, SmallVectorImpl<Value> &)>
        prologue) {
  OpBuilder::InsertionGuard guard(rewriter);

  rewriter.setInsertionPoint(op);
  auto newOp = cinm::ComputeBlockOp::create(rewriter, op.getLoc(), newOperands,
                                            newResultTypes);
  newOp->setAttrs(op->getAttrs());

  Block *newBody = &newOp.getBody().front();
  rewriter.setInsertionPointToStart(newBody);
  SmallVector<Value> argRepl;
  prologue(rewriter, newBody, argRepl);

  Block *oldBody = &op.getBody().front();
  assert(argRepl.size() == oldBody->getNumArguments() &&
         "prologue must map every block argument");
  rewriter.inlineBlockBefore(oldBody, newBody, newBody->end(), argRepl);
  return newOp;
}

/// Whether `user`, which takes a block result as its first operand, can be
/// moved into that block. Such an op must be pure and have a single result,
/// which the block then yields in place of the one it consumes.
bool isAbsorbableConsumer(Operation *user) {
  if (isa<tensor::InsertSliceOp>(user))
    return true;
  // Extracting the only element of a result turns it into a scalar, which
  // spares materializing a one-element buffer outside the block.
  if (auto extract = llvm::dyn_cast<tensor::ExtractOp>(user)) {
    RankedTensorType tensorTy = extract.getTensor().getType();
    return tensorTy.hasStaticShape() && tensorTy.getNumElements() == 1;
  }
  return false;
}

/// Moves the op consuming a result of the block into the block.
///
///   %r = cinm.compute_block (...) -> tensor<8xf32> { ... cinm.yield %v }
///   %s = tensor.insert_slice %r into %d[%i] [8] [1]
/// becomes
///   %s = cinm.compute_block (..., %bd = %d, %bi = %i) -> tensor<64xf32> {
///          ...
///          %w = tensor.insert_slice %v into %bd[%bi] [8] [1]
///          cinm.yield %w
///        }
struct AbsorbResultConsumer : OpRewritePattern<cinm::ComputeBlockOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cinm::ComputeBlockOp op,
                                PatternRewriter &rewriter) const override {
    // Inputs of the rewritten block: everything the block already takes, plus
    // what the consumer needs.
    std::optional<BlockInputs> inputs;
    Operation *consumer = nullptr;
    unsigned resultIdx = 0;
    SmallVector<unsigned> consumerIds;

    for (OpResult result : op->getResults()) {
      if (!result.hasOneUse())
        continue;
      OpOperand &use = *result.getUses().begin();
      Operation *candidate = use.getOwner();
      // The result must be the data the consumer works on, e.g. the inserted
      // slice rather than the destination of a tensor.insert_slice.
      if (use.getOperandNumber() != 0 || !isAbsorbableConsumer(candidate))
        continue;
      // Being in the same block guarantees the consumer is executed exactly
      // once, right after the compute block.
      if (candidate->getBlock() != op->getBlock())
        continue;
      // Whatever else the consumer takes, e.g. a destination or an index, has
      // to be usable from the position of the compute block. Start from a fresh
      // input set, so that a rejected candidate leaves nothing behind.
      BlockInputs candidateInputs(op, op.getOperands());
      SmallVector<unsigned> ids;
      if (failed(addAll(candidateInputs, candidate->getOperands().drop_front(),
                        ids)))
        continue;
      consumer = candidate;
      resultIdx = result.getResultNumber();
      consumerIds = std::move(ids);
      inputs.emplace(std::move(candidateInputs));
      break;
    }
    if (!consumer)
      return failure();

    SmallVector<Type> resultTypes(op->getResultTypes());
    resultTypes[resultIdx] = consumer->getResult(0).getType();

    inputs->applyHoists(rewriter);
    unsigned numOldArgs = op.getBody().front().getNumArguments();
    auto newOp = rebuildComputeBlock(
        rewriter, op, inputs->getOperands(), resultTypes,
        [&](RewriterBase &builder, Block *body,
            SmallVectorImpl<Value> &argRepl) {
          inputs->materialize(builder, body);
          llvm::append_range(argRepl,
                             body->getArguments().take_front(numOldArgs));
        });

    // Redo the consumer inside the block, on the yielded value.
    Operation *yield = newOp.getBody().front().getTerminator();
    rewriter.setInsertionPoint(yield);
    IRMapping mapping;
    mapping.map(consumer->getOperand(0), yield->getOperand(resultIdx));
    for (auto [operand, id] :
         llvm::zip(consumer->getOperands().drop_front(), consumerIds))
      mapping.map(operand, inputs->getValue(id));
    Operation *absorbed = rewriter.clone(*consumer, mapping);
    rewriter.modifyOpInPlace(
        yield, [&]() { yield->setOperand(resultIdx, absorbed->getResult(0)); });

    for (auto [idx, result] : llvm::enumerate(op->getResults())) {
      if (idx == resultIdx)
        rewriter.replaceAllUsesWith(consumer->getResult(0),
                                    newOp->getResult(idx));
      else
        rewriter.replaceAllUsesWith(result, newOp->getResult(idx));
    }
    rewriter.eraseOp(consumer);
    rewriter.eraseOp(op);
    return success();
  }

private:
  static LogicalResult addAll(BlockInputs &inputs, OperandRange values,
                              SmallVectorImpl<unsigned> &ids) {
    for (Value value : values) {
      FailureOr<unsigned> id = inputs.add(value);
      if (failed(id))
        return failure();
      ids.push_back(*id);
    }
    return success();
  }
};

/// Moves the ops producing the operands of the block into the block.
///
///   %e = tensor.extract_slice %s[%i] [8] [1]
///   %r = cinm.compute_block (%a = %e : tensor<8xf32>) -> ... { ... }
/// becomes
///   %r = cinm.compute_block (%bs = %s : tensor<64xf32>, %bi = %i : index) ->
///   ... {
///          %a = tensor.extract_slice %bs[%bi] [8] [1]
///          ...
///        }
struct AbsorbOperandProducers : OpRewritePattern<cinm::ComputeBlockOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cinm::ComputeBlockOp op,
                                PatternRewriter &rewriter) const override {
    // The operands of the block all dominate it, so they can only fail to be
    // added if they are not worth rematerializing.
    BlockInputs inputs(op);
    SmallVector<unsigned> argIds;
    bool absorbedAny = false;
    for (Value operand : op.getOperands()) {
      FailureOr<unsigned> id = inputs.add(operand);
      assert(succeeded(id) && "operands of the block dominate it");
      absorbedAny |= inputs.isRemat(*id);
      argIds.push_back(*id);
    }
    if (!absorbedAny)
      return failure();

    auto newOp = rebuildComputeBlock(rewriter, op, inputs.getOperands(),
                                     op->getResultTypes(),
                                     [&](RewriterBase &builder, Block *body,
                                         SmallVectorImpl<Value> &argRepl) {
                                       inputs.materialize(builder, body);
                                       for (unsigned id : argIds)
                                         argRepl.push_back(inputs.getValue(id));
                                     });

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ExpandComputeScopePass
    : public impl::CinmExpandComputeScopePassBase<ExpandComputeScopePass> {
  using Base::Base;

  void runOnOperation() override {
    if (!absorbOperands && !absorbResults)
      return;

    RewritePatternSet patterns(&getContext());
    if (absorbOperands)
      patterns.add<AbsorbOperandProducers>(&getContext());
    if (absorbResults)
      patterns.add<AbsorbResultConsumer>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cinm
