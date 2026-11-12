//===- AnnotateOperandReuse.cpp - Reuse distance of compute block operands ===//
//
// Walks each function's compute blocks in program order, keeping a running
// count of the bytes the blocks touch (each reads its operands and writes its
// results, whole), and the count at which each piece of data was last
// touched. A block's operand is annotated with the difference: the bytes
// touched between its data's last touch and this read. See the pass
// description in Passes.td.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/CinmTransforms.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>

#include <map>
#include <vector>

namespace mlir::cinm {

#define GEN_PASS_DEF_CINMANNOTATEOPERANDREUSEPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

namespace {

/// The bytes a value of `type` occupies, 0 for a scalar or a dynamic shape.
int64_t bytesOf(Type type) {
  auto shaped = llvm::dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape() ||
      !shaped.getElementType().isIntOrFloat())
    return 0;
  return shaped.getNumElements() *
         llvm::divideCeil(shaped.getElementTypeBitWidth(), 8);
}

class ReuseWalk {
public:
  explicit ReuseWalk(func::FuncOp func) : func(func) {}

  void walk(Block &block) {
    for (Operation &op : block) {
      if (auto compute = llvm::dyn_cast<ComputeBlockOp>(op))
        annotate(compute);
      else if (auto loop = llvm::dyn_cast<scf::ForOp>(op))
        walkLoop(loop);
      else
        // Any other op with blocks inside (none, after
        // --cinm-complete-compute-graph, but for a loop it leaves around
        // compute blocks): its regions once, in order.
        for (Region &region : op.getRegions())
          for (Block &nested : region)
            walk(nested);
    }
  }

private:
  func::FuncOp func;
  /// Bytes touched so far.
  int64_t clock = 0;
  /// Data id -> `clock` right after the block that last touched it.
  DenseMap<unsigned, int64_t> lastTouch;
  /// Interned identities: a value (with the iterations of the loops being
  /// walked that it is defined in), or a slice of some data.
  std::map<std::vector<int64_t>, unsigned> ids;
  /// Loop iteration arguments and results: the data they stand for.
  DenseMap<Value, unsigned> bound;
  /// The iteration of each loop body being walked.
  DenseMap<Block *, int64_t> iteration;

  unsigned intern(std::vector<int64_t> key) {
    return ids.try_emplace(std::move(key), ids.size()).first->second;
  }

  /// A value as an SSA identity: the same value in another iteration of a
  /// loop it is defined in is another value.
  unsigned valueId(Value value) {
    std::vector<int64_t> key{
        0, reinterpret_cast<intptr_t>(value.getAsOpaquePointer())};
    for (Block *block = value.getParentBlock(); block;) {
      auto it = iteration.find(block);
      if (it != iteration.end())
        key.push_back(it->second);
      Operation *parent = block->getParentOp();
      if (!parent || parent == func)
        break;
      block = parent->getBlock();
    }
    return intern(std::move(key));
  }

  /// An offset, size or stride of a slice: a constant, or a value.
  void appendIndex(OpFoldResult index, std::vector<int64_t> &key) {
    if (std::optional<int64_t> constant = getConstantIntValue(index)) {
      key.push_back(0);
      key.push_back(*constant);
      return;
    }
    key.push_back(1);
    key.push_back(valueId(llvm::cast<Value>(index)));
  }

  /// The data `value` holds.
  unsigned dataOf(Value value) {
    auto it = bound.find(value);
    if (it != bound.end())
      return it->second;
    Operation *def = value.getDefiningOp();
    if (llvm::isa_and_nonnull<tensor::CastOp, tensor::CollapseShapeOp,
                              tensor::ExpandShapeOp, tensor::ReshapeOp>(def))
      return dataOf(def->getOperand(0));
    if (auto slice = llvm::dyn_cast_or_null<tensor::ExtractSliceOp>(def)) {
      std::vector<int64_t> key{1, dataOf(slice.getSource())};
      for (OpFoldResult index : slice.getMixedOffsets())
        appendIndex(index, key);
      for (OpFoldResult index : slice.getMixedSizes())
        appendIndex(index, key);
      for (OpFoldResult index : slice.getMixedStrides())
        appendIndex(index, key);
      return intern(std::move(key));
    }
    return valueId(value);
  }

  void annotate(ComputeBlockOp block) {
    SmallVector<int64_t> entries;
    SmallVector<unsigned> read;
    for (Value operand : block->getOperands()) {
      unsigned data = dataOf(operand);
      auto it = lastTouch.find(data);
      entries.push_back(it == lastTouch.end() ? -1 : clock - it->second);
      read.push_back(data);
    }
    block->setAttr(CinmDialect::OPERAND_REUSE_NAME,
                   DenseI64ArrayAttr::get(block->getContext(), entries));

    // The block touches each piece of data it reads once, and writes its
    // results.
    llvm::SmallDenseSet<unsigned> counted;
    for (auto [operand, data] : llvm::zip(block->getOperands(), read))
      if (counted.insert(data).second)
        clock += bytesOf(operand.getType());
    for (Value result : block->getResults())
      clock += bytesOf(result.getType());
    for (unsigned data : read)
      lastTouch[data] = clock;
    for (Value result : block->getResults())
      lastTouch[dataOf(result)] = clock;
  }

  /// Two iterations of the body: the first binds the iteration arguments to
  /// the loop's inits, the second to the first's yields and leaves the
  /// annotations. The loop's results stand for the second's yields.
  void walkLoop(scf::ForOp loop) {
    Block &body = *loop.getBody();
    SmallVector<unsigned> carried = llvm::map_to_vector(
        loop.getInitArgs(), [&](Value init) { return dataOf(init); });
    for (int64_t it = 1; it <= 2; ++it) {
      iteration[&body] = it;
      for (auto [arg, data] : llvm::zip(loop.getRegionIterArgs(), carried))
        bound[arg] = data;
      walk(body);
      carried =
          llvm::map_to_vector(body.getTerminator()->getOperands(),
                              [&](Value yielded) { return dataOf(yielded); });
    }
    iteration.erase(&body);
    for (auto [result, data] : llvm::zip(loop.getResults(), carried))
      bound[result] = data;
  }
};

struct AnnotateOperandReusePass
    : public impl::CinmAnnotateOperandReusePassBase<AnnotateOperandReusePass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(requireIsolatedComputeOps(getOperation(), getArgument())))
      return signalPassFailure();
    getOperation()->walk([](func::FuncOp func) {
      if (!func.isExternal())
        ReuseWalk(func).walk(func.getBody().front());
    });
  }
};

} // namespace
} // namespace mlir::cinm
