//===- FuseLaunches.cpp - Fuse a launch into its consumer ---------------===//
//
// Implements `--cnm-fuse-launches`, described in docs/LaunchFusionDesign.md.
//
// When a distributed op's consumer is distributed the same way, the value
// travels leaf -> host -> leaf and comes back to the leaf it left. That shows
// up as a `cnm.gather` immediately followed by a `cnm.scatter` with the same
// map and the same buffer type: leaf `i` hands back exactly the elements leaf
// `i` is about to be handed again. Both transfers are then dead, and the two
// launches can run as one.
//
// Whether the schedules line up that way is the search's business (design
// §C3); this pass only recognises that they did. The test is syntactic, so
// nothing here has to reason about tiling factors, and a configuration whose
// schedules do not agree is simply left alone.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "cinm-mlir/Dialect/Cnm/Transforms/Passes.h"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMFUSELAUNCHESPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

} // namespace mlir::cnm

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Buffer effects
//===----------------------------------------------------------------------===//

/// Which of the workgroup's buffers `op` reads and which it writes.
///
/// Buffers are SSA values produced by `cnm.declare_buffer` and never aliased,
/// so identity comparison of the values is an exact dependence test -- there
/// is nothing here for an alias analysis to do.
struct BufferEffects {
  SmallVector<Value> reads, writes;

  explicit BufferEffects(Operation *op) {
    if (auto scatter = llvm::dyn_cast<cnm::ScatterOp>(op)) {
      writes.push_back(scatter.getBuffer());
      return;
    }
    if (auto gather = llvm::dyn_cast<cnm::GatherOp>(op)) {
      reads.push_back(gather.getBuffer());
      return;
    }
    if (auto launch = llvm::dyn_cast<cnm::LaunchOp>(op)) {
      for (Value v : launch.getInputs())
        if (llvm::isa<cnm::BufferType>(v.getType()))
          reads.push_back(v);
      // An `outs` buffer is read as well as written: a launch body may
      // accumulate into it.
      for (Value v : launch.getOutBuffers()) {
        reads.push_back(v);
        writes.push_back(v);
      }
      return;
    }
    // Anything else naming a buffer is not something this pass models, so
    // assume it both reads and writes every buffer it can see.
    for (Value v : op->getOperands())
      if (llvm::isa<cnm::BufferType>(v.getType())) {
        reads.push_back(v);
        writes.push_back(v);
      }
  }

  bool conflictsWith(const BufferEffects &other) const {
    auto intersects = [](ArrayRef<Value> a, ArrayRef<Value> b) {
      return llvm::any_of(a, [&](Value v) { return llvm::is_contained(b, v); });
    };
    return intersects(writes, other.reads) ||
           intersects(writes, other.writes) || intersects(reads, other.writes);
  }
};

/// Whether `a` certainly runs before `b`.
///
/// The two are routinely at different region depths here: once
/// `--cnm-hoist-workgroups` has run, a workgroup, its buffers and its
/// `cnm.free_workgroup` sit outside the compute block whose scatters, launches
/// and gathers use them. `DominanceInfo` answers the question only when `b`
/// lies inside `a`'s region -- it normalises `b` into it and gives up
/// otherwise -- so lift whichever of the two is deeper until they are
/// siblings, and ask about those.
///
/// Conservative when one op encloses the other, which is not a case this pass
/// has to decide.
bool happensBefore(DominanceInfo &dominance, Operation *a, Operation *b) {
  if (Operation *lifted = b->getParentRegion()->findAncestorOpInRegion(*a))
    return lifted != b && dominance.properlyDominates(lifted, b);
  if (Operation *lifted = a->getParentRegion()->findAncestorOpInRegion(*b))
    return lifted != a && dominance.properlyDominates(a, lifted);
  return false;
}

/// Whether `op` can be moved down to just before `before` -- both in the same
/// block, `op` first -- without changing what any of them observes.
bool canSinkTo(Operation *op, Operation *before) {
  assert(op->getBlock() == before->getBlock() && op->isBeforeInBlock(before));
  BufferEffects effects(op);
  for (Operation *it = op->getNextNode(); it != before;
       it = it->getNextNode()) {
    // An SSA use would be left dangling.
    if (llvm::any_of(it->getOperands(),
                     [&](Value v) { return v.getDefiningOp() == op; }))
      return false;
    if (effects.conflictsWith(BufferEffects(it)))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Pattern 1: cancel a gather/scatter round trip
//===----------------------------------------------------------------------===//

/// The `cnm.workgroup` a value comes from, or null.
cnm::WorkgroupOp workgroupDefOf(Value wg) {
  return wg.getDefiningOp<cnm::WorkgroupOp>();
}

/// Make `victim` the same workgroup as `keep`, so that the two launches around
/// a cancelled round trip run on the same leaves.
///
/// This is deliberately not a canonicalization on its own. Acquiring a
/// workgroup twice asks for two sets of leaves, and coalescing them is only
/// obviously the right thing when something has just established that the two
/// halves want to be co-located -- which is exactly what cancelling the round
/// trip establishes. Whether both halves' buffers still *fit* once they share
/// leaves is a capacity question, settled later by the occupancy check on the
/// lowered program.
///
/// Fails without touching anything if the two cannot be coalesced.
LogicalResult unifyWorkgroups(IRRewriter &rewriter, DominanceInfo &dominance,
                              cnm::WorkgroupOp keep, cnm::WorkgroupOp victim) {
  if (keep == victim)
    return success();
  if (keep.getType() != victim.getType())
    return failure();
  if (!happensBefore(dominance, keep.getOperation(), victim.getOperation()))
    return failure();

  // Both handles are released, and afterwards there is one. Keep whichever
  // release comes last, and only if it really is last: a release that some
  // surviving use outlives would be a use-after-free.
  SmallVector<cnm::FreeWorkgroupOp> frees;
  for (cnm::WorkgroupOp wg : {keep, victim})
    for (Operation *user : wg->getUsers())
      if (auto free = llvm::dyn_cast<cnm::FreeWorkgroupOp>(user))
        frees.push_back(free);
  cnm::FreeWorkgroupOp survivor;
  for (cnm::FreeWorkgroupOp free : frees)
    if (!survivor ||
        happensBefore(dominance, survivor.getOperation(), free.getOperation()))
      survivor = free;
  if (survivor)
    for (cnm::WorkgroupOp wg : {keep, victim})
      for (Operation *user : wg->getUsers())
        if (!llvm::isa<cnm::FreeWorkgroupOp>(user) &&
            !happensBefore(dominance, user, survivor.getOperation()))
          return failure();

  rewriter.replaceAllUsesWith(victim.getResult(), keep.getResult());
  for (cnm::FreeWorkgroupOp free : frees)
    if (free != survivor)
      rewriter.eraseOp(free);
  rewriter.eraseOp(victim);
  return success();
}

/// Whether `buffer` is written by anything other than `except`.
bool hasOtherWriter(Value buffer, Operation *except) {
  for (Operation *user : buffer.getUsers()) {
    if (user == except)
      continue;
    if (llvm::is_contained(BufferEffects(user).writes, buffer))
      return true;
  }
  return false;
}

/// `cnm.gather %src[m] ... ` followed by `cnm.scatter <that> into %dst[m]`:
/// every leaf writes out its own block and reads the same one back. Redirect
/// the consumer at `%src` and drop both transfers.
LogicalResult cancelRoundTrip(IRRewriter &rewriter, DominanceInfo &dominance,
                              cnm::ScatterOp scatter) {
  auto gather = scatter.getInput().getDefiningOp<cnm::GatherOp>();
  if (!gather || !gather.getOutput())
    return failure();
  if (gather->getBlock() != scatter->getBlock())
    return failure();

  // Same map into the same shape means leaf i's block of one is leaf i's block
  // of the other, element for element.
  if (gather.getGatherMap() != scatter.getScatterMap())
    return failure();
  Value src = gather.getBuffer(), dst = scatter.getBuffer();
  if (src.getType() != dst.getType())
    return failure();
  if (src == dst)
    return failure(); // nothing to redirect

  // The scatter must be the only thing that fills `dst`, or redirecting its
  // readers at `src` would lose whatever else was written there.
  if (hasOtherWriter(dst, scatter))
    return failure();
  // And every writer of `src` must already be done by the time the gather
  // runs, since the readers being redirected start reading `src` from there
  // on. The producing launch qualifies; anything after the gather does not.
  for (Operation *user : src.getUsers())
    if (user != gather.getOperation() &&
        llvm::is_contained(BufferEffects(user).writes, src) &&
        !happensBefore(dominance, user, gather.getOperation()))
      return failure();

  cnm::WorkgroupOp srcWg = workgroupDefOf(gather.getWg());
  cnm::WorkgroupOp dstWg = workgroupDefOf(scatter.getWg());
  if (!srcWg || !dstWg)
    return failure();
  if (failed(unifyWorkgroups(rewriter, dominance, srcWg, dstWg)))
    return failure();

  rewriter.replaceAllUsesWith(dst, src);
  rewriter.eraseOp(scatter);
  // The gather stays if anything else still wants the value on the host --
  // the block's result, typically. Only the transfer *into* the workgroup was
  // certainly redundant.
  if (gather.getOutput().use_empty())
    rewriter.eraseOp(gather);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern 2: merge two launches on one workgroup
//===----------------------------------------------------------------------===//

/// Merge `first` and `second`, which must run on the same workgroup, into one
/// launch at `second`'s position. `first` is required to be sinkable there.
LogicalResult mergeLaunches(IRRewriter &rewriter, cnm::LaunchOp first,
                            cnm::LaunchOp second) {
  // The memref each parameter is seen as inside a body. A buffer passed to
  // both launches must be seen the same way by both, which its `cnm.buffer`
  // type guarantees.
  llvm::MapVector<Value, Type> paramTypes;
  for (cnm::LaunchOp launch : {first, second}) {
    SmallVector<Value> params = launch.getParams();
    Block &body = launch.getBody().front();
    for (auto [param, arg] : llvm::zip_equal(params, body.getArguments())) {
      auto [it, inserted] = paramTypes.insert({param, arg.getType()});
      if (!inserted && it->second != arg.getType())
        return failure();
    }
  }

  // A buffer either launch writes is an output of the merged one, so that the
  // consumer reads what the producer left rather than a stale scatter.
  SmallVector<Value> outs, ins;
  for (cnm::LaunchOp launch : {first, second})
    for (Value v : launch.getOutBuffers())
      if (!llvm::is_contained(outs, v))
        outs.push_back(v);
  for (cnm::LaunchOp launch : {first, second})
    for (Value v : launch.getInputs())
      if (!llvm::is_contained(outs, v) && !llvm::is_contained(ins, v))
        ins.push_back(v);

  SmallVector<Value> params(ins);
  llvm::append_range(params, outs);

  rewriter.setInsertionPoint(second);
  auto merged = cnm::LaunchOp::create(rewriter, second.getLoc(), first.getWg(),
                                      ins, outs);
  Block &body = merged.getBody().emplaceBlock();
  for (Value param : params)
    body.addArgument(paramTypes.lookup(param), param.getLoc());

  rewriter.setInsertionPointToEnd(&body);
  for (cnm::LaunchOp launch : {first, second}) {
    IRMapping mapping;
    SmallVector<Value> launchParams = launch.getParams();
    Block &launchBody = launch.getBody().front();
    for (auto [param, arg] :
         llvm::zip_equal(launchParams, launchBody.getArguments())) {
      auto *it = llvm::find(params, param);
      assert(it != params.end() && "every parameter was collected above");
      mapping.map(arg, body.getArgument(std::distance(params.begin(), it)));
    }
    for (Operation &op : launchBody.without_terminator())
      rewriter.clone(op, mapping);
  }
  cnm::ReturnOp::create(rewriter, merged.getLoc());

  rewriter.eraseOp(second);
  rewriter.eraseOp(first);
  return success();
}

/// The launch that produced a buffer `launch` reads, if there is exactly one
/// such producer and the two can be brought together.
cnm::LaunchOp findMergeableProducer(cnm::LaunchOp launch) {
  cnm::LaunchOp producer;
  for (Value input : launch.getInputs()) {
    if (!llvm::isa<cnm::BufferType>(input.getType()))
      continue;
    for (Operation *user : input.getUsers()) {
      auto candidate = llvm::dyn_cast<cnm::LaunchOp>(user);
      if (!candidate || candidate == launch)
        continue;
      if (!llvm::is_contained(candidate.getOutBuffers(), input))
        continue;
      if (candidate.getWg() != launch.getWg())
        continue;
      if (candidate->getBlock() != launch->getBlock() ||
          !candidate->isBeforeInBlock(launch))
        continue;
      // Two producers of two different inputs would both have to be sunk, and
      // the second sink would have to step over the first. Not worth the case
      // analysis for something that does not occur today.
      if (producer && producer != candidate)
        return nullptr;
      producer = candidate;
    }
  }
  if (producer && !canSinkTo(producer, launch))
    return nullptr;
  return producer;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CnmFuseLaunchesPass
    : public cnm::impl::CnmFuseLaunchesPassBase<CnmFuseLaunchesPass> {
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    // A chain of three ops fuses in two rounds: cancelling one round trip is
    // what makes the next pair adjacent. Both steps only ever remove ops, so
    // the loop terminates.
    bool changed = true;
    while (changed) {
      changed = false;

      // Dominance rather than block order: the ops this pass compares
      // straddle region boundaries. Once --cnm-hoist-workgroups has run, a
      // workgroup and its buffers are declared outside the compute block whose
      // launches, scatters and gathers use them, so "does this happen first"
      // cannot be asked of a single block's order. Rebuilt per round because
      // merging launches introduces ops the previous round never saw.
      DominanceInfo dominance(getOperation());

      SmallVector<cnm::ScatterOp> scatters;
      getOperation()->walk([&](cnm::ScatterOp op) { scatters.push_back(op); });
      for (cnm::ScatterOp scatter : scatters)
        changed |= succeeded(cancelRoundTrip(rewriter, dominance, scatter));

      if (mergeLaunchBodies) {
        SmallVector<cnm::LaunchOp> launches;
        getOperation()->walk([&](cnm::LaunchOp op) { launches.push_back(op); });
        for (cnm::LaunchOp launch : launches) {
          if (!launch->getBlock())
            continue; // already merged away this round
          if (cnm::LaunchOp producer = findMergeableProducer(launch))
            changed |= succeeded(mergeLaunches(rewriter, producer, launch));
        }
      }
    }
  }
};

} // namespace
