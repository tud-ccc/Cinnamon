/// Issue DPU launches and transfers asynchronously, and wait for each set
/// only where the host needs what it did.
///
/// @file

#include <cinm-mlir/Dialect/Cinm/IR/CinmDialect.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "upmem-async-launches"

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMASYNCLAUNCHESPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

} // namespace mlir::upmem

using namespace mlir;

namespace {

/// What an op, its nested ops included, does to host memory and to DPU sets.
struct Footprint {
  SmallVector<Value> reads, writes; // memrefs
  SmallVector<Value> sets;          // DPU sets it operates on
  /// It has effects nothing here can describe (a call, say): it is ordered
  /// against everything.
  bool unknown = false;

  void append(const Footprint &other) {
    llvm::append_range(reads, other.reads);
    llvm::append_range(writes, other.writes);
    llvm::append_range(sets, other.sets);
    unknown |= other.unknown;
  }
  bool empty() const {
    return reads.empty() && writes.empty() && sets.empty() && !unknown;
  }
};

bool isDpuSet(Value v) { return isa<upmem::DeviceHierarchyType>(v.getType()); }

/// The asynchronous ops: issued on their set's queue, done at its next sync.
bool isAsync(Operation *op) {
  return op->hasAttr(upmem::UPMEMDialect::ASYNC_NAME);
}

/// An outlined host block's call touches host memory through its memref
/// operands alone, each the way its prototype's argument attributes say
/// (cinm.reads, cinm.writes: what the outlining found in the body).
bool outlinedCallFootprint(Operation *op, Footprint &fp) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call)
    return false;
  auto callee = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(op, call.getCalleeAttr()));
  if (!callee || !callee->hasAttr(cinm::CinmDialect::OUTLINED_NAME))
    return false;
  for (auto [i, v] : llvm::enumerate(call.getOperands())) {
    if (!isa<BaseMemRefType>(v.getType()))
      continue;
    if (callee.getArgAttr(i, cinm::CinmDialect::READS_NAME))
      fp.reads.push_back(v);
    if (callee.getArgAttr(i, cinm::CinmDialect::WRITES_NAME))
      fp.writes.push_back(v);
  }
  return true;
}

Footprint footprintOf(Operation *root) {
  Footprint fp;
  root->walk([&](Operation *op) {
    for (Value v : op->getOperands())
      if (isDpuSet(v))
        fp.sets.push_back(v);
    if (outlinedCallFootprint(op, fp))
      return;
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        Value v = effect.getValue();
        if (!v) {
          // Effects on a resource rather than a value: the DPU sets are
          // tracked through the operands above; anything else is a side
          // effect this pass cannot order.
          if (!isa<upmem::DpuSetResource>(effect.getResource()))
            fp.unknown = true;
          continue;
        }
        if (!isa<BaseMemRefType>(v.getType())) {
          // Memory reached some other way (a raw pointer): anything at all.
          fp.unknown = true;
          continue;
        }
        if (isa<MemoryEffects::Read>(effect.getEffect()))
          fp.reads.push_back(v);
        else if (isa<MemoryEffects::Write, MemoryEffects::Free>(
                     effect.getEffect()))
          fp.writes.push_back(v);
      }
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      fp.unknown = true;
    }
  });
  return fp;
}

/// How many ops past a sync are considered for moving before it.
constexpr size_t kMaxLookahead = 2000;

class Scheduler {
public:
  Scheduler(Operation *root, bool noaliasArguments)
      : aliases(root), noaliasArguments(noaliasArguments) {}

  void run(Operation *root) {
    // Host code only: device programs issue nothing.
    SmallVector<Block *> blocks;
    root->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<upmem::DpuProgramOp>(op))
        return WalkResult::skip();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
      return WalkResult::advance();
    });
    root->walk([&](Operation *op) {
      if (isa<upmem::WaitForOp, upmem::ScatterOnArrayOp,
              upmem::GatherFromArrayOp, upmem::BroadcastOp>(op))
        op->setAttr(upmem::UPMEMDialect::ASYNC_NAME,
                    UnitAttr::get(op->getContext()));
    });
    for (Block *block : blocks)
      placeSyncs(*block);
    for (Block *block : blocks)
      sinkSyncs(*block);
  }

private:
  AliasAnalysis aliases;
  bool noaliasArguments;
  /// Per inserted sync, the host buffers of the transfers it completes:
  /// what they read, what they write.
  llvm::DenseMap<Operation *, Footprint> completes;

  bool mayAlias(Value a, Value b) {
    if (aliases.alias(a, b).isNo())
      return false;
    // What local alias analysis does not see: an allocation and two globals
    // are distinct storage, and the storage of a module-private global never
    // reaches the module's functions through their arguments, since no
    // caller outside the module can name it.
    Value ra = rootOf(a), rb = rootOf(b);
    // Storage of its own: an allocation aliases nothing but its views.
    if (ra != rb && (isAllocation(ra) || isAllocation(rb)))
      return false;
    auto ga = ra.getDefiningOp<memref::GetGlobalOp>();
    auto gb = rb.getDefiningOp<memref::GetGlobalOp>();
    if (ga && gb && ga.getNameAttr() != gb.getNameAttr())
      return false;
    if (noaliasArguments && ra != rb && isFunctionArgument(ra) &&
        isFunctionArgument(rb))
      return false;
    return !((isPrivateGlobal(ra) && isFunctionArgument(rb)) ||
             (isPrivateGlobal(rb) && isFunctionArgument(ra)));
  }

  /// The memref `v` is a view of, through view-like ops.
  static Value rootOf(Value v) {
    while (auto view = v.getDefiningOp<ViewLikeOpInterface>())
      v = view.getViewSource();
    return v;
  }

  static bool isAllocation(Value v) {
    Operation *def = v.getDefiningOp();
    return def && hasEffect<MemoryEffects::Allocate>(def, v);
  }

  static bool isFunctionArgument(Value v) {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner()->isEntryBlock() &&
           isa<FunctionOpInterface>(arg.getOwner()->getParentOp());
  }

  static bool isPrivateGlobal(Value v) {
    auto get = v.getDefiningOp<memref::GetGlobalOp>();
    if (!get)
      return false;
    Operation *global =
        SymbolTable::lookupNearestSymbolFrom(get, get.getNameAttr());
    auto symbol = dyn_cast_or_null<SymbolOpInterface>(global);
    return symbol && symbol.isPrivate();
  }

  bool anyAlias(ArrayRef<Value> as, ArrayRef<Value> bs) {
    for (Value a : as)
      for (Value b : bs)
        if (mayAlias(a, b))
          return true;
    return false;
  }

  bool conflict(const Footprint &a, const Footprint &b) {
    if ((a.unknown && !b.empty()) || (b.unknown && !a.empty()))
      return true;
    for (Value s : a.sets)
      if (llvm::is_contained(b.sets, s))
        return true;
    return anyAlias(a.writes, b.reads) || anyAlias(a.writes, b.writes) ||
           anyAlias(a.reads, b.writes);
  }

  /// A sync's footprint: its set, and the buffers of the transfers it
  /// completes -- a later write to what a scatter reads, or any access to
  /// what a gather writes, must stay after it.
  Footprint syncFootprint(upmem::SyncOp sync) {
    Footprint fp = completes.lookup(sync);
    fp.sets = {sync.getDpuSet()};
    return fp;
  }

  /// Walk `block` keeping, per set, what its asynchronous transfers not yet
  /// synced read and write on the host; sync a set right before the first op
  /// that touches those buffers, and every set still in flight before the
  /// terminator. Ops issued on the same set need no sync between them: the
  /// set runs them in order.
  void placeSyncs(Block &block) {
    llvm::MapVector<Value, Footprint> inFlight;
    auto syncBefore = [&](Operation *where, Value set) {
      LLVM_DEBUG({
        const Footprint &pending = inFlight.lookup(set);
        Footprint fp = footprintOf(where);
        llvm::dbgs() << "[async] sync before " << where->getName()
                     << " (unknown " << fp.unknown << ", reads "
                     << fp.reads.size() << ", writes " << fp.writes.size()
                     << ", uses set " << llvm::is_contained(fp.sets, set)
                     << "; in flight: unknown " << pending.unknown << ", reads "
                     << pending.reads.size() << ", writes "
                     << pending.writes.size() << ")\n";
      });
      OpBuilder builder(where);
      auto sync = upmem::SyncOp::create(builder, where->getLoc(), set);
      completes[sync] = inFlight.lookup(set);
      inFlight.erase(set);
    };
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        for (Value set : llvm::to_vector(llvm::make_first_range(inFlight)))
          syncBefore(&op, set);
        break;
      }
      if (isAsync(&op)) {
        // An issued op waits in its set's queue behind the others; what it
        // does on the host joins what the set has in flight.
        Value set = *llvm::find_if(op.getOperands(), isDpuSet);
        Footprint fp = footprintOf(&op);
        fp.sets.clear();
        // Its host buffers may be what another set has in flight.
        for (Value other : llvm::to_vector(llvm::make_first_range(inFlight)))
          if (other != set && conflict(fp, inFlight.lookup(other)))
            syncBefore(&op, other);
        inFlight[set].append(fp);
        continue;
      }
      Footprint fp = footprintOf(&op);
      for (Value set : llvm::to_vector(llvm::make_first_range(inFlight))) {
        Footprint pending = inFlight.lookup(set);
        pending.sets = {set};
        if (conflict(fp, pending))
          syncBefore(&op, set);
      }
    }
  }

  /// Move each sync as late as the ops after it allow, together with what
  /// depends on it: an op that does not depend on the sync, nor on anything
  /// carried along with it, goes before it -- another set's launch, say, or
  /// host work that does not read the results -- so that it overlaps with
  /// what the synced set is still doing.
  void sinkSyncs(Block &block) {
    SmallVector<upmem::SyncOp> syncs(block.getOps<upmem::SyncOp>());
    for (upmem::SyncOp sync : syncs) {
      Footprint carried = syncFootprint(sync);
      llvm::SmallPtrSet<Operation *, 16> carriedOps{sync};
      // Bounded, so that compile time stays linear in the block: what lies
      // further than this stays after the sync.
      SmallVector<Operation *> after;
      for (Operation *op = sync->getNextNode();
           op && after.size() < kMaxLookahead; op = op->getNextNode())
        after.push_back(op);
      for (Operation *op : after) {
        if (op->hasTrait<OpTrait::IsTerminator>())
          break;
        Footprint fp = isa<upmem::SyncOp>(op)
                           ? syncFootprint(cast<upmem::SyncOp>(op))
                           : footprintOf(op);
        bool uses = usesAny(op, carriedOps);
        if (uses || conflict(fp, carried)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[async] kept after sync: " << op->getName() << " ("
                     << (uses ? "uses a carried op" : "conflicts")
                     << "; unknown " << fp.unknown << ", reads "
                     << fp.reads.size() << ", writes " << fp.writes.size()
                     << ", sets " << fp.sets.size() << ")\n");
          carriedOps.insert(op);
          carried.append(fp);
          continue;
        }
        op->moveBefore(sync);
      }
    }
  }

  /// Whether `op` or a nested op uses a value defined by one of `defs`.
  static bool usesAny(Operation *op,
                      const llvm::SmallPtrSet<Operation *, 16> &defs) {
    bool found = false;
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands())
        if (Operation *def = v.getDefiningOp(); def && defs.contains(def))
          found = true;
    });
    return found;
  }
};

struct UpmemAsyncLaunchesPass
    : upmem::impl::UpmemAsyncLaunchesPassBase<UpmemAsyncLaunchesPass> {
  using UpmemAsyncLaunchesPassBase::UpmemAsyncLaunchesPassBase;

  void runOnOperation() override {
    Scheduler scheduler(getOperation(), noaliasArguments);
    scheduler.run(getOperation());
  }
};

} // namespace
