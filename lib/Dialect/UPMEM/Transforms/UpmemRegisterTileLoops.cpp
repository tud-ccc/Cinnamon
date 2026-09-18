//===- UpmemRegisterTileLoops.cpp - Register-bounded unroll-and-jam -------===//
//
// Shapes a DPU kernel's innermost loop nests for the DPU compiler's register
// allocator: unroll-and-jam of the loop across which an operand is reused,
// bounded by a register budget, then a partial unroll of the innermost loop,
// bounded by code size. See the pass description in Passes.td for why.
//
//===----------------------------------------------------------------------===//

#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "upmem-register-tile-loops"

namespace mlir::upmem {

#define GEN_PASS_DEF_UPMEMREGISTERTILELOOPSPASS
#include <cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc>

namespace {

using affine::AffineForOp;

/// Whether `v` is computed from `iv`. Block arguments other than `iv` itself
/// (another loop's induction variable, an iteration argument) do not count.
static bool dependsOnIV(Value v, Value iv) {
  if (v == iv)
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  return llvm::any_of(def->getOperands(),
                      [&](Value o) { return dependsOnIV(o, iv); });
}

/// A load or store seen as an address: the buffer and the values it is
/// indexed by. Affine accesses index through a map; memref accesses directly.
struct Access {
  Operation *op;
  Value memref;
  SmallVector<Value> indices;
  AffineMap map; // Null for a memref access.
  bool isStore;

  static std::optional<Access> of(Operation *op) {
    if (auto ld = dyn_cast<affine::AffineReadOpInterface>(op))
      return Access{op, ld.getMemRef(), llvm::to_vector(ld.getMapOperands()),
                    ld.getAffineMap(), false};
    if (auto st = dyn_cast<affine::AffineWriteOpInterface>(op))
      return Access{op, st.getMemRef(), llvm::to_vector(st.getMapOperands()),
                    st.getAffineMap(), true};
    if (auto ld = dyn_cast<memref::LoadOp>(op))
      return Access{op, ld.getMemRef(), llvm::to_vector(ld.getIndices()),
                    AffineMap(), false};
    if (auto st = dyn_cast<memref::StoreOp>(op))
      return Access{op, st.getMemRef(), llvm::to_vector(st.getIndices()),
                    AffineMap(), true};
    return std::nullopt;
  }

  bool dependsOn(Value iv) const {
    return llvm::any_of(indices, [&](Value i) { return dependsOnIV(i, iv); });
  }

  /// Whether both name the same element on every iteration.
  bool sameAddressAs(const Access &other) const {
    return memref == other.memref && map == other.map &&
           indices == other.indices;
  }
};

/// What the innermost loop's body keeps in registers, per copy of the parent
/// loop's iteration and shared between copies.
struct Pressure {
  /// Loads whose address moves with the parent loop: an accumulator (invariant
  /// in the inner loop) or an operand indexed by both loops. Each jammed copy
  /// needs its own. Iteration arguments of the inner loop count here too.
  unsigned perCopy = 0;
  /// Loads whose address moves with the inner loop only: the operand the jam
  /// exists to share. Loaded once per inner iteration, whatever the factor.
  unsigned reused = 0;
  /// Loads whose address moves with neither loop: hoisted once.
  unsigned hoisted = 0;
};

static bool isInnermost(AffineForOp forOp) {
  bool innermost = true;
  forOp.getBody()->walk([&](AffineForOp) { innermost = false; });
  return innermost;
}

static unsigned countBodyOps(AffineForOp forOp) {
  unsigned n = 0;
  forOp.getBody()->walk([&](Operation *op) {
    if (!isa<affine::AffineYieldOp>(op))
      ++n;
  });
  return n;
}

/// The bits of `v` that can be significant: those of the value it was
/// extended from, if it was.
static unsigned significantBits(Value v) {
  if (Operation *def = v.getDefiningOp();
      def && isa<arith::ExtSIOp, arith::ExtUIOp>(def))
    v = def->getOperand(0);
  return getElementTypeOrSelf(v.getType()).getIntOrFloatBitWidth();
}

/// Whether the DPU compiler lowers `op` to a call into its runtime library:
/// integer multiplies of operands wider than 16 bits (`__mulsi3`,
/// `__muldi3`; up to 16 bits, even extended to a wider product, it composes
/// the 8x8 `mul_*` instructions), integer division and remainder of any
/// width, and all floating-point arithmetic, which is software-emulated.
static bool lowersToCall(Operation *op) {
  if (auto mul = dyn_cast<arith::MulIOp>(op))
    return significantBits(mul.getLhs()) > 16 ||
           significantBits(mul.getRhs()) > 16;
  if (isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
          arith::CeilDivSIOp, arith::CeilDivUIOp, arith::FloorDivSIOp>(op))
    return true;
  Dialect *dialect = op->getDialect();
  if (!dialect || isa<arith::ConstantOp>(op) ||
      !(isa<arith::ArithDialect>(dialect) || dialect->getNamespace() == "math"))
    return false;
  auto isFloat = [](Type t) { return isa<FloatType>(getElementTypeOrSelf(t)); };
  return llvm::any_of(op->getOperandTypes(), isFloat) ||
         llvm::any_of(op->getResultTypes(), isFloat);
}

/// The registers `forOp`'s body may keep live. A value live across a call
/// has to be in a callee-saved register, of which the DPU ABI has 8
/// (r14-r21) out of 24; a body making calls gets `callRegisterBudget`.
static unsigned bodyRegisterBudget(AffineForOp forOp, unsigned registerBudget,
                                   unsigned callRegisterBudget) {
  WalkResult calls = forOp.getBody()->walk([](Operation *op) {
    return lowersToCall(op) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return calls.wasInterrupted() ? std::min(registerBudget, callRegisterBudget)
                                : registerBudget;
}

/// The unroll-and-jam factor for `parent` over its only inner loop `inner`,
/// or 0 when jamming is unsafe or pointless. Safe means each copy touches
/// locations of its own: every buffer written in the parent's body, inside
/// the inner loop or around it, is accessed only at the written address, and
/// that address moves with the parent's induction variable. Pointless means
/// no operand is shared between copies.
static uint64_t chooseJamFactor(AffineForOp parent, AffineForOp inner,
                                unsigned registerBudget) {
  std::optional<uint64_t> trip = affine::getConstantTripCount(parent);
  if (!trip || *trip < 2 || parent.getNumIterOperands() > 0)
    return 0;

  // The parent's own body: the inner loop, index arithmetic, and loads and
  // stores -- such as those of an accumulator promoted to the inner loop's
  // iteration arguments, loaded before it and stored after. The jam groups
  // every copy's accesses before the inner loop ahead of it, and every copy's
  // accesses after it behind it, which the check on written buffers below
  // makes sound. A transfer or an allocation duplicated per copy would not be
  // the same program.
  SmallVector<Access> outerAccesses;
  for (Operation &op : parent.getBody()->without_terminator()) {
    if (&op == inner.getOperation())
      continue;
    if (auto access = Access::of(&op)) {
      outerAccesses.push_back(*access);
      continue;
    }
    if (!isMemoryEffectFree(&op))
      return 0;
  }

  Value parentIV = parent.getInductionVar();
  Value innerIV = inner.getInductionVar();
  SmallVector<Access> accesses;
  bool safe = true;
  inner.getBody()->walk([&](Operation *op) {
    if (isa<affine::AffineYieldOp>(op))
      return;
    if (auto access = Access::of(op)) {
      accesses.push_back(*access);
      return;
    }
    if (!isMemoryEffectFree(op))
      safe = false;
  });
  if (!safe)
    return 0;

  auto allAccesses = llvm::concat<const Access>(accesses, outerAccesses);
  for (const Access &store : allAccesses) {
    if (!store.isStore)
      continue;
    if (!store.dependsOn(parentIV))
      return 0;
    for (const Access &other : allAccesses)
      if (other.memref == store.memref && !other.sameAddressAs(store))
        return 0;
  }

  Pressure pressure;
  pressure.perCopy = inner.getNumIterOperands();
  for (const Access &access : accesses) {
    if (access.isStore)
      continue;
    if (access.dependsOn(parentIV))
      ++pressure.perCopy;
    else if (access.dependsOn(innerIV))
      ++pressure.reused;
    else
      ++pressure.hoisted;
  }
  if (pressure.reused == 0)
    return 0;

  uint64_t sharedRegs = pressure.reused + pressure.hoisted;
  if (sharedRegs >= registerBudget)
    return 0;
  uint64_t maxFactor =
      pressure.perCopy == 0
          ? *trip
          : std::min<uint64_t>(*trip, (registerBudget - sharedRegs) /
                                          pressure.perCopy);
  if (maxFactor < 2)
    return 0;
  // A factor that divides the trip count leaves no cleanup loop behind.
  for (uint64_t f = maxFactor; f >= 2; --f)
    if (*trip % f == 0)
      return f;
  return maxFactor;
}

/// The allocation or block argument `v` is a view of.
static Value viewRoot(Value v) {
  while (auto view = v.getDefiningOp<ViewLikeOpInterface>())
    v = view.getViewSource();
  return v;
}

/// Whether an op in `loop`'s body may write a view of `root`. An op whose
/// effects are unknown, or that writes without saying where, may.
static bool mayBeWrittenIn(AffineForOp loop, Value root) {
  WalkResult walk = loop.getBody()->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      return WalkResult::interrupt();
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects)
      if (isa<MemoryEffects::Write>(effect.getEffect()) &&
          (!effect.getValue() || viewRoot(effect.getValue()) == root))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walk.wasInterrupted();
}

/// How many values a full unroll of the innermost loop `forOp` makes
/// invariant in the loop around it, per iteration of `forOp`: the distinct
/// loads addressed by `forOp`'s induction variable and not by the parent's,
/// from a buffer the parent does not write. Unrolled, each iteration's copy
/// of such a load holds still across the parent's trips, and loop-invariant
/// code motion hoists it -- all of them, whatever the registers. 0 when there
/// is no enclosing loop left to hoist into.
static unsigned invariantLoadsAfterFullUnroll(AffineForOp forOp) {
  auto parent = dyn_cast<AffineForOp>(forOp->getParentOp());
  if (!parent)
    return 0;
  std::optional<uint64_t> parentTrip = affine::getConstantTripCount(parent);
  if (parentTrip && *parentTrip < 2)
    return 0;
  Value iv = forOp.getInductionVar();
  Value parentIV = parent.getInductionVar();
  SmallVector<Access> loads;
  forOp.getBody()->walk([&](Operation *op) {
    std::optional<Access> access = Access::of(op);
    if (!access || access->isStore || !access->dependsOn(iv) ||
        access->dependsOn(parentIV))
      return;
    if (llvm::any_of(loads,
                     [&](const Access &a) { return a.sameAddressAs(*access); }))
      return;
    if (mayBeWrittenIn(parent, viewRoot(access->memref)))
      return;
    loads.push_back(*access);
  });
  return loads.size();
}

/// The unroll factor for the innermost loop `forOp`: its trip count when that
/// is at most `maxUnroll` or the fully unrolled body fits `maxBodyOps`, else
/// the largest factor in [minUnroll, maxUnroll] dividing the trip count, else
/// `maxUnroll` with a cleanup loop. Lowered while the unrolled body would
/// exceed `maxBodyOps`.
///
/// Not a full unroll when it would hoist more values out of the enclosing loop
/// than `registerBudget`, as a full unroll of the reduction loop under a
/// partially jammed row loop does with its whole operand vector. The loop is
/// then unrolled by a factor that leaves it at least two trips, so that its
/// loads keep moving with it.
static uint64_t chooseUnrollFactor(AffineForOp forOp, unsigned minUnroll,
                                   unsigned maxUnroll, unsigned maxBodyOps,
                                   unsigned registerBudget) {
  std::optional<uint64_t> trip = affine::getConstantTripCount(forOp);
  if (!trip || *trip < 2)
    return 1;
  unsigned bodyOps = countBodyOps(forOp);
  uint64_t factor;
  if (*trip * invariantLoadsAfterFullUnroll(forOp) > registerBudget) {
    uint64_t cap = std::min<uint64_t>(maxUnroll, *trip / 2);
    factor = cap;
    for (uint64_t f = cap; f >= 2; --f) {
      if (*trip % f == 0) {
        factor = f;
        break;
      }
    }
  } else if (*trip <= maxUnroll || *trip * bodyOps <= maxBodyOps) {
    factor = *trip;
  } else {
    factor = maxUnroll;
    for (uint64_t f = maxUnroll; f >= minUnroll; --f) {
      if (*trip % f == 0) {
        factor = f;
        break;
      }
    }
  }
  while (factor > 1 && bodyOps * factor > maxBodyOps)
    --factor;
  return factor;
}

struct UpmemRegisterTileLoopsPass
    : impl::UpmemRegisterTileLoopsPassBase<UpmemRegisterTileLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    DpuProgramOp program = getOperation();

    SmallVector<AffineForOp> innermostLoops;
    program.walk([&](AffineForOp forOp) {
      if (isInnermost(forOp))
        innermostLoops.push_back(forOp);
    });

    // Jam first: the factor is decided on the rolled inner body, and the
    // unroll below then sees the jammed body's size.
    for (AffineForOp inner : innermostLoops) {
      auto parent = dyn_cast<AffineForOp>(inner->getParentOp());
      if (!parent)
        continue;
      uint64_t factor = chooseJamFactor(
          parent, inner,
          bodyRegisterBudget(inner, registerBudget, callRegisterBudget));
      if (factor < 2)
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "unroll-and-jam by " << factor << ": " << parent << "\n");
      if (failed(affine::loopUnrollJamByFactor(parent, factor)))
        LLVM_DEBUG(llvm::dbgs() << "unroll-and-jam failed\n");
    }

    // The jam may have promoted a parent or left a cleanup nest behind, so
    // the innermost loops are collected afresh.
    innermostLoops.clear();
    program.walk([&](AffineForOp forOp) {
      if (isInnermost(forOp))
        innermostLoops.push_back(forOp);
    });
    for (AffineForOp inner : innermostLoops) {
      uint64_t factor = chooseUnrollFactor(
          inner, minUnroll, maxUnroll, maxBodyOps,
          bodyRegisterBudget(inner, registerBudget, callRegisterBudget));
      if (factor < 2)
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "unroll by " << factor << ": " << inner << "\n");
      if (failed(affine::loopUnrollByFactor(inner, factor)))
        LLVM_DEBUG(llvm::dbgs() << "unroll failed\n");
    }
  }
};

} // namespace
} // namespace mlir::upmem
