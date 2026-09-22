#include "SimulatorBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/IR/CnmTypes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h>
#include <cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h>
#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>

#include <upmem_cost_model/ScatterGatherCm.h>
#include <upmem_cost_model/Simulation.h>
#include <upmem_cost_model/Types.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::upmem {

namespace {

static std::optional<UpmemAcceleratorAttr>
upmemAccelOf(cnm::WorkgroupType buf) {
  return llvm::dyn_cast_or_null<UpmemAcceleratorAttr>(buf.getAccelerator());
}

static int64_t staticElementCount(ShapedType ty) {
  if (!ty.hasStaticShape())
    return 1;
  return ty.getNumElements();
}

static double elementBytes(Type elemTy) {
  if (elemTy.isIntOrFloat())
    return static_cast<double>(elemTy.getIntOrFloatBitWidth()) / 8.0;
  return 4.0;
}

/// Whether a serving deployment would pay this transfer once rather than on
/// every inference. Two conditions, the same ones
/// measurements._amortizable_index asks of the runtime's timing rows: the
/// data it moves is the same on every inference (`upmem.timing_tag` says
/// `static:`, decided by the cnm -> upmem conversion, the last stage that
/// could still see where the host value came from), and it runs once per
/// invocation -- a transfer under a loop moves a different tile every trip,
/// so no single load-time transfer replaces it. An untagged transfer is not
/// amortizable: unattributed means unproven.
static bool isAmortizableTransfer(Operation *op) {
  auto tag = op->getAttrOfType<StringAttr>(UPMEMDialect::TIMING_TAG_NAME);
  if (!tag || !tag.getValue().starts_with("static:"))
    return false;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (llvm::isa<LoopLikeOpInterface>(parent))
      return false;
  return true;
}

/// Cost of a host -> device transfer, excluded from the totals when the data
/// stays pinned on the device across inferences (isAmortizableTransfer). The
/// cost keeps its transfer category and label, so the breakdown still reports
/// it; what changes is that SimCost::total() -- what the search minimizes --
/// stops charging it, which is what measurements.net_time_ms does to the same
/// transfers on the measured side.
///
/// The other direction is never excluded: a gather writes a result, which is
/// produced on every inference by definition.
static SimCost scatterCost(Operation *op, double ms, llvm::StringRef label) {
  SimCost cost = SimCost::forTransfer(ms, label);
  if (isAmortizableTransfer(op))
    cost.markExcluded();
  return cost;
}

static SimCost costOfRegionCb(Region &region, bool annotate,
                              const WaitForCostFn &cb,
                              const cinm::HostModel &host);

static int64_t tripCountOf(LoopLikeOpInterface forOp) {
  if (auto tc = forOp.getStaticTripCount())
    return tc->getZExtValue();
  // In the dynamic case, for now we assume a big number divided by
  // the loop step We should use integer range analysis
  int64_t step = 1;
  if (auto steps = forOp.getLoopSteps())
    if (!steps->empty())
      if (auto sv = mlir::getConstantIntValue(steps->front()))
        step = *sv;
  return std::max(1L, 2048 / std::max(1L, step));
}

/// Whether `loop` is a host loop nest of plain arithmetic over memrefs --
/// the shape of the loops that combine a split reduction's partial results
/// -- which LLVM vectorizes, and whose cost is therefore not the sum of its
/// ops at scalar latency. Anything else under the loop (a transfer, a
/// launch, a call) keeps the loop on the per-op path.
static bool isHostArithmeticNest(Operation *loop) {
  if (loop->getParentOfType<DpuProgramOp>())
    return false;
  WalkResult walk = loop->walk([&](Operation *op) {
    if (op == loop)
      return WalkResult::advance();
    llvm::StringRef ns =
        op->getDialect() ? op->getDialect()->getNamespace() : llvm::StringRef();
    if (ns == "arith" || ns == "affine" || ns == "scf" ||
        llvm::isa<memref::LoadOp, memref::StoreOp>(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return !walk.wasInterrupted();
}

/// Vector-instruction time and bytes of memory traffic of the ops in
/// `region`, each multiplied out by `trips`, the product of the trip counts
/// of the loops it sits in. Index arithmetic (affine.apply, and the loops'
/// own induction) folds into addressing and is not counted.
static void accumulateHostNestWork(Region &region, double trips,
                                   const cinm::HostModel &host,
                                   double &computeNs, double &bytes) {
  for (Operation &op : region.getOps()) {
    if (auto loop = llvm::dyn_cast<LoopLikeOpInterface>(&op)) {
      double inner = trips * static_cast<double>(tripCountOf(loop));
      for (Region *body : loop.getLoopRegions())
        accumulateHostNestWork(*body, inner, host, computeNs, bytes);
      continue;
    }
    if (llvm::isa<affine::AffineLoadOp, memref::LoadOp>(op)) {
      bytes += trips * elementBytes(op.getResult(0).getType());
    } else if (auto store = llvm::dyn_cast<affine::AffineStoreOp>(op)) {
      bytes += trips * elementBytes(store.getValueToStore().getType());
    } else if (auto store = llvm::dyn_cast<memref::StoreOp>(op)) {
      bytes += trips * elementBytes(store.getValueToStore().getType());
    } else if (op.getDialect() && op.getDialect()->getNamespace() == "arith" &&
               !llvm::isa<arith::ConstantOp>(op) && op.getNumResults() == 1) {
      computeNs += trips * host.vectorOpNs *
                   elementBytes(op.getResult(0).getType()) / host.vectorBytes;
    }
    for (Region &nested : op.getRegions())
      accumulateHostNestWork(nested, trips, host, computeNs, bytes);
  }
}

/// A host arithmetic nest, priced as a roofline on one core: the larger of
/// its vectorized arithmetic and its memory traffic at streaming bandwidth.
/// The partial-sum loops this is for are bound by the traffic by a wide
/// margin, so charging each add at scalar latency overprices them several
/// times over, and by more the more ways the reduction was split.
static SimCost hostArithmeticNestCost(LoopLikeOpInterface loop,
                                      const cinm::HostModel &host) {
  double computeNs = 0.0, bytes = 0.0;
  double trips = static_cast<double>(tripCountOf(loop));
  for (Region *body : loop.getLoopRegions())
    accumulateHostNestWork(*body, trips, host, computeNs, bytes);
  double memoryNs = bytes / host.streamBytesPerSecond * 1e9;
  return SimCost::forCpu(std::max(computeNs, memoryNs) / 1e6, "other");
}

/// A repack between a host buffer's layout and a transfer's, at the host's
/// copy bandwidth. `shaped` is either side: both hold the same elements.
static double repackMs(ShapedType shaped, const cinm::HostModel &host) {
  double bytes = static_cast<double>(staticElementCount(shaped)) *
                 elementBytes(shaped.getElementType());
  return bytes / host.copyBytesPerSecond * 1e3;
}

static SimCost costOfOpCb(Operation &op, bool annotate, const WaitForCostFn &cb,
                          const cinm::HostModel &host) {
  SimCost cost =
      llvm::TypeSwitch<Operation *, SimCost>(&op)
          .Case([&](LoopLikeOpInterface forOp) {
            if (isHostArithmeticNest(forOp))
              return hostArithmeticNestCost(forOp, host);
            int64_t tripCount = tripCountOf(forOp);
            SimCost bodyCost =
                costOfRegionCb(*forOp.getLoopRegions()[0], annotate, cb, host);
            // Scale each cost component independently by the trip count,
            // rather than collapsing to a single aggregate first.
            return bodyCost * static_cast<double>(tripCount);
          })
          .Case<arith::AddIOp>([&](arith::AddIOp addOp) {
            // Inside a DPU program this is the op-count simulator's flat
            // per-op charge, which has nothing to do with the host.
            if (addOp->getParentOfType<DpuProgramOp>())
              return SimCost::forCpu(3e-6, "other");
            return SimCost::forCpu(host.scalarOpNs / 1e6, "other");
          })
          .Case<cnm::CompactBufferOp>([&](cnm::CompactBufferOp op) {
            auto cost = SimCost::forCpu(
                repackMs(op.getSource().getType(), host), "compact");
            if (cinm::isStaticValue(op.getSource()))
              cost.markExcluded();
            return cost;
          })
          .Case<cnm::ExpandBufferOp>([&](cnm::ExpandBufferOp op) {
            // Writes a gathered result into the host's layout, so it recurs
            // on every inference: never excluded, unlike a compact of
            // static data.
            return SimCost::forCpu(repackMs(op.getSource().getType(), host),
                                   "expand");
          })
          .Case<memref::CopyOp>([&](memref::CopyOp copyOp) {
            return SimCost::forCpu(repackMs(copyOp.getSource().getType(), host),
                                   "copy");
          })
          // .Case<memref::LoadOp, memref::StoreOp>([](auto) { return 1e-7; })
          .Case<upmem::ScatterOnArrayOp>(
              [](upmem::ScatterOnArrayOp xferOp) -> SimCost {
                auto hier = llvm::cast<DeviceHierarchyType>(
                    xferOp.getHierarchy().getType());
                int numDpus = hier.getNumDpus();
                return scatterCost(
                    xferOp,
                    upmem_cm::scatterBlockCostMs(
                        numDpus, xferOp.getDpuBufferSizeInBytes()),
                    "array");
              })
          .Case<upmem::GatherFromArrayOp>(
              [](upmem::GatherFromArrayOp xferOp) -> SimCost {
                auto hier = llvm::cast<DeviceHierarchyType>(
                    xferOp.getHierarchy().getType());
                int numDpus = hier.getNumDpus();
                return SimCost::forTransferBack(
                    upmem_cm::gatherCostMs(numDpus,
                                           xferOp.getDpuBufferSizeInBytes()),
                    "array");
              })
          .Case<upmem::ScatterBlocksOp>([](auto xferOp) -> SimCost {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            int numDpus = hier.getNumDpus();
            return scatterCost(
                xferOp,
                upmem_cm::scatterSgCostMs(numDpus, xferOp.getNumBlocksPerDpu(),
                                          xferOp.getDpuBufferSizeInBytes()),
                "blocks");
          })
          .Case<upmem::GatherBlocksOp>([](auto xferOp) -> SimCost {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            int numDpus = hier.getNumDpus();
            return SimCost::forTransferBack(
                upmem_cm::gatherSgCostMs(numDpus, xferOp.getNumBlocksPerDpu(),
                                         xferOp.getDpuBufferSizeInBytes()),
                "blocks");
          })
          .Case<upmem::BroadcastOp>([](auto xferOp) -> SimCost {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            int numDpus = hier.getNumDpus();
            // Same size is sent to every DPU; model it like a scatter of
            // that buffer's full size.
            return scatterCost(xferOp,
                               upmem_cm::broadcastCostMs(
                                   numDpus, xferOp.getDpuBufferSizeInBytes()),
                               "broadcast");
          })
          .Case<LocalTransferOp>([](auto xferOp) {
            auto srcTy = llvm::cast<MemRefType>(xferOp.getSource().getType());
            double bytes = static_cast<double>(staticElementCount(srcTy)) *
                           elementBytes(srcTy.getElementType());
            // DPU-internal WRAM<->MRAM transfer: contributes to kernel
            // (upmem.wait_for) time, not host<->DPU transfer time.
            return SimCost::forKernel(36.0 * std::max(1.0, bytes / 2048.0),
                                      "local_transfer");
          })
          // Delegate DPU kernel cost to the callback.
          .Case<WaitForOp>([&](auto waitForOp) -> SimCost {
            return cb(waitForOp.getOperation(), annotate);
          })
          // alloc/free dpus are not counted as they are considered amortized
          .Case<cnm::LaunchOp>([](auto launchOp) -> SimCost {
            if (auto acc = upmemAccelOf(launchOp.getWg().getType())) {
              double c = 1;
              for (auto buf : launchOp.getBody().getArguments())
                if (auto mr = llvm::dyn_cast_or_null<MemRefType>(buf.getType()))
                  c *= mr.getNumElements();
              return SimCost::forKernel(c / acc->getNumTaskletsPerDpu(),
                                        "launch");
            }
            return SimCost::forKernel(1.0, "launch");
          })
          .Case<arith::ConstantOp, upmem::StaticAllocOp, cinm::YieldOp,
                memref::SubViewOp>([](auto) { return SimCost{}; })
          .Default([&](Operation *o) {
            SimCost c;
            for (auto &region : o->getRegions())
              c += costOfRegionCb(region, annotate, cb, host);
            return c;
          });

  if (annotate)
    op.setAttr(kSimCostAttr,
               FloatAttr::get(Float64Type::get(op.getContext()), cost.total()));
  return cost;
}

static SimCost costOfRegionCb(Region &region, bool annotate,
                              const WaitForCostFn &cb,
                              const cinm::HostModel &host) {
  SimCost cost;
  for (auto &block : region) {
    for (auto &op : block) {
      cost += costOfOpCb(op, annotate, cb, host);
      if (!cost.isFinite())
        return cost;
    }
  }

  return cost;
}

struct OpCountSimulator : UpmemSimulator {
  bool annotateOpCosts;
  explicit OpCountSimulator(bool annotateOpCosts)
      : annotateOpCosts(annotateOpCosts) {}
  OpCountSimulator(OpCountSimulator &&) = default;

  std::unique_ptr<UpmemSimulator> clone() override {
    return std::make_unique<OpCountSimulator>(annotateOpCosts);
  }
  mlir::cinm::utils::Maybe<SimCost> simulate(Region &region) override {
    // Recursive callback: recurse into the DPU program body with the same
    // heuristics, divided by tasklet parallelism. The callback returns a
    // single scalar (kernel-launch time as seen from the host): whatever
    // happens inside the DPU program body all counts towards the kernel
    // component of the enclosing upmem.wait_for.
    std::function<SimCost(Operation *, bool)> waitForCb;
    waitForCb = [&](Operation *op, bool ann) -> SimCost {
      auto waitFor = llvm::cast<WaitForOp>(op);
      auto dpuProgram = waitFor.getDpuProgram();
      if (!dpuProgram)
        return {};
      auto hier =
          llvm::cast<DeviceHierarchyType>(waitFor.getDpuSet().getType());
      return SimCost::forKernel(
          simulateHostRegion(dpuProgram.getBody(), ann, waitForCb).total() /
              hier.getNumTaskletsPerDpu(),
          "opcount");
    };
    return simulateHostRegionOrFail(region, annotateOpCosts, waitForCb);
  }
};

/// The host `region` runs on, as the nearest #cinm.host_platform around it
/// declares it.
static cinm::HostModel hostOf(Region &region) {
  if (Operation *parent = region.getParentOp())
    return cinm::HostPlatformAttr::getInScope(parent).getModel();
  return cinm::HostModel{};
}

} // namespace

SimCost simulateHostRegion(Region &region, bool annotate,
                           const WaitForCostFn &waitForCb) {
  return costOfRegionCb(region, annotate, waitForCb, hostOf(region));
}

mlir::cinm::utils::Maybe<SimCost>
simulateHostRegionOrFail(Region &region, bool annotate,
                         const WaitForCostFn &waitForCb) {
  SimCost cost = costOfRegionCb(region, annotate, waitForCb, hostOf(region));
  if (cost.isFinite())
    return cost;

  // Name the entry that went non-finite: the whole point of refusing here is
  // that the number is not a cost, and which model produced it is what a
  // reader needs to know next.
  std::string offenders;
  llvm::raw_string_ostream os(offenders);
  cost.forEachEntry(
      [&](CostCategory category, llvm::StringRef label, double value, bool) {
        if (std::isfinite(value))
          return;
        os << (offenders.empty() ? "" : ", ") << costCategoryName(category);
        if (!label.empty())
          os << "." << label;
        os << " = " << value;
      });
  Operation *parent = region.getParentOp();
  return mlir::emitSilenceableFailure(
             parent ? parent->getLoc() : UnknownLoc::get(region.getContext()))
         << "cost model produced a non-finite cost (" << offenders
         << "); the configuration is refused rather than scored, since the "
            "walk stops at the first such op and the remaining cost -- the "
            "kernel included -- is never computed";
}

std::unique_ptr<UpmemSimulator> createOpCountSimulator(bool annotateOpCosts) {
  return std::make_unique<OpCountSimulator>(annotateOpCosts);
}

} // namespace mlir::upmem
