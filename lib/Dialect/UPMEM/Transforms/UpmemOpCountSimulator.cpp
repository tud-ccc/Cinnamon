#include "SimulatorBase.h"
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

static SimCost costOfRegionCb(Region &region, bool annotate,
                              const WaitForCostFn &cb);

static SimCost costOfOpCb(Operation &op, bool annotate,
                          const WaitForCostFn &cb) {
  SimCost cost =
      llvm::TypeSwitch<Operation *, SimCost>(&op)
          .Case([&](LoopLikeOpInterface forOp) {
            int64_t tripCount;
            if (auto tc = forOp.getStaticTripCount()) {
              tripCount = tc->getZExtValue();
            } else {
              // In the dynamic case, for now we assume a big number divided by
              // the loop step We should use integer range analysis
              int64_t step = 1;
              if (auto steps = forOp.getLoopSteps())
                if (!steps->empty())
                  if (auto sv = mlir::getConstantIntValue(steps->front()))
                    step = *sv;
              tripCount = std::max(1L, 2048 / std::max(1L, step));
            }
            SimCost bodyCost =
                costOfRegionCb(*forOp.getLoopRegions()[0], annotate, cb);
            // Scale each cost component independently by the trip count,
            // rather than collapsing to a single aggregate first.
            return bodyCost * static_cast<double>(tripCount);
          })
          .Case<arith::AddIOp>([](auto) {
            // Between 1.6 and 10 ns on chios.
            // It's lower with more iterations of the enclosing loop
            return SimCost::forCpu(3e-6, "other");
          })
          .Case<memref::CopyOp>([](memref::CopyOp copyOp) {
            // Experiment: try to account for the copy happening
            // LLVM O3 usually unroll the tile copy loop
            auto hostTy = copyOp.getSource().getType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            double time_ns = 0.63 * pow(bytes, 0.907);
            return SimCost::forCpu(time_ns / 1e6, "copy"); // ns -> ms
          })
          // .Case<memref::LoadOp, memref::StoreOp>([](auto) { return 1e-7; })
          .Case<cnm::ScatterOp>([](cnm::ScatterOp scatterOp) {
            auto hostTy = scatterOp.getHostType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            int numRanks = 1;
            if (auto accel = upmemAccelOf(scatterOp.getWg().getType()))
              numRanks =
                  accel->getPlatform().numRanksForTransfer(accel->getNumDpus());
            return SimCost::forTransfer(transferCost(bytes, numRanks),
                                        "scatter");
          })
          .Case<cnm::GatherOp>([](cnm::GatherOp gatherOp) {
            auto hostTy = gatherOp.getHostType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            int numRanks = 1;
            if (auto accel = upmemAccelOf(gatherOp.getWg().getType()))
              numRanks =
                  accel->getPlatform().numRanksForTransfer(accel->getNumDpus());
            return SimCost::forTransferBack(transferCost(bytes, numRanks),
                                            "gather");
          })
          .Case<upmem::ScatterOnArrayOp>(
              [](upmem::ScatterOnArrayOp xferOp) -> SimCost {
                auto hier = llvm::cast<DeviceHierarchyType>(
                    xferOp.getHierarchy().getType());
                int numDpus = hier.getNumDpus();
                return SimCost::forTransfer(
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
            // getDpuBufferSizeInBytes() is the size of a single block; the
            // actual per-DPU transfer covers numBlocksPerDpu of them.
            return SimCost::forTransfer(
                upmem_cm::scatterSgCostMs(numDpus,
                                          xferOp.getDpuBufferSizeInBytes(),
                                          xferOp.getNumBlocksPerDpu()),
                "blocks");
          })
          .Case<upmem::GatherBlocksOp>([](auto xferOp) -> SimCost {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            int numDpus = hier.getNumDpus();
            // No sg-specific gather cost has been characterized yet, so this
            // charges the flat gather rate for the whole per-DPU volume --
            // an underestimate whenever the blocks are scattered.
            return SimCost::forTransferBack(
                upmem_cm::gatherCostMs(numDpus,
                                       xferOp.getDpuBufferSizeInBytes() *
                                           xferOp.getNumBlocksPerDpu()),
                "blocks");
          })
          .Case<upmem::BroadcastOp>([](auto xferOp) -> SimCost {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            int numDpus = hier.getNumDpus();
            // Same size is sent to every DPU; model it like a scatter of
            // that buffer's full size.
            return SimCost::forTransfer(
                upmem_cm::broadcastCostMs(numDpus,
                                          xferOp.getDpuBufferSizeInBytes()),
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
              c += costOfRegionCb(region, annotate, cb);
            return c;
          });

  if (annotate)
    op.setAttr(kSimCostAttr,
               FloatAttr::get(Float64Type::get(op.getContext()), cost.total()));
  return cost;
}

static SimCost costOfRegionCb(Region &region, bool annotate,
                              const WaitForCostFn &cb) {
  SimCost cost;
  for (auto &block : region) {
    for (auto &op : block) {
      cost += costOfOpCb(op, annotate, cb);
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
    return simulateHostRegion(region, annotateOpCosts, waitForCb);
  }
};

} // namespace

SimCost simulateHostRegion(Region &region, bool annotate,
                           const WaitForCostFn &waitForCb) {
  return costOfRegionCb(region, annotate, waitForCb);
}

std::unique_ptr<UpmemSimulator> createOpCountSimulator(bool annotateOpCosts) {
  return std::make_unique<OpCountSimulator>(annotateOpCosts);
}

} // namespace mlir::upmem
