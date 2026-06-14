#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <algorithm>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <memory>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

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

static double transferCost(double numBytes, int numRanks) {
  return numBytes / 1024 / numRanks / 100;
}

static constexpr llvm::StringLiteral kSimCostAttr = "upmem.sim_cost";

static double costOfRegionCb(Region &region, bool annotate,
                             const WaitForCostFn &cb);

static double costOfOpCb(Operation &op, bool annotate,
                         const WaitForCostFn &cb) {
  double cost =
      llvm::TypeSwitch<Operation *, double>(&op)
          .Case([&](LoopLikeOpInterface forOp) {
            int64_t tripCount = 8;
            if (auto tc = forOp.getStaticTripCount())
              tripCount = tc->getZExtValue();
            return costOfRegionCb(*forOp.getLoopRegions()[0], annotate, cb) *
                   static_cast<double>(tripCount);
          })
          .Case<memref::CopyOp>([](memref::CopyOp copyOp) {
            auto hostTy = copyOp.getSource().getType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            return transferCost(bytes, 1);
          })
          .Case<cnm::ScatterOp, cnm::GatherOp>([](auto scatterOp) {
            auto hostTy = scatterOp.getHostType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            int numRanks = 1;
            if (auto accel = upmemAccelOf(scatterOp.getWg().getType()))
              numRanks = accel->getNumRanks();
            return transferCost(bytes, numRanks);
          })
          .Case<ScatterOp, GatherOp>([](auto xferOp) {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            double totalBytes =
                static_cast<double>(xferOp.getDpuBufferSizeInBytes()) *
                hier.getNumRanks() * hier.getNumDpusPerRank();
            return transferCost(totalBytes, hier.getNumRanks());
          })
          .Case<LocalTransferOp>([](auto xferOp) {
            auto srcTy = llvm::cast<MemRefType>(xferOp.getSource().getType());
            double bytes = static_cast<double>(staticElementCount(srcTy)) *
                           elementBytes(srcTy.getElementType());
            return 36.0 * std::max(1.0, bytes / 2048.0);
          })
          // Delegate DPU kernel cost to the callback.
          .Case<WaitForOp>([&](auto waitForOp) -> double {
            return cb(waitForOp.getOperation(), annotate);
          })
          .Case<cnm::LaunchOp>([](auto launchOp) -> double {
            if (auto acc = upmemAccelOf(launchOp.getWg().getType())) {
              double c = 1;
              for (auto buf : launchOp.getBody().getArguments())
                if (auto mr = llvm::dyn_cast_or_null<MemRefType>(buf.getType()))
                  c *= mr.getNumElements();
              return c / acc->getNumTaskletsPerDpu();
            }
            return 1.0;
          })
          .Case<arith::ConstantOp, upmem::StaticAllocOp, cinm::YieldOp,
                memref::SubViewOp>([](auto) { return 0.0; })
          .Default([&](Operation *o) {
            double c = o->getNumRegions() > 0 ? 0.0 : 5e-3;
            for (auto &region : o->getRegions())
              c += costOfRegionCb(region, annotate, cb);
            return c;
          });

  if (annotate)
    op.setAttr(kSimCostAttr,
               FloatAttr::get(Float64Type::get(op.getContext()), cost));
  return cost;
}

static double costOfRegionCb(Region &region, bool annotate,
                             const WaitForCostFn &cb) {
  double cost = 0.0;
  for (auto &block : region)
    for (auto &op : block)
      cost += costOfOpCb(op, annotate, cb);
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

  mlir::cinm::utils::Maybe<double> simulate(Region &region) override {
    // Recursive callback: recurse into the DPU program body with the same
    // heuristics, divided by tasklet parallelism.
    std::function<double(Operation *, bool)> waitForCb;
    waitForCb = [&](Operation *op, bool ann) -> double {
      auto waitFor = llvm::cast<WaitForOp>(op);
      auto dpuProgram = waitFor.getDpuProgram();
      if (!dpuProgram)
        return 1.0;
      auto hier =
          llvm::cast<DeviceHierarchyType>(waitFor.getDpuSet().getType());
      return simulateHostRegion(dpuProgram.getBody(), ann, waitForCb) /
             hier.getNumTaskletsPerDpu();
    };
    return simulateHostRegion(region, annotateOpCosts, waitForCb);
  }
};

} // namespace

double simulateHostRegion(Region &region, bool annotate,
                          const WaitForCostFn &waitForCb) {
  return costOfRegionCb(region, annotate, waitForCb);
}

std::unique_ptr<UpmemSimulator> createOpCountSimulator(bool annotateOpCosts) {
  return std::make_unique<OpCountSimulator>(annotateOpCosts);
}

} // namespace mlir::upmem
