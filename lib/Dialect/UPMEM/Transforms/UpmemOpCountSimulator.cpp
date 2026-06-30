#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include "upmem_cost_model/Types.h"

#include <cmath>
#include <limits>
#include <llvm/Support/Debug.h>
#include <type_traits>
#include <upmem_cost_model/Simulation.h>

#include <algorithm>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <memory>
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

static double costOfRegionCb(Region &region, bool annotate,
                             const WaitForCostFn &cb);

static double costOfOpCb(Operation &op, bool annotate,
                         const WaitForCostFn &cb) {
  double cost =
      llvm::TypeSwitch<Operation *, double>(&op)
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
            double bodyCost =
                costOfRegionCb(*forOp.getLoopRegions()[0], annotate, cb);
            return bodyCost * tripCount;
          })
          .Case<arith::AddIOp>([](auto) {
            // Between 1.6 and 10 ns on chios.
            // It's lower with more iterations of the enclosing loop
            return 3e-6;
          })
          .Case<memref::CopyOp>([](memref::CopyOp copyOp) {
            // Experiment: try to account for the copy happening
            // LLVM O3 usually unroll the tile copy loop
            auto hostTy = copyOp.getSource().getType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            double time_ns = 0.63 * pow(bytes, 0.907);
            return time_ns / 1e6; // ns -> ms
          })
          // .Case<memref::LoadOp, memref::StoreOp>([](auto) { return 1e-7; })
          .Case<cnm::ScatterOp, cnm::GatherOp>([](auto scatterOp) {
            auto hostTy = scatterOp.getHostType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            int numRanks = 1;
            if (auto accel = upmemAccelOf(scatterOp.getWg().getType()))
              numRanks = accel->getNumRanks();
            return transferCost(bytes, numRanks);
          })
          .Case<upmem::ScatterOp, upmem::GatherOp>([](auto xferOp) -> double {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            int numDpus = hier.getNumRanks() * hier.getNumDpusPerRank();
            if constexpr (std::is_same_v<decltype(xferOp), upmem::ScatterOp>) {
              return upmem_cm::scatterCostMs(numDpus,
                                             xferOp.getDpuBufferSizeInBytes());
            } else {
              return upmem_cm::gatherCostMs(numDpus,
                                            xferOp.getDpuBufferSizeInBytes());
            }
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
          // alloc/free dpus are not counted as they are considered amortized
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
            // double c = o->getNumRegions() > 0 ? 0.0 : 5e-9;
            double c = 0.0;
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
  for (auto &block : region) {
    for (auto &op : block) {
      cost += costOfOpCb(op, annotate, cb);
      if (!std::isfinite(cost))
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
  double simulateGemv(std::chrono::milliseconds timeout, int nTasklets,
                      int64_t mramRows, int64_t mramCols, int64_t rowTile,
                      int64_t colTile, upmem_cm::DType) override;
  double simulateReduction(std::chrono::milliseconds timeout,
                           cinm::ReduceMethod reduction, int taskletRows,
                           int taskletCols, int64_t mramRows, int64_t mramCols,
                           int64_t wramRows, int64_t wramCols,
                           upmem_cm::DType dty) override;

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

double wramToMramCost(long numelts, int nTasklets, upmem_cm::DType dty) {
  return upmem_cm::lookupDmaLatency(true, upmem_cm::dtypeBytes(dty) * numelts) *
         std::max(1, nTasklets / 2);
}
double mramToWramCost(long numelts, int nTasklets, upmem_cm::DType dty) {
  return upmem_cm::lookupDmaLatency(false,
                                    upmem_cm::dtypeBytes(dty) * numelts) *
         std::max(1.0, nTasklets / 1.5);
}
double dpuOpLatency(upmem_cm::StatOp op, upmem_cm::DType dty) {
  return upmem_cm::lookupStaticLatency(op, dty);
}
double dpuOpLatency(upmem_cm::ArithOp op, upmem_cm::DType dty) {
  return dpuOpLatency(upmem_cm::arithToStatOp(op), dty);
}

double mlir::upmem::OpCountSimulator::simulateReduction(
    std::chrono::milliseconds, cinm::ReduceMethod reduction, int taskletRows,
    int taskletCols, int64_t mramRows, int64_t mramCols, int64_t wramRows,
    int64_t wramCols, upmem_cm::DType dty) {

  // - The DPU receives an <mramRows x mramCols> buffer, it sends back an
  // <mramRows> buffer
  // - The DPU runs taskletRows * taskletCols concurrent tasklets
  // - Each tasklet reduces a buffer <wramRows * wramCols> in a loop this number
  // of times: mramCols / wramCols / taskletCols
  // - When that's done, the partial results of the tasklets running on the same
  // row are reduced, we start over with a different set of rows

  auto init =
      // Move result partial bufs
      mramToWramCost(taskletCols, taskletRows, dty);

  int64_t nRowTiles = mramRows / wramRows / taskletRows;
  int64_t nColTiles = mramCols / wramCols / taskletCols;

  auto cycles =
      init +
      nRowTiles *
          (nColTiles * (mramToWramCost(wramRows * wramCols,
                                       taskletRows * taskletCols, dty) +
                        wramRows * wramCols *
                            (dpuOpLatency(upmem_cm::StatOp::LOAD, dty) +
                             dpuOpLatency(upmemCmOp(reduction), dty) +
                             dpuOpLatency(upmem_cm::StatOp::STORE, dty))) +
           taskletCols * (dpuOpLatency(upmem_cm::StatOp::LOAD, dty) +
                          dpuOpLatency(upmemCmOp(reduction), dty) +
                          dpuOpLatency(upmem_cm::StatOp::STORE, dty)))

      + wramToMramCost(mramRows, 1, dty);

  return cycles / 350'000;
}

double mlir::upmem::OpCountSimulator::simulateGemv(
    std::chrono::milliseconds, int nTasklets, int64_t mramRows,
    int64_t mramCols, int64_t rowTile, int64_t colTile, upmem_cm::DType dty) {

  // result
  auto init = mramToWramCost(rowTile, 1, dty);

  int64_t nRowTiles = mramRows / rowTile / nTasklets;
  int64_t nColTiles = mramCols / colTile;

  double trcost = mramToWramCost(rowTile * colTile, nTasklets, dty) +
                  mramToWramCost(colTile, nTasklets, dty);

  double innerLoopCost = rowTile * colTile *
                         (dpuOpLatency(upmem_cm::StatOp::LOAD, dty) * 2 +
                          dpuOpLatency(upmem_cm::StatOp::MUL, dty) +
                          dpuOpLatency(upmem_cm::StatOp::ADD, dty) +
                          dpuOpLatency(upmem_cm::StatOp::STORE, dty));

  double cycleCount = init + nRowTiles * nColTiles * (trcost + innerLoopCost) +
                      // result write back
                      wramToMramCost(mramRows, 1, dty);
  return cycleCount / 350'000;
}

double simulateHostRegion(Region &region, bool annotate,
                          const WaitForCostFn &waitForCb) {
  return costOfRegionCb(region, annotate, waitForCb);
}

std::unique_ptr<UpmemSimulator> createOpCountSimulator(bool annotateOpCosts) {
  return std::make_unique<OpCountSimulator>(annotateOpCosts);
}

} // namespace mlir::upmem
