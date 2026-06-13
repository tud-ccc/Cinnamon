#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <algorithm>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
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

// Return the UPMEM accelerator attribute from a cnm buffer type, if present.
static std::optional<UpmemAcceleratorAttr>
upmemAccelOf(cnm::WorkgroupType buf) {
  return llvm::dyn_cast_or_null<UpmemAcceleratorAttr>(buf.getAccelerator());
}

// Total number of elements in a statically shaped tensor/memref.
// Returns 1 if the shape is dynamic (conservative estimate).
static int64_t staticElementCount(ShapedType ty) {
  if (!ty.hasStaticShape())
    return 1;
  return ty.getNumElements();
}

// Byte size of a single element (rounded up to whole bytes).
static double elementBytes(Type elemTy) {
  if (elemTy.isIntOrFloat())
    return static_cast<double>(elemTy.getIntOrFloatBitWidth()) / 8.0;
  return 4.0; // conservative fallback
}

static double transferCost(double numBytes, int numRanks) {
  return numBytes / 1024 / numRanks;
}

static constexpr llvm::StringLiteral kSimCostAttr = "upmem.sim_cost";

// Forward declaration.
static double costOfRegion(Region &region, bool annotate);

static double costOfOp(Operation &op, bool annotate) {
  double cost =
      llvm::TypeSwitch<Operation *, double>(&op)
          // ── loops
          // ───────────────────────────────────────────────────────────
          .Case([&](LoopLikeOpInterface forOp) {
            int64_t tripCount = 8;
            if (auto tc = forOp.getStaticTripCount())
              tripCount = tc->getZExtValue();
            return costOfRegion(*forOp.getLoopRegions()[0], annotate) *
                   static_cast<double>(tripCount);
          })
          // ── cnm.scatter / cnm.gather ─────────────────────────────────────
          .Case<cnm::ScatterOp, cnm::GatherOp>([](auto scatterOp) {
            auto hostTy = scatterOp.getHostType();
            double bytes = static_cast<double>(staticElementCount(hostTy)) *
                           elementBytes(hostTy.getElementType());
            int numRanks = 1;
            if (auto accel = upmemAccelOf(scatterOp.getWg().getType()))
              numRanks = accel->getNumRanks();
            return transferCost(bytes, numRanks);
          })
          // ── upmem.scatter / upmem.gather
          // ───────────────────────────────────── Same transfer-cost formula as
          // cnm equivalents: total bytes moved across the bus divided by the
          // rank-level parallelism.
          .Case<ScatterOp, GatherOp>([](auto xferOp) {
            auto hier = llvm::cast<DeviceHierarchyType>(
                xferOp.getHierarchy().getType());
            double totalBytes =
                static_cast<double>(xferOp.getDpuBufferSizeInBytes()) *
                hier.getNumRanks() * hier.getNumDpusPerRank();
            return transferCost(totalBytes, hier.getNumRanks());
          })
          // ── upmem.local_transfer
          // ───────────────────────────────────────────── Models the DMA
          // latency for WRAM↔MRAM copies: ~36 cycles per 2 KiB chunk, with a
          // minimum of one chunk for small transfers.
          .Case<LocalTransferOp>([](auto xferOp) {
            auto srcTy = llvm::cast<MemRefType>(xferOp.getSource().getType());
            double bytes = static_cast<double>(staticElementCount(srcTy)) *
                           elementBytes(srcTy.getElementType());
            return 36.0 * std::max(1.0, bytes / 2048.0);
          })
          // ── upmem.wait_for
          // ──────────────────────────────────────────────────── Estimates
          // kernel execution cost by recursing into the DPU program body, then
          // dividing by the number of tasklets (intra-DPU parallelism).
          .Case<WaitForOp>([&](auto waitForOp) -> double {
            auto dpuProgram = waitForOp.getDpuProgram();
            if (!dpuProgram)
              return 1.0;
            auto hier = llvm::cast<DeviceHierarchyType>(
                waitForOp.getDpuSet().getType());
            return costOfRegion(dpuProgram.getBody(), annotate) /
                   hier.getNumTaskletsPerDpu();
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
          // ── default: recurse into sub-regions and charge 1 per leaf op
          // ───────────
          .Default([&](Operation *o) {
            double c = o->getNumRegions() > 0 ? 0.0 : 1.0;
            for (auto &region : o->getRegions())
              c += costOfRegion(region, annotate);
            return c;
          });

  if (annotate)
    op.setAttr(kSimCostAttr,
               FloatAttr::get(Float64Type::get(op.getContext()), cost));
  return cost;
}

static double costOfRegion(Region &region, bool annotate) {
  double cost = 0.0;
  for (auto &block : region)
    for (auto &op : block)
      cost += costOfOp(op, annotate);
  return cost;
}

struct OpCountSimulator : UpmemSimulator {
  bool annotateOpCosts;
  explicit OpCountSimulator(bool annotateOpCosts)
      : annotateOpCosts(annotateOpCosts) {}

  mlir::cinm::utils::Maybe<double> simulate(Region &region) override {
    return costOfRegion(region, true);
  }
};

} // namespace

std::unique_ptr<UpmemSimulator> createOpCountSimulator(bool annotateOpCosts) {
  return std::make_unique<OpCountSimulator>(annotateOpCosts);
}

} // namespace mlir::upmem
