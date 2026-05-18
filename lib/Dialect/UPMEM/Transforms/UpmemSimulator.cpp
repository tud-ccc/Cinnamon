#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

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
  double cost;

  // ── affine.for ─────────────────────────────────────────────────────────────
  if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
    int64_t tripCount = 1;
    if (forOp.hasConstantBounds()) {
      int64_t lb = forOp.getConstantLowerBound();
      int64_t ub = forOp.getConstantUpperBound();
      int64_t step = forOp.getStepAsInt();
      if (step > 0 && ub > lb)
        tripCount = (ub - lb + step - 1) / step;
    }
    cost = costOfRegion(forOp->getRegion(0), annotate) *
           static_cast<double>(tripCount);
  }

  // ── cnm.scatter ────────────────────────────────────────────────────────────
  // Cost proportional to total bytes transferred divided by the number of
  // ranks that transfer data in parallel.
  else if (auto scatterOp = dyn_cast<cnm::ScatterOp>(&op)) {
    auto inputTy = scatterOp.getInput().getType();
    double bytes = static_cast<double>(staticElementCount(inputTy)) *
                   elementBytes(inputTy.getElementType());
    int numRanks = 1;
    if (auto accel = upmemAccelOf(scatterOp.getWg().getType()))
      numRanks = accel->getNumRanks();
    cost = transferCost(bytes, numRanks);
  }

  // ── cnm.gather ─────────────────────────────────────────────────────────────
  else if (auto gatherOp = dyn_cast<cnm::GatherOp>(&op)) {
    auto outputTy = llvm::cast<ShapedType>(gatherOp.getOutputBuf().getType());
    double bytes = static_cast<double>(staticElementCount(outputTy)) *
                   elementBytes(outputTy.getElementType());
    int numRanks = 1;
    if (auto accel = upmemAccelOf(gatherOp.getWg().getType()))
      numRanks = accel->getNumRanks();
    cost = transferCost(bytes, numRanks);
  }

  // ── cnm.launch ─────────────────────────────────────────────────────────────
  // All DPUs execute in parallel; within each DPU tasklets share execution.
  // Approximate the kernel cost as a fixed constant divided by the number of
  // tasklets (more tasklets → faster per-DPU execution).
  else if (auto launchOp = dyn_cast<cnm::LaunchOp>(&op)) {
    if (auto acc = upmemAccelOf(launchOp.getWg().getType())) {
      double c = 1;
      for (auto buf : launchOp.getBody().getArguments()) {
        if (auto mr = llvm::dyn_cast_or_null<MemRefType>(buf.getType())) {
          c *= mr.getNumElements();
        }
      }
      cost = c / static_cast<double>(acc->getNumTaskletsPerDpu());
    } else {
      cost = 1.0;
    }
  }

  // ── default: recurse into sub-regions and charge 1 per leaf op ─────────────
  else {
    cost = op.getNumRegions() > 0 ? 0.0 : 1.0;
    for (auto &region : op.getRegions())
      cost += costOfRegion(region, annotate);
  }

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
