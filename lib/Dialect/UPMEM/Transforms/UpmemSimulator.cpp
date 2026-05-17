#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include "cinm-mlir/Dialect/Cnm/IR/CnmOps.h"
#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Operation.h>

namespace mlir::upmem {

namespace {

// Return the UPMEM accelerator attribute from a cnm buffer type, if present.
static std::optional<UpmemAcceleratorAttr>
upmemAccelOf(cnm::BufferType buf) {
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

// Forward declaration.
static double costOfRegion(Region &region);

static double costOfOp(Operation &op) {
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
    return costOfRegion(forOp->getRegion(0)) * static_cast<double>(tripCount);
  }

  // ── cnm.scatter ────────────────────────────────────────────────────────────
  // Cost proportional to total bytes transferred divided by the number of
  // ranks that transfer data in parallel.
  if (auto scatterOp = dyn_cast<cnm::ScatterOp>(&op)) {
    auto inputTy = scatterOp.getInput().getType();
    double bytes = static_cast<double>(staticElementCount(inputTy)) *
                   elementBytes(inputTy.getElementType());
    double numRanks = 1.0;
    if (auto accel = upmemAccelOf(scatterOp.getBuffer().getType()))
      numRanks = static_cast<double>(accel->getNumRanks());
    return bytes / numRanks;
  }

  // ── cnm.gather ─────────────────────────────────────────────────────────────
  if (auto gatherOp = dyn_cast<cnm::GatherOp>(&op)) {
    auto outputTy =
        llvm::cast<ShapedType>(gatherOp.getOutputBuf().getType());
    double bytes = static_cast<double>(staticElementCount(outputTy)) *
                   elementBytes(outputTy.getElementType());
    double numRanks = 1.0;
    if (auto accel = upmemAccelOf(gatherOp.getBuffer().getType()))
      numRanks = static_cast<double>(accel->getNumRanks());
    return bytes / numRanks;
  }

  // ── cnm.launch ─────────────────────────────────────────────────────────────
  // All DPUs execute in parallel; within each DPU tasklets share execution.
  // Approximate the kernel cost as a fixed constant divided by the number of
  // tasklets (more tasklets → faster per-DPU execution).
  if (auto launchOp = dyn_cast<cnm::LaunchOp>(&op)) {
    double tasklets = 1.0;
    // Infer the accelerator from the first buffer input, if any.
    for (auto input : launchOp.getInputs()) {
      if (auto bufTy = dyn_cast<cnm::BufferType>(input.getType())) {
        if (auto accel = upmemAccelOf(bufTy))
          tasklets = static_cast<double>(accel->getNumTaskletsPerDpu());
        break;
      }
    }
    return 1000.0 / tasklets;
  }

  // ── default: recurse into sub-regions and charge 1 per leaf op ─────────────
  double cost = op.getNumRegions() > 0 ? 0.0 : 1.0;
  for (auto &region : op.getRegions())
    cost += costOfRegion(region);
  return cost;
}

static double costOfRegion(Region &region) {
  double cost = 0.0;
  for (auto &block : region)
    for (auto &op : block)
      cost += costOfOp(op);
  return cost;
}

struct OpCountSimulator : UpmemSimulator {
  mlir::cinm::utils::Maybe<double> simulate(Region &module) override {
    return costOfRegion(module);
  }
};

} // namespace

std::unique_ptr<UpmemSimulator> createOpCountSimulator() {
  return std::make_unique<OpCountSimulator>();
}

} // namespace mlir::upmem
