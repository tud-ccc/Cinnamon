#pragma once

#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <functional>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::upmem {

/// Abstract cost estimator for UPMEM programs in UPMEM dialect.
/// Returns an estimated cost (lower is better) given a region.
/// Implementations may range from simple op counts to full simulation.
struct UpmemSimulator {
  virtual ~UpmemSimulator() = default;
  virtual cinm::utils::Maybe<double> simulate(mlir::Region &region) = 0;
};

/// Simple baseline: weighted op count over the UPMEM dialect IR.
/// Assigns higher weight to data-transfer ops and kernel launches.
/// When annotateOpCosts is true, each visited op is tagged with a
/// 'upmem.sim_cost' FloatAttr containing its individual simulated cost.
std::unique_ptr<UpmemSimulator>
createOpCountSimulator(bool annotateOpCosts = false);

/// Creates a UpmemSimulator that translates the lowered UPMEM DPU program to
/// the Python upmem_simulator high-level IR and runs cycle-accurate simulation.
/// Requires the upmem_simulator Python package to be importable.
/// Falls back to the op-count simulator on any Python error.
/// When annotateOpCosts is true, annotates host-side ops with 'upmem.sim_cost'
/// and annotates each WaitForOp with the Python-estimated cycle count.
std::unique_ptr<UpmemSimulator>
createPythonSimulator(bool annotateOpCosts = false);

inline std::unique_ptr<UpmemSimulator>
createSimulator(StringRef simulator, bool annotateOpCosts = false) {
  if (simulator == "cycleaccurate")
    return createPythonSimulator(annotateOpCosts);
  else if (simulator == "opcount")
    return createOpCountSimulator(annotateOpCosts);

  return nullptr;
}

/// Callback invoked by simulateHostRegion for each WaitForOp encountered.
/// `op` is the WaitForOp (as Operation*). `annotate` mirrors the outer flag.
/// Must return the estimated cost of that DPU kernel launch.
/// The WaitForOp itself will be annotated by simulateHostRegion using the
/// returned value; the callback need not annotate it.
using WaitForCostFn =
    std::function<double(mlir::Operation * /*WaitForOp*/, bool /*annotate*/)>;

/// Estimates the cost of a host-side UPMEM region (scatter/gather/loops/etc.)
/// and optionally annotates each visited op with 'upmem.sim_cost'.
/// WaitForOp cost is delegated to `waitForCb`; all other op costs use the
/// built-in weighted heuristics (same as OpCountSimulator).
double simulateHostRegion(mlir::Region &region, bool annotate,
                          const WaitForCostFn &waitForCb);

} // namespace mlir::upmem
