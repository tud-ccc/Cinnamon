#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <memory>

namespace mlir::upmem {

/// Abstract cost estimator for UPMEM programs in UPMEM dialect.
/// Returns an estimated cost (lower is better) given a module.
/// Implementations may range from simple op counts to full simulation.
struct UpmemSimulator {
  virtual ~UpmemSimulator() = default;
  virtual cinm::utils::Maybe<double> simulate(mlir::Region& region) = 0;
};

/// Simple baseline: weighted op count over the UPMEM dialect IR.
/// Assigns higher weight to data-transfer ops and kernel launches.
/// When annotateOpCosts is true, each visited op is tagged with a
/// 'upmem.sim_cost' FloatAttr containing its individual simulated cost.
std::unique_ptr<UpmemSimulator> createOpCountSimulator(bool annotateOpCosts = false);

/// Creates a UpmemSimulator that translates the lowered UPMEM DPU program to
/// the Python upmem_simulator high-level IR and runs cycle-accurate simulation.
/// Requires the upmem_simulator Python package to be importable.
/// Falls back to the op-count simulator on any Python error.
std::unique_ptr<UpmemSimulator> createPythonSimulator();

} // namespace mlir::upmem
