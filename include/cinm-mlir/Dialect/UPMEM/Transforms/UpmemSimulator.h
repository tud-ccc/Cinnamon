#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>
#include <memory>

namespace mlir::upmem {

/// Abstract cost estimator for UPMEM programs in UPMEM dialect.
/// Returns an estimated cost (lower is better) given a module.
/// Implementations may range from simple op counts to full simulation.
struct UpmemSimulator {
  virtual ~UpmemSimulator() = default;
  virtual mlir::FailureOr<double> simulate(mlir::ModuleOp module) = 0;
};

/// Simple baseline: weighted op count over the UPMEM dialect IR.
/// Assigns higher weight to data-transfer ops and kernel launches.
std::unique_ptr<UpmemSimulator> createOpCountSimulator();

} // namespace mlir::upmem
