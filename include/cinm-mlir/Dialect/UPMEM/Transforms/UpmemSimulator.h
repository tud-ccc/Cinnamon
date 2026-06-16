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
  virtual std::unique_ptr<UpmemSimulator> clone() = 0;
  /// Whether this simulator is safe to call concurrently from multiple threads.
  /// If false, exhaustive search will use a single thread.
  virtual bool supportsMultithreading() const { return true; }
  /// Called on the thread that will later call simulate(), before the first
  /// simulate() call.  Implementations may use this to initialise per-thread
  /// resources on the correct thread.
  virtual void warmUp() {}
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

inline double transferCost(double numBytes, int numRanks) {
  return numBytes / 1024 / numRanks / 100;
}

inline double scatterGatherCost(int64_t elemsPerDpu, int64_t elemBytes,
                         int64_t ranks, int64_t dpusPerRank) {
  double totalBytes =
      static_cast<double>(elemsPerDpu * elemBytes) * ranks * dpusPerRank;
  return transferCost(totalBytes, static_cast<int>(ranks));
}

/// Estimates the cost of a host-side UPMEM region (scatter/gather/loops/etc.)
/// and optionally annotates each visited op with 'upmem.sim_cost'.
/// WaitForOp cost is delegated to `waitForCb`; all other op costs use the
/// built-in weighted heuristics (same as OpCountSimulator).
double simulateHostRegion(mlir::Region &region, bool annotate,
                          const WaitForCostFn &waitForCb);

/// Cost model for a single upmem.scatter or upmem.gather operation.
/// Models the off-chip transfer of `elemsPerDpu` elements (each `elemBytes`
/// bytes) to/from all DPUs in a hierarchy of `ranks` ranks × `dpusPerRank`
/// DPUs per rank, assuming all ranks transfer in parallel.
/// Matches the formula used by the OpCount simulator's ScatterOp/GatherOp case.
double scatterGatherCost(int64_t elemsPerDpu, int64_t elemBytes, int64_t ranks,
                         int64_t dpusPerRank);

/// Estimate the total cost of the host-side tiled GEMV (mv2) kernel,
/// including scatter/gather transfers and DPU compute.
///   M, N        — full matrix dimensions
///   mramRows    — output rows per DPU in MRAM
///   mramCols    — input columns per DPU in MRAM
///   wramRows    — row tile size (rowTile for simulateGemv)
///   wramCols    — column tile size (colTile for simulateGemv)
///   ranks       — number of UPMEM ranks
///   dpus        — DPUs per rank
///   tasklets    — tasklets per DPU
double simulateFullGemv(int64_t M, int64_t N, int64_t mramRows,
                        int64_t mramCols, int64_t wramRows, int64_t wramCols,
                        int64_t ranks, int64_t dpus, int64_t tasklets);

} // namespace mlir::upmem
