#pragma once

#include <chrono>
#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cstdint>
#include <functional>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <upmem_cost_model/Types.h>

#include <optional>

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
  virtual void printStats() const {}

  virtual double simulateGemv(std::chrono::milliseconds timeout, int nTasklets,
                              int64_t mramRows, int64_t mramCols,
                              int64_t rowTile, int64_t colTile,
                              upmem_cm::DType dty) = 0;

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
  double simulateFullGemv(std::chrono::milliseconds timeoutMs, int64_t M,
                          int64_t K, int64_t mramRows, int64_t mramCols,
                          int64_t wramRows, int64_t wramCols, int64_t dpuRows,
                          int64_t dpuCols, int64_t tasklets,
                          upmem_cm::DType dty);

  virtual double simulateReduction(std::chrono::milliseconds timeout,
                                   cinm::ReduceMethod reduction,
                                   int taskletRows, int taskletCols,
                                   int64_t mramRows, int64_t mramCols,
                                   int64_t wramRows, int64_t wramCols,
                                   upmem_cm::DType dty) = 0;

  /// Simulate a reduction operation.
  /// The reduction is like reducing a tensor <MxK> into a tensor <M>.
  /// The M rows are tiled into dpus, tasklets, mram and wram.
  /// The K cols are also tiled into dpus, tasklets, mram and wram and
  /// influence a partial reductions.

  double simulateTailReduction(std::chrono::milliseconds timeoutMs, int64_t M,
                               int64_t K, cinm::ReduceMethod reduction,
                               int64_t mramRows, int64_t mramCols,
                               int64_t wramRows, int64_t wramCols,
                               int64_t dpuRows, int64_t dpuCols,
                               int64_t taskletRows, int64_t taskletCols,
                               upmem_cm::DType dty);
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
std::unique_ptr<UpmemSimulator> createPythonSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0));

inline std::unique_ptr<UpmemSimulator> createSimulator(
    StringRef simulator, bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0)) {
  if (simulator == "cycleaccurate")
    return createPythonSimulator(annotateOpCosts, timeoutMs);
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

inline upmem_cm::ArithOp upmemCmOp(cinm::ReduceMethod red) {
  switch (red) {
  case cinm::ReduceMethod::ADD:
    return upmem_cm::ArithOp::ADD;
  case cinm::ReduceMethod::MUL:
    return upmem_cm::ArithOp::MUL;
  default:
    // todo the simulator doesn't have measurements for the remaining
    // operations.
    assert(false && "Unsupported reduce method");
  }
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

} // namespace mlir::upmem

namespace mlir {

/// Emit the host-side tiled loop nest for a tail reduction.
///
/// Replaces the body of a cinm::ReduceOp with two nested loops (M then K)
/// each of whose bodies scatters the input tile and the running partial-sum
/// buffer to all DPUs, fires the kernel, and gathers the updated partial sum
/// back. Data is staged through flat-DPU-major host buffers; the actual
/// DPU kernel and its MRAM symbol declarations must be set up by the caller.
///
///   input  - memref<M x K x elt> (caller collapses leading dims first)
///   output - memref<M x elt>
///   dpus   - !upmem.hierarchy<1 x (dpuRows*dpuCols) x tasklets>
///   aBufSym / yBufSym - symbol names of the MRAM buffers inside the DPU program
void generateTailReduction(cinm::ReduceOp op, RewriterBase &rewriter,
                            Value input, Value output, Value dpus,
                            int64_t M, int64_t K,
                            int64_t dpuRows, int64_t dpuCols,
                            int64_t mramRows, int64_t mramCols,
                            StringRef aBufSym, StringRef yBufSym);

} // namespace mlir
