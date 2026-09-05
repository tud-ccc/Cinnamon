#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>
#include <functional>
#include <upmem_cost_model/Types.h>

namespace mlir::upmem {

/// Callback invoked by simulateHostRegion for each WaitForOp encountered.
/// `op` is the WaitForOp (as Operation*). `annotate` mirrors the outer flag.
/// Must return the estimated cost of that DPU kernel launch.
/// The WaitForOp itself will be annotated by simulateHostRegion using the
/// returned value; the callback need not annotate it.
using WaitForCostFn =
    std::function<SimCost(mlir::Operation * /*WaitForOp*/, bool /*annotate*/)>;

/// Estimates the cost of a host-side UPMEM region (scatter/gather/loops/etc.)
/// and optionally annotates each visited op with 'upmem.sim_cost'.
/// WaitForOp cost is delegated to `waitForCb`; all other op costs use the
/// built-in weighted heuristics (same as OpCountSimulator).
SimCost simulateHostRegion(mlir::Region &region, bool annotate,
                           const WaitForCostFn &waitForCb);

/// simulateHostRegion, refusing rather than returning a cost some part of
/// which could not be computed.
///
/// A non-finite entry is not an expensive configuration, it is one the models
/// were asked something outside what they can answer -- and the walk stops at
/// the first of them, so what comes back is a *truncated* cost missing every
/// op after it, kernel included. Reported as a number that would be a
/// configuration scoring far below its true cost, and one that scores low
/// wins a search. A failure is what the callers already know how to handle:
/// the search records a failed evaluation and moves on.
mlir::cinm::utils::Maybe<SimCost>
simulateHostRegionOrFail(mlir::Region &region, bool annotate,
                         const WaitForCostFn &waitForCb);

} // namespace mlir::upmem
