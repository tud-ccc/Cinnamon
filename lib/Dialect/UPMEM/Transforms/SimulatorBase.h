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

} // namespace mlir::upmem
