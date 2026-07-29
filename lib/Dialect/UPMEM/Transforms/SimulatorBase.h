#include <cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h>
#include <functional>
#include <upmem_cost_model/Types.h>

namespace mlir::upmem {

inline upmem_cm::DType from_upmem_dty(upmem::DType dty) {
  switch (dty) {
  case DType::U8:
    return upmem_cm::DType::U8;
  case DType::I8:
    return upmem_cm::DType::I8;
  case DType::U16:
    return upmem_cm::DType::U16;
  case DType::I16:
    return upmem_cm::DType::I16;
  case DType::U32:
    return upmem_cm::DType::U32;
  case DType::I32:
    return upmem_cm::DType::I32;
  case DType::F32:
    return upmem_cm::DType::F32;
  case DType::U64:
    return upmem_cm::DType::U64;
  case DType::I64:
    return upmem_cm::DType::I64;
  case DType::F64:
    return upmem_cm::DType::F64;
  }
}

/// Callback invoked by simulateHostRegion for each WaitForOp encountered.
/// `op` is the WaitForOp (as Operation*). `annotate` mirrors the outer flag.
/// Must return the estimated cost of that DPU kernel launch.
/// The WaitForOp itself will be annotated by simulateHostRegion using the
/// returned value; the callback need not annotate it.
using WaitForCostFn =
    std::function<SimCost(mlir::Operation * /*WaitForOp*/, bool /*annotate*/)>;

inline double transferCost(double numBytes, int numRanks) {
  return numBytes / 1024 / numRanks / 100'000;
}

inline double scatterGatherCost(int64_t elemsPerDpu, DType elemTy,
                                int64_t ranks, int64_t dpusPerRank) {
  double totalBytes =
      static_cast<double>(elemsPerDpu * dtypeBytes(elemTy)) * ranks * dpusPerRank;
  return transferCost(totalBytes, static_cast<int>(ranks));
}

inline upmem_cm::ArithOp upmemCmOp(mlir::cinm::ReduceMethod red) {
  switch (red) {
  case mlir::cinm::ReduceMethod::ADD:
    return upmem_cm::ArithOp::ADD;
  case mlir::cinm::ReduceMethod::MUL:
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
SimCost simulateHostRegion(mlir::Region &region, bool annotate,
                          const WaitForCostFn &waitForCb);

/// Cost model for a single upmem.scatter or upmem.gather operation.
/// Models the off-chip transfer of `elemsPerDpu` elements (each `elemBytes`
/// bytes) to/from all DPUs in a hierarchy of `ranks` ranks × `dpusPerRank`
/// DPUs per rank, assuming all ranks transfer in parallel.
/// Matches the formula used by the OpCount simulator's ScatterOp/GatherOp case.
double scatterGatherCost(int64_t elemsPerDpu, int64_t elemBytes, int64_t ranks,
                         int64_t dpusPerRank);
} // namespace mlir::upmem