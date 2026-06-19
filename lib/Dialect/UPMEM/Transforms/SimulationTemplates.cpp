#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include "upmem_cost_model/Types.h"

#include <cstdint>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <upmem_cost_model/ProgramBuilder.h>

#define DEBUG_TYPE "upmem-cpp-sim"

namespace mlir::upmem {

/// Estimate the cost of the host side of a tiled GEMV (mv2) kernel.
///
/// Transfer costs use the same formula as OpCountSimulator's ScatterOp/GatherOp
/// case via scatterGatherCost().
double UpmemSimulator::simulateFullGemv(std::chrono::milliseconds timeout,
                                        int64_t M, int64_t K, int64_t mramRows,
                                        int64_t mramCols, int64_t wramRows,
                                        int64_t wramCols, int64_t dpuRows,
                                        int64_t dpuCols, int64_t tasklets,
                                        upmem_cm::DType dty) {
  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, upmem_cm::dtypeBytes(dty),
                             std::max(1L, dpuCols * dpuRows / 64), 64);
  };

  // DPU compute cost (one DPU, accounts for tasklet parallelism inside).
  double dpuCost =
      this->simulateGemv(timeout, static_cast<int>(tasklets), mramRows,
                         mramCols, wramRows, wramCols, dty);

  // Per inner-loop (col-tile) iteration: 3 scatters + wait + 1 gather.
  double innerIterCost = xferCost(mramRows * mramCols) // scatter A tile
                         + xferCost(mramCols)          // scatter x tile
                         + xferCost(mramRows)          // scatter y (init)
                         + dpuCost                     // DPU kernel
                         + xferCost(mramRows);         // gather y (result)

  int64_t innerTrips = K / (dpuCols * mramCols);
  int64_t outerTrips = M / (dpuRows * mramRows);
  return static_cast<double>(outerTrips * innerTrips) * innerIterCost;
}
} // namespace mlir::upmem
