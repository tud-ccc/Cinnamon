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

/// Simulate the mv2 DPU kernel (tiled matrix-vector product) using the
/// cycle-accurate ProgramBuilder.
///
/// The kernel structure mirrors mv2.6.upmem.mlir:
///   - A in MRAM: nTasklets × mramRows × mramCols (i32)
///   - x in MRAM: mramCols (i32)
///   - y in MRAM: nTasklets × mramRows (i32)
///   - A tile in WRAM: rowTile × colTile
///   - x tile in WRAM: colTile
///   - y accumulator in WRAM: mramRows
///
/// Execution:
///   1. Load all y from MRAM → WRAM
///   2. For each row tile (mramRows / rowTile iterations):
///      For each col tile (mramCols / colTile iterations):
///        Transfer A tile (MRAM → WRAM, rowTile × colTile elems)
///        Transfer x tile (MRAM → WRAM, colTile elems)  [tasklet-0 only;
///        modeled unconditionally] For each row in tile:
///          Dot-product loop over colTile: y_wram[row] += A_wram[row,col] *
///          x_wram[col]
///   3. Store all y from WRAM → MRAM
///
/// Returns the estimated wall-clock time in seconds for one DPU.
namespace mlir::upmem {

/// Estimate the cost of the host side of a tiled GEMV (mv2) kernel.
///
/// Loop structure (from the MLIR host region):
///   outer: 0 → M  step (ranks*dpus*mramRows)   [one batch of rows per
///   iteration]
///     inner: 0 → N  step wramCols               [one column tile per
///     iteration]
///       scatter A tile : mramRows*wramCols elems/DPU  (i32)
///       scatter x tile : wramCols elems/DPU           (i32, broadcast)
///       scatter y tile : mramRows elems/DPU            (i32, initial values)
///       wait_for       : DPU compute cost
///       gather  y tile : mramRows elems/DPU            (i32, results)
///
/// Transfer costs use the same formula as OpCountSimulator's ScatterOp/GatherOp
/// case via scatterGatherCost().
double UpmemSimulator::simulateFullGemv(std::chrono::milliseconds timeout,
                                        int64_t M, int64_t N, int64_t mramRows,
                                        int64_t mramCols, int64_t wramRows,
                                        int64_t wramCols, int64_t ranks,
                                        int64_t dpus, int64_t tasklets,
                                        upmem_cm::DType dty) {
  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, upmem_cm::dtypeBytes(dty), ranks,
                             dpus);
  };

  // DPU compute cost (one DPU, accounts for tasklet parallelism inside).
  double dpuCost =
      this->simulateGemv(timeout, static_cast<int>(tasklets), mramRows,
                         mramCols, wramRows, wramCols, dty);

  // Per inner-loop (col-tile) iteration: 3 scatters + wait + 1 gather.
  double innerIterCost =
      xferCost(tasklets * mramRows * mramCols) // scatter A tile
      + xferCost(mramCols)                     // scatter x tile
      + xferCost(mramRows)                     // scatter y (init)
      + dpuCost                                // DPU kernel
      + xferCost(tasklets * mramRows);         // gather y (result)

  int64_t innerTrips = N / mramCols;
  int64_t outerTrips = M / (ranks * dpus * mramRows * tasklets);
  return static_cast<double>(outerTrips * innerTrips) * innerIterCost;
}
} // namespace mlir::upmem
