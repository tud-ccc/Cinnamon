#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMAttributes.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/UpmemSimulator.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <cstdint>
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

#include <memory>
#include <string>

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
double simulateGemv(int nTasklets, int64_t mramRows, int64_t mramCols,
                    int64_t rowTile, int64_t colTile) {
  using namespace upmem_cm;
  ProgramBuilder b;

  // MRAM buffers
  auto A_mram = b.addBuffer("A_mram", MemSpace::MRAM, DType::I32);
  auto x_mram = b.addBuffer("x_mram", MemSpace::MRAM, DType::I32);
  auto y_mram = b.addBuffer("y_mram", MemSpace::MRAM, DType::I32);

  // WRAM tile buffers
  auto A_wram = b.addBuffer("A_wram", MemSpace::WRAM, DType::I32);
  auto x_wram = b.addBuffer("x_wram", MemSpace::WRAM, DType::I32);
  auto y_wram = b.addBuffer("y_wram", MemSpace::WRAM, DType::I32);

  // Load all y from MRAM to WRAM before loops
  b.createTransfer(y_mram, y_wram, mramRows);

  int64_t nRowTiles = mramRows / rowTile;
  int64_t nColTiles = mramCols / colTile;

  b.beginLoop(0, nRowTiles); // row tile loop
  b.beginLoop(0, nColTiles); // col tile loop

  // Transfer A tile [rowTile × colTile] from MRAM; address advances per
  // col-tile iter
  b.createTransfer(A_mram, A_wram, rowTile * colTile, /*src_iv_indexed=*/true);
  // Transfer x tile [colTile] from MRAM; address advances per col-tile iter
  // (in the kernel only tasklet 0 does this via scf.if; modeled
  // unconditionally)
  b.createTransfer(x_mram, x_wram, colTile, /*src_iv_indexed=*/true);

  b.beginLoop(0, rowTile); // row loop within tile
  b.beginLoop(0, colTile); // dot-product loop

  // acc += A_wram[row, col] * x_wram[col]; both stride by 1 per inner iteration
  auto a_val = b.createLoad(A_wram, /*iv_indexed=*/true);
  auto x_val = b.createLoad(x_wram, /*iv_indexed=*/true);
  auto prod = b.createArith(ArithOp::MUL, DType::I32, a_val, x_val);
  // load-add-store into y_wram (models the iter_args reduction pattern)
  b.createReduceStore(y_wram, ArithOp::ADD, prod);

  b.endLoop(); // dot-product loop
  b.endLoop(); // row loop within tile
  b.endLoop(); // col tile loop
  b.endLoop(); // row tile loop

  // Store all y from WRAM back to MRAM
  b.createTransfer(y_wram, y_mram, mramRows);

  return b.simulate(nTasklets);
}

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
double mlir::upmem::simulateFullGemv(int64_t M, int64_t N, int64_t mramRows,
                                     int64_t mramCols, int64_t wramRows,
                                     int64_t wramCols, int64_t ranks,
                                     int64_t dpus, int64_t tasklets) {
  // Cost of one scatter/gather of `elemsPerDpu` i32 elements across all DPUs.
  auto xferCost = [&](int64_t elemsPerDpu) {
    return scatterGatherCost(elemsPerDpu, /*elemBytes=*/4, ranks, dpus);
  };

  // DPU compute cost (one DPU, accounts for tasklet parallelism inside).
  double dpuCost = simulateGemv(static_cast<int>(tasklets), mramRows, mramCols,
                                wramRows, wramCols);

  // Per inner-loop (col-tile) iteration: 3 scatters + wait + 1 gather.
  double innerIterCost = xferCost(mramRows * wramCols) // scatter A tile
                         + xferCost(wramCols)          // scatter x tile
                         + xferCost(mramRows)          // scatter y (init)
                         + dpuCost                     // DPU kernel
                         + xferCost(mramRows);         // gather y (result)

  int64_t innerTrips = N / wramCols;
  int64_t outerTrips = M / (ranks * dpus * mramRows);
  return static_cast<double>(outerTrips * innerTrips) * innerIterCost;
}
