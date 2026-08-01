#pragma once

#include <array>
#include <chrono>
#include <cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h>
#include <cinm-mlir/Dialect/Cinm/IR/CinmOps.h>
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <optional>
#include <string>
#include <utility>

namespace mlir::upmem {
enum class DType : uint8_t { U8, I8, U16, I16, U32, I32, F32, U64, I64, F64 };

inline int dtypeBits(DType dt) {
  switch (dt) {
  case DType::U8:
  case DType::I8:
    return 8;
  case DType::U16:
  case DType::I16:
    return 16;
  case DType::U32:
  case DType::I32:
  case DType::F32:
    return 32;
  case DType::U64:
  case DType::I64:
  case DType::F64:
    return 64;
  }
  return 32;
}

inline int dtypeBytes(DType dt) { return dtypeBits(dt) / 8; }

/// SimCost's categories, as used by UPMEM simulators:
///   Kernel       — DPU kernel launch/execution time (upmem.wait_for)
///   Cpu          — host CPU time (memref.copy and other host instructions)
///   Transfer     — host->DPU transfers (upmem.scatter,
///                  upmem.scatter_blocks, upmem.broadcast)
///   TransferBack — DPU->host transfers (upmem.gather)
using cinm::utils::CostCategory;
using cinm::utils::costCategoryName;
using cinm::utils::kNumCostCategories;
using cinm::utils::SimCost;
using cinm::utils::operator+;
using cinm::utils::operator*;
using cinm::utils::operator/;

/// Abstract cost estimator for UPMEM programs in UPMEM dialect.
/// Returns an estimated cost (lower is better) given a region.
/// Implementations may range from simple op counts to full simulation.
struct UpmemSimulator {
  virtual ~UpmemSimulator() = default;
  virtual cinm::utils::Maybe<SimCost> simulate(mlir::Region &region) = 0;
  virtual std::unique_ptr<UpmemSimulator> clone() = 0;
  /// Whether this simulator is safe to call concurrently from multiple threads.
  /// If false, exhaustive search will use a single thread.
  virtual bool supportsMultithreading() const { return true; }
  /// Called on the thread that will later call simulate(), before the first
  /// simulate() call.  Implementations may use this to initialise per-thread
  /// resources on the correct thread.
  virtual void warmUp() {}
  virtual void printStats() const {}

  virtual SimCost simulateGemv(std::chrono::milliseconds timeout,
                               int nTasklets, int64_t mramRows,
                               int64_t mramCols, int64_t rowTile,
                               int64_t colTile, DType dty) = 0;

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
  SimCost simulateFullGemv(std::chrono::milliseconds timeoutMs, int64_t M,
                          int64_t K, int64_t mramRows, int64_t mramCols,
                          int64_t wramRows, int64_t wramCols, int64_t dpuRows,
                          int64_t dpuCols, int64_t tasklets, DType dty);

  virtual SimCost simulateReduction(std::chrono::milliseconds timeout,
                                    cinm::ReduceMethod reduction,
                                    int taskletRows, int taskletCols,
                                    int64_t mramRows, int64_t mramCols,
                                    int64_t wramRows, int64_t wramCols,
                                    DType dty) = 0;

  /// Simulate a reduction operation.
  /// The reduction is like reducing a tensor <MxK> into a tensor <M>.
  /// The M rows are tiled into dpus, tasklets, mram and wram.
  /// The K cols are also tiled into dpus, tasklets, mram and wram and
  /// influence a partial reductions.

  SimCost simulateTailReduction(std::chrono::milliseconds timeoutMs, int64_t M,
                               int64_t K, cinm::ReduceMethod reduction,
                               int64_t mramRows, int64_t mramCols,
                               int64_t wramRows, int64_t wramCols,
                               int64_t dpuRows, int64_t dpuCols,
                               int64_t taskletRows, int64_t taskletCols,
                               DType dty);
};

/// Simple baseline: weighted op count over the UPMEM dialect IR.
/// Assigns higher weight to data-transfer ops and kernel launches.
/// When annotateOpCosts is true, each visited op is tagged with a
/// 'upmem.sim_cost' FloatAttr containing its individual simulated cost.
std::unique_ptr<UpmemSimulator>
createOpCountSimulator(bool annotateOpCosts = false);

/// Which lowering the inference plugin evaluates a configuration through.
enum class UpmemLoweringPath {
  /// The hand-written per-op generators in SimulationTemplates.cpp. These bake
  /// every decision into the generator, and are the quality bar the generic
  /// path has to reach.
  TEMPLATES,
  /// The real pass pipeline: --cinm-tiling, --convert-cinm-to-cnm,
  /// --upmem-tile-mram-buffers, --convert-cnm-to-upmem. What a committed
  /// solution is actually compiled with, so its cost cannot drift from what
  /// the search measured.
  GENERIC,
};

enum class UpmemSimulatorId {
  CYCLE_ACCURATE = 0,
  FAST = 1,
  HYBRID = 2,
  OPCOUNT = 3
};

inline raw_ostream &operator<<(raw_ostream &os, UpmemSimulatorId simid) {
  switch (simid) {
  case UpmemSimulatorId::CYCLE_ACCURATE:
    os << "cycle-accurate";
    return os;
  case UpmemSimulatorId::FAST:
    os << "fast";
    return os;
  case UpmemSimulatorId::HYBRID:
    os << "hybrid";
    return os;
  case UpmemSimulatorId::OPCOUNT:
    os << "opcount";
    return os;
  }
}

/// Creates a UpmemSimulator that translates the lowered UPMEM DPU program to
/// the Python upmem_simulator high-level IR and runs cycle-accurate simulation.
/// Requires the upmem_simulator Python package to be importable.
/// Falls back to the op-count simulator on any Python error.
/// When annotateOpCosts is true, annotates host-side ops with 'upmem.sim_cost'
/// and annotates each WaitForOp with the Python-estimated cycle count.
std::unique_ptr<UpmemSimulator> createCycleAccurateSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0));

std::unique_ptr<UpmemSimulator> createFastSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0));

std::unique_ptr<UpmemSimulator> createHybridSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0));

inline std::unique_ptr<UpmemSimulator> createSimulator(
    UpmemSimulatorId simulator, bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0)) {
  if (simulator == UpmemSimulatorId::CYCLE_ACCURATE)
    return createCycleAccurateSimulator(annotateOpCosts, timeoutMs);
  if (simulator == UpmemSimulatorId::HYBRID)
    return createHybridSimulator(annotateOpCosts, timeoutMs);
  if (simulator == UpmemSimulatorId::FAST)
    return createFastSimulator(annotateOpCosts, timeoutMs);
  else if (simulator == UpmemSimulatorId::OPCOUNT)
    return createOpCountSimulator(annotateOpCosts);

  return nullptr;
}

static constexpr llvm::StringLiteral kSimCostAttr = "upmem.sim_cost";

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
///   aBufSym / yBufSym - symbol names of the MRAM buffers inside the DPU
///   program
void generateTailReduction(cinm::ReduceOp op, RewriterBase &rewriter,
                           int64_t dpuRows, int64_t dpuCols, int64_t mramRows,
                           int64_t mramCols, int64_t wramRows, int64_t wramCols,
                           int64_t taskletRows, int64_t taskletCols);

/// Emit the host-side tiled loop nest for a memref GEMV (out += lhs * rhs).
/// Parameters mirror simulateFullGemv(): dpuRows/dpuCols partition the DPU
/// grid over M and K; mramRows/mramCols are the per-DPU MRAM tile;
/// wramRows/wramCols are the per-tasklet WRAM tile. taskletRows tasklet
/// groups each own a distinct wramRows-row slice; within a group, taskletCols
/// siblings split the column-wise (K) reduction and merge their partial sums
/// at the end of each mr-tile.
void generateGemv(cinm::GemvOp op, RewriterBase &rewriter, int64_t dpuRows,
                  int64_t dpuCols, int64_t mramRows, int64_t mramCols,
                  int64_t wramRows, int64_t wramCols, int64_t taskletRows,
                  int64_t taskletCols);
} // namespace mlir::upmem
