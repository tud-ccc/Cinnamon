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
};

/// Simple baseline: weighted op count over the UPMEM dialect IR.
/// Assigns higher weight to data-transfer ops and kernel launches.
/// When annotateOpCosts is true, each visited op is tagged with a
/// 'upmem.sim_cost' FloatAttr containing its individual simulated cost.
std::unique_ptr<UpmemSimulator>
createOpCountSimulator(bool annotateOpCosts = false);

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
/// `programDumpDir`, when non-empty, additionally writes each simulated DPU
/// program to `<dir>/<kernel>.cnmprog.json` — the symbolic interchange format
/// the Python reference cost model reads (see
/// upmem-cost-model's ProgramBuilder::emitJson and the Python side's
/// Predictor/cnmprog.py), for cross-checking this simulator against it.
///
/// Only ever set this on a one-shot path such as --upmem-annotate-costs. The
/// search pipeline simulates once per candidate configuration, so a dump there
/// would write a file per evaluation.
std::unique_ptr<UpmemSimulator> createCycleAccurateSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0),
    llvm::StringRef programDumpDir = {});

std::unique_ptr<UpmemSimulator> createFastSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0),
    llvm::StringRef programDumpDir = {});

std::unique_ptr<UpmemSimulator> createHybridSimulator(
    bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0),
    llvm::StringRef programDumpDir = {});

inline std::unique_ptr<UpmemSimulator> createSimulator(
    UpmemSimulatorId simulator, bool annotateOpCosts = false,
    std::chrono::milliseconds timeoutMs = std::chrono::milliseconds(0),
    llvm::StringRef programDumpDir = {}) {
  if (simulator == UpmemSimulatorId::CYCLE_ACCURATE)
    return createCycleAccurateSimulator(annotateOpCosts, timeoutMs,
                                        programDumpDir);
  if (simulator == UpmemSimulatorId::HYBRID)
    return createHybridSimulator(annotateOpCosts, timeoutMs, programDumpDir);
  if (simulator == UpmemSimulatorId::FAST)
    return createFastSimulator(annotateOpCosts, timeoutMs, programDumpDir);
  else if (simulator == UpmemSimulatorId::OPCOUNT)
    return createOpCountSimulator(annotateOpCosts);

  return nullptr;
}

static constexpr llvm::StringLiteral kSimCostAttr = "upmem.sim_cost";

} // namespace mlir::upmem
