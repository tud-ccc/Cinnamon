#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Graph-level resource allocation
// ===----------------------------------------------------------------------===//
//
// Divide a device among the compute blocks of one graph: each block is
// assigned to a device set that holds its program and pinned weights for the
// serving lifetime, and the steady-state throughput bottleneck -- the
// busiest set's per-inference work -- is minimized. Works on measured cost
// profiles only: no IR, no target, no search.
//
// The solve is exact and polynomial. Only blocks with identical programs may
// share a set, so the assignment decomposes per program-identity class, and
// identical members make groups within a class interchangeable. That leaves
// a min-max problem over finitely many achievable per-set loads: binary
// search the smallest feasible bottleneck, using a per-class minimum-budget
// dynamic program as the feasibility oracle.

/// Allocation view of one program-identity class: how many members it has
/// and its measured cost profile (best cost per device size). Points must be
/// in strictly increasing resource order (profileComputeBlock returns them
/// that way).
struct ClassProfile {
  unsigned multiplicity = 1;
  SmallVector<ProfilePoint> points;
};

struct AllocationOptions {
  /// What the whole graph may pin, in profile-resource units: the total
  /// device size. Sets are carved out of this and never returned, so the sum
  /// over all pinned sets must stay within it.
  int64_t resourceBudget = 0;
  /// Per-device-unit memory capacity (MRAM bytes per DPU). Bounds how many
  /// members may be co-resident on one set: their pinned weights all stay in
  /// memory at once, while their working buffers run sequentially and reuse
  /// one region. 0 disables the check (targets without a residency model).
  int64_t capacityBytes = 0;
  /// Cost of switching a device set to a different program, per inference
  /// per op. Assumed constant (40 ms) until measured on hardware.
  double programReloadMs = 40.0;
  /// Whether ops may be left unpinned (timeshared): zero reserved budget,
  /// but every inference pays programReloadMs + the point's weightScatterMs
  /// on top of its kernel cost. Note the limitation: the eviction cost a
  /// timeshared op inflicts on pinned co-tenants of the DPUs it borrows is
  /// not modeled; the comparison that matters -- everything pinned versus
  /// everything timeshared -- is priced faithfully.
  bool allowTimeshare = true;
};

/// One device set: `size` members of one class co-resident on `resource`
/// units. `resource == 0` means unpinned (timeshared). `loadMs` is the set's
/// per-inference work, the term the objective takes the max over.
struct GroupAllocation {
  unsigned size = 0;
  int64_t resource = 0;
  double loadMs = 0;
};

/// The chosen grouping of one class's members (sums to its multiplicity).
struct ClassAllocation {
  SmallVector<GroupAllocation> groups;
};

/// The outer solve's output: a budget per class (parallel to the input),
/// and the steady-state bottleneck it achieves.
struct AllocationResult {
  SmallVector<ClassAllocation> perClass;
  /// max over all groups of loadMs: the steady-state per-inference time.
  double bottleneckMs = 0;
  /// Sum of pinned groups' resources: how much of the budget is used.
  int64_t resourceUsed = 0;
};

/// Solve the allocation exactly. Returns std::nullopt when no allocation is
/// feasible -- some class has members that neither fit any pinned
/// configuration nor may timeshare.
std::optional<AllocationResult> allocateGraph(ArrayRef<ClassProfile> classes,
                                              const AllocationOptions &opts);

} // namespace mlir::cinm
