#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Stage B: graph-level resource allocation
// ===----------------------------------------------------------------------===//
//
// The exact outer solve of docs/GraphOptimizationDesign.md, over Stage-A
// profiles only -- no IR, no target, no search. Under the throughput
// objective the problem factors: C8 confines every device set to one
// program-identity class, identical members make groups interchangeable, and
// the min-max objective admits a parametric solve (binary search over the
// finitely many achievable bottleneck values, with a per-class minimum-budget
// DP as the feasibility oracle). Exact, polynomial, no CSP.

/// Stage-B view of one program-identity class: how many members it has and
/// its measured profile L(D). Points must be in strictly increasing resource
/// order (profileComputeBlock returns them that way).
struct ClassProfile {
  unsigned multiplicity = 1;
  SmallVector<ProfilePoint> points;
};

struct AllocationOptions {
  /// D_max: what the whole graph may pin, in profile-resource units (C7).
  int64_t resourceBudget = 0;
  /// C_MRAM: per-device-unit capacity bound for co-residency packing (C9).
  /// 0 disables the capacity check (targets without a residency model).
  int64_t capacityBytes = 0;
  /// Cost of switching a device set to a different program, per inference
  /// per op (the design's measured-risk constant; 40 ms until measured).
  double programReloadMs = 40.0;
  /// Whether ops may be left unpinned (timeshared): zero reserved budget,
  /// but every inference pays programReloadMs + the point's weightScatterMs
  /// on top of its kernel cost. Note the limitation: the eviction cost a
  /// timeshared op inflicts on pinned co-tenants of the DPUs it borrows is
  /// not modeled; the design's headline comparison (all-pinned vs
  /// all-timeshared, section 3) is priced faithfully.
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
