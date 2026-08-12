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
// serving lifetime. Works on measured cost profiles only: no IR, no target,
// no search. Two objectives, sharing everything but the evaluator.
//
// *Throughput* minimizes the steady-state bottleneck -- the busiest set's
// per-inference work -- and is solved exactly and polynomially. Only blocks
// with identical programs may share a set, so the assignment decomposes per
// program-identity class, and identical members make groups within a class
// interchangeable. That leaves a min-max problem over finitely many
// achievable per-set loads: binary search the smallest feasible bottleneck,
// using a per-class minimum-budget dynamic program as the feasibility
// oracle. Dependencies do not enter it: whatever the order, a set's
// per-inference work is the sum over the ops it holds.
//
// *Latency* minimizes the makespan of one inference, which is a longest path
// and so does depend on the dependencies. It is solved by a critical-path
// greedy, with no claim to optimality; see allocateGraphForLatency.

/// Allocation view of one program-identity class: how many members it has
/// and its measured cost profile (best cost per device size). Points must be
/// in strictly increasing resource order (profileComputeBlock returns them
/// that way).
struct ClassProfile {
  unsigned multiplicity = 1;
  SmallVector<ProfilePoint> points;
};

/// Declared capacity of one memory level of the device, in the same
/// per-instance unit the profiles' LevelResidency entries use. Typically
/// copied straight off the platform's level declarations.
struct LevelCapacity {
  std::string level;
  int64_t bytes = 0;
};

struct AllocationOptions {
  /// What the whole graph may pin, in profile-resource units: the total
  /// device size. Sets are carved out of this and never returned, so the sum
  /// over all pinned sets must stay within it.
  int64_t resourceBudget = 0;
  /// Capacity per memory level. Bounds how many members may be co-resident
  /// on one set: per level, their pinned footprints all stay resident at
  /// once and sum, while their working buffers run sequentially and reuse
  /// one region (max). Levels in which a configuration pins nothing never
  /// bind, so listing every level of the platform is the right default.
  /// Empty disables the check (targets without a residency model).
  SmallVector<LevelCapacity> capacities;
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
/// and the objective value it achieves.
struct AllocationResult {
  SmallVector<ClassAllocation> perClass;
  /// The achieved objective, in ms: the steady-state bottleneck (the busiest
  /// set's per-inference work) for the throughput solve, the makespan of one
  /// inference for the latency solve.
  double objectiveMs = 0;
  /// Sum of pinned groups' resources: how much of the budget is used.
  int64_t resourceUsed = 0;
  /// Latency solve only: which group of its class each node was assigned to,
  /// indexed by node. Empty after a throughput solve, whose class members are
  /// interchangeable and may be handed to the groups in any order.
  SmallVector<unsigned> groupOfNode;
};

/// One node of the dependency graph the latency objective walks. Nodes must
/// be given in a topological order -- every predecessor index smaller than
/// the node's own -- which is what a walk of the IR produces naturally.
struct GraphNode {
  /// Index into the `classes` array the node's cost profile comes from.
  unsigned classIndex = 0;
  /// Position among that class's members. Co-resident members execute in
  /// this order.
  unsigned memberIndex = 0;
  /// Nodes whose results this one consumes.
  SmallVector<unsigned> predecessors;
};

/// Solve the throughput allocation exactly. Returns std::nullopt when no
/// allocation is feasible -- some class has members that neither fit any
/// pinned configuration nor may timeshare.
std::optional<AllocationResult> allocateGraph(ArrayRef<ClassProfile> classes,
                                              const AllocationOptions &opts);

/// Allocate for single-inference latency: the makespan of `nodes`, where a
/// node costs its group's profiled time, co-resident nodes serialize in
/// member order, and nodes on different sets overlap wherever the
/// dependencies allow.
///
/// This is the critical-path greedy of the mixed task/data-parallel
/// scheduling literature (CPA/CPR), and makes no claim to optimality. It
/// starts from maximal merging -- one set per class at the smallest size
/// that holds all its members, the cheapest feasible allocation -- and then
/// repeatedly applies whichever move buys the largest makespan reduction per
/// additional device unit: *grow* a set to the next size on its menu, or
/// *split* a member out into a set of its own. Every accepted move spends
/// budget, so the loop terminates. Timesharing is not offered: it can only
/// lengthen the path.
///
/// Note what the starting point already achieves: members of one class that
/// are sequentially dependent never run concurrently, so merging them costs
/// no latency at all while dividing their budget by their number. On a
/// layered model that collapses every layer's copy of a kernel onto one set
/// for free, and the greedy then spends the whole device widening those sets.
///
/// Returns std::nullopt when even maximal merging at the smallest sizes
/// exceeds the budget or violates a capacity.
std::optional<AllocationResult>
allocateGraphForLatency(ArrayRef<ClassProfile> classes,
                        ArrayRef<GraphNode> nodes,
                        const AllocationOptions &opts);

} // namespace mlir::cinm
