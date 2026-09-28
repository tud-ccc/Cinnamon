//===- AllocationReport.h - What a graph allocation did, as JSON ---------===//
//
// The report of one graph's allocation (InferenceOptions::dumpDir,
// InferenceOptions::allocationReportDir): runGraphAllocation records what it
// does in a GraphRecord as it goes, and writeAllocationReport writes it out
// however the run ends. python/experiments cinm_experiments.profiles is its
// reader.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOffloadModel.h"

#include <llvm/Support/JSON.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mlir::cinm {

/// Whether a class entered the allocation, and if not, why it stays on the
/// host. A class that entered it may still be placed on the host by it
/// (InferenceOptions::allowHostPlacement); that is the allocation's record,
/// not this one.
struct ClassFate {
  enum Kind {
    /// It entered the allocation.
    Solved,
    /// Profiling found no device point, and there is no host point to offer
    /// the allocation instead: no reference, no menu, a menu the screen
    /// emptied, or one no search found feasible.
    Unprofiled,
    /// Its best measured point is no faster than the host roofline.
    LosesToHost,
    /// InferenceOptions::hostTransferBoundShare.
    TransferBound,
    /// The allocation places classes, but this one's footprint cannot be
    /// read, so there is no host cost to offer it: a device point would
    /// be chosen against nothing.
    HostUnpriced,
  };
  Kind kind = Solved;
  /// The warning that said so; empty when it was solved.
  std::string reason;
};

/// What one graph's allocation did, recorded as it happens so that the
/// report (writeAllocationReport) describes a run wherever it stopped: after
/// the screen in a dry run, with every class on the host, or with no
/// feasible allocation.
struct GraphRecord {
  /// What the report says about the IR, read before anything is committed:
  /// the commit lowers the blocks it places, and outside stamp mode
  /// replaces them, so the graph's own ops are not readable at the end.
  /// Per graph node, and per graph class.
  llvm::json::Array nodes;
  std::vector<llvm::json::Object> classes;
  SmallVector<OffloadFootprint> footprints;
  SmallVector<std::optional<HostModel>> hosts;
  /// Per graph class.
  SmallVector<std::optional<ReferenceModule>> references;
  SmallVector<ProfileTrace> traces;
  SmallVector<ClassFate> fates;
  /// False when the run stopped before profiling (InferenceOptions::
  /// gateDryRun): the traces then only carry the screen's verdicts.
  bool profiled = false;
  /// Per class in the solve, and which one that is per graph class (-1 for
  /// none), as the allocation sees them.
  SmallVector<ClassProfile> profiles;
  SmallVector<int> solveIndexOfClass;
  std::optional<AllocationOptions> allocOpts;
  std::optional<AllocationResult> alloc;
  std::optional<AllocationScore> score;
  /// Per graph class and member, the group it landed on.
  SmallVector<SmallVector<unsigned>> groupOfMember;
};

/// Read what the report says about `graph`'s IR into `record`, while the IR
/// is still what the run started from (GraphRecord::nodes). The references
/// must be prepared: the operator is read off them.
void snapshotGraph(const ComputeGraph &graph, GraphRecord &record);

/// Write each class's reference module to
/// `<graphDir>/class_<i>/reference.mlir`, its function renamed
/// `<graphName>_class<i>`, and record where in the class's entry of the
/// report (`reference`: path relative to `graphDir`, and function). It holds
/// the class's one compute block in the form the search space is read off,
/// so a configuration from the report can be compiled from it on its own
/// (InferenceOptions::evalSingleSolution).
void writeReferenceModules(const std::filesystem::path &graphDir,
                           StringRef graphName, GraphRecord &record);

/// Write everything one graph's allocation did to `path` as JSON: the machine
/// it priced against, the options it ran under, the graph, and per class the
/// operator, every point of its menu with how it was priced and what became
/// of it, and the groups the allocation gave it.
void writeAllocationReport(const std::filesystem::path &path,
                           const ComputeGraph &graph, StringRef graphName,
                           StringRef platformName, InferencePlugin &plugin,
                           const InferenceOptions &opts,
                           const GraphRecord &record);

} // namespace mlir::cinm
