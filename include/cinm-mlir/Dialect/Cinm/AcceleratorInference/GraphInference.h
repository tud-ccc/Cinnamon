#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Graph-level accelerator inference
// ===----------------------------------------------------------------------===//
//
// AcceleratorInference.h searches one compute block. This is the level above:
// finding the blocks a target owns, grouping the ones that have to be decided
// together, and driving a search for each group. The grouping is what
// docs/GraphOptimizationDesign.md calls the graph -- the unit over which DPU
// sets are partitioned (C7), programs merged (C8) and weights packed (C9).

/// An equivalence class of compute blocks with identical programs: same body
/// structure, same operand/result types, same per-operand staticness — the
/// canonicalization the design's C8 (program identity) requires. Constant
/// *payloads* are not part of the signature: weights are scattered data, not
/// program text, so three same-shape projections with different matrices are
/// one class. Everything downstream operates on classes, not blocks: the
/// class is searched once (its members are interchangeable) and the winning
/// configuration is stamped onto every member.
struct BlockClass {
  /// The members, in walk order; the first is the representative that gets
  /// searched. Since a block only ever consumes values defined before it,
  /// walk order is a topological order of the dataflow between blocks.
  SmallVector<ComputeBlockOp> members;

  ComputeBlockOp representative() const { return members.front(); }
  unsigned size() const { return members.size(); }
};

/// One group of compute blocks that is optimized as a whole: a connected
/// component of the dataflow between blocks, restricted to the blocks that
/// target one platform, canonicalized into program-identity classes.
struct ComputeGraph {
  /// The platform every block in the group targets. Blocks of one component
  /// that carry different platform attributes are different groups: they are
  /// different pieces of hardware, so nothing has to be decided jointly.
  CinmPlatformAttrInterface platform;
  /// The classes, ordered by first appearance of a member.
  SmallVector<BlockClass> classes;

  unsigned numBlocks() const {
    unsigned n = 0;
    for (const BlockClass &c : classes)
      n += c.size();
    return n;
  }
};

/// The platform named `platformName` in the `cinm.available_platforms` list of
/// `op` or of one of its ancestors, innermost first. Falsy if no enclosing
/// scope offers that platform, which is how "this op is not ours" is spelled.
CinmPlatformAttrInterface findAvailablePlatform(Operation *op,
                                                StringRef platformName);

/// Group every compute block under `root` that targets `platformName` into
/// the graphs to optimize.
///
/// Connectivity is dataflow between blocks: two blocks are in one component
/// when a value flows from one to the other, or when they read from a common
/// producer (the two gemms of a parallel 2MM share their input and are
/// therefore one graph). Control flow is not interpreted; the relation is a
/// deliberate over-approximation, since the grid capacity C7 is shared by
/// everything pinned on the device and a missing edge would let two graphs
/// hand out the same DPUs.
SmallVector<ComputeGraph> collectComputeGraphs(Operation *root,
                                               StringRef platformName);

/// Builds the plugin that searches blocks running on `platform`. Called once
/// per block, so the plugin may keep per-block state. Returning null aborts
/// inference with a diagnostic.
using InferencePluginFactory =
    llvm::function_ref<std::unique_ptr<InferencePlugin>(
        CinmPlatformAttrInterface platform)>;

/// Run accelerator inference over every compute block under `root` that
/// targets `platformName`, committing the winning configuration of each.
///
/// `opts` is taken by value because it is specialized per block -- the dump
/// directory in particular, which gets one subdirectory per searched block.
DiagnosedSilenceableFailure
inferAcceleratorConfigs(Operation *root, StringRef platformName,
                        InferencePluginFactory makePlugin,
                        InferenceOptions opts);

} // namespace mlir::cinm
