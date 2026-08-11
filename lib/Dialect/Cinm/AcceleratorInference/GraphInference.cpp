#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <filesystem>
#include <string>
#include <utility>

#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Graph collection
// ===----------------------------------------------------------------------===//

CinmPlatformAttrInterface findAvailablePlatform(Operation *op,
                                                StringRef platformName) {
  for (Operation *scope = op; scope; scope = scope->getParentOp()) {
    auto available =
        scope->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
    if (!available)
      continue;
    for (Attribute attr : available)
      if (auto platform = llvm::dyn_cast<CinmPlatformAttrInterface>(attr))
        if (platform.getName() == platformName)
          return platform;
  }
  return {};
}

namespace {

/// Key of the union-find below. Values and operations live in one pointer
/// space here: an operation's results are allocated ahead of the operation
/// itself and block arguments are allocated separately, so no value ever has
/// the address of an operation.
using GraphKey = const void *;

GraphKey keyOf(Value value) { return value.getAsOpaquePointer(); }
GraphKey keyOf(Operation *op) { return op; }

} // namespace

SmallVector<ComputeGraph> collectComputeGraphs(Operation *root,
                                               StringRef platformName) {
  llvm::EquivalenceClasses<GraphKey> components;
  SmallVector<ComputeBlockOp> blocks;
  SmallVector<CinmPlatformAttrInterface> platforms;

  // An operation is a hyperedge: everything it touches lands in one class, and
  // two compute blocks are connected when their classes meet. `regionArgs`
  // pulls in the arguments of the operation's own regions, which is what keeps
  // a value carried through a loop in the component -- control flow is not
  // interpreted, the arguments are simply assumed to be related to the
  // operands. Compute blocks are the exception: they are nodes of the graph,
  // not conduits, and their bodies are opaque.
  auto unite = [&components](Operation *op, bool regionArgs) {
    GraphKey node = keyOf(op);
    components.insert(node);
    for (Value operand : op->getOperands())
      components.unionSets(node, keyOf(operand));
    for (Value result : op->getResults())
      components.unionSets(node, keyOf(result));
    if (!regionArgs)
      return;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Value arg : block.getArguments())
          components.unionSets(node, keyOf(arg));
  };

  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto block = llvm::dyn_cast<ComputeBlockOp>(op)) {
      if (auto platform = findAvailablePlatform(op, platformName)) {
        blocks.push_back(block);
        platforms.push_back(platform);
      }
      // A block that another platform owns is still a conduit between the ones
      // we do own, so it is united either way -- only its body is skipped.
      unite(op, /*regionArgs=*/false);
      return WalkResult::skip();
    }
    unite(op, /*regionArgs=*/true);
    return WalkResult::advance();
  });

  // One graph per (component, platform) pair, in the order the components are
  // first met so that the result does not depend on pointer values.
  llvm::MapVector<std::pair<GraphKey, Attribute>, unsigned> graphOf;
  SmallVector<ComputeGraph> graphs;
  for (auto [block, platform] : llvm::zip_equal(blocks, platforms)) {
    std::pair<GraphKey, Attribute> key{
        components.getOrInsertLeaderValue(keyOf(block.getOperation())),
        platform};
    auto [entry, inserted] = graphOf.try_emplace(key, graphs.size());
    if (inserted)
      graphs.push_back(ComputeGraph{platform, {}});
    graphs[entry->second].blocks.push_back(block);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[cinm-inference] " << graphs.size() << " graph(s) on '"
                 << platformName << "'";
    for (const ComputeGraph &graph : graphs)
      llvm::dbgs() << " (" << graph.blocks.size() << " blocks)";
    llvm::dbgs() << "\n";
  });
  return graphs;
}

// ===----------------------------------------------------------------------===//
// Driver
// ===----------------------------------------------------------------------===//

/// Where the search of the block named `name` dumps its data.
static std::string dumpDirFor(StringRef baseDir, StringRef name,
                              const InferenceOptions &opts) {
  auto path = std::filesystem::path(baseDir.str()) / name.str();
  // Multi-seed mode appends its own seed_<value>/ per seed, so pass the base
  // (per-block) dir. Single-seed BO gets the seed_<rngSeed>/ suffix here.
  // Neither exhaustive search nor random sampling are seeded BO runs, so both
  // dump straight to the base dir.
  if (!opts.exhaustiveSearch && !opts.sampleN && opts.nSeeds <= 1)
    path /= "seed_" + std::to_string(opts.rngSeed);
  return path.string();
}

DiagnosedSilenceableFailure
inferAcceleratorConfigs(Operation *root, StringRef platformName,
                        InferencePluginFactory makePlugin,
                        InferenceOptions opts) {
  SmallVector<ComputeGraph> graphs = collectComputeGraphs(root, platformName);

  const std::string baseDumpDir = std::move(opts.dumpDir);
  utils::NameInventor namer(root->getContext(), "infer_");

  for (const ComputeGraph &graph : graphs) {
    // Every block of the graph is still searched on its own, with the whole
    // device to itself. Graph-level allocation (Stage A/B/C of
    // docs/GraphOptimizationDesign.md) replaces this loop: the per-block
    // search becomes a profiling run over a menu of device sizes, the sizes
    // are then allotted across the graph, and only the winning budget is
    // committed.
    for (ComputeBlockOp block : graph.blocks) {
      std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
      if (!plugin)
        return emitSilenceableFailure(block.getLoc())
               << "no inference plugin for platform '" << platformName << "'";

      auto scope = block->getParentOfType<SymbolOpInterface>();
      StringRef nameHint =
          scope && scope.getNameAttr() ? scope.getName() : "op";
      StringAttr name = namer.getUniqueName(nameHint);
      LLVM_DEBUG(llvm::dbgs()
                 << "===== START INFERENCE " << name << " =====\n");

      InferenceOptions blockOpts = opts;
      if (!baseDumpDir.empty())
        blockOpts.dumpDir = dumpDirFor(baseDumpDir, name, opts);

      TRY(inferAcceleratorConfig(block, *plugin, blockOpts));
    }
  }

  return DiagnosedSilenceableFailure::success();
}

} // namespace mlir::cinm
