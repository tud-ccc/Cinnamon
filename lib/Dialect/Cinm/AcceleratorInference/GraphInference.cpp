#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
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

/// The program-identity signature of a compute block: a structural
/// fingerprint of the body with values replaced by local numbering, plus
/// operand/result types and the per-operand staticness pattern. Two blocks
/// with equal signatures lower to the same device program under the same
/// configuration, which is what lets them share a device set.
///
/// Constant payloads are deliberately not part of it: constants are
/// materialized as data and moved to the device like any operand, so two
/// bodies differing only in a weight tensor's values (or a scalar factor's)
/// are the same program over different data. The debug tag is excluded as
/// well — it names ops for humans and differs between structurally identical
/// blocks.
void appendOpSignature(Operation *op, DenseMap<Value, unsigned> &valueId,
                       llvm::raw_ostream &os) {
  os << op->getName() << "(";
  for (Value operand : op->getOperands())
    os << valueId.lookup(operand) << ",";
  os << ")";
  if (!op->hasTrait<OpTrait::ConstantLike>()) {
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName() != CinmDialect::DEBUG_TAG_NAME)
        os << attr.getName().getValue() << "=" << attr.getValue() << ";";
  }
  for (Value result : op->getResults()) {
    valueId[result] = valueId.size();
    os << result.getType() << ";";
  }
  for (Region &region : op->getRegions())
    for (Block &block : region) {
      os << "^(";
      for (BlockArgument arg : block.getArguments()) {
        valueId[arg] = valueId.size();
        os << arg.getType() << ",";
      }
      os << "):";
      for (Operation &inner : block)
        appendOpSignature(&inner, valueId, os);
    }
}

std::string blockSignature(ComputeBlockOp block) {
  std::string sig;
  llvm::raw_string_ostream os(sig);
  DenseMap<Value, unsigned> valueId;
  for (Value operand : block->getOperands())
    os << operand.getType() << (isStaticValue(operand) ? "S" : "D") << ";";
  for (Type resultType : block->getResultTypes())
    os << resultType << ";";
  for (BlockArgument arg : block.getBodyArguments())
    valueId[arg] = valueId.size();
  for (Operation &op : block.getBody().front())
    appendOpSignature(&op, valueId, os);
  return sig;
}

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

  // One graph per (component, platform) pair, and one class per signature
  // within a graph, each in the order first met so that the result does not
  // depend on pointer values.
  llvm::MapVector<std::pair<GraphKey, Attribute>, unsigned> graphOf;
  SmallVector<ComputeGraph> graphs;
  SmallVector<llvm::StringMap<unsigned>> classOf;
  for (auto [block, platform] : llvm::zip_equal(blocks, platforms)) {
    std::pair<GraphKey, Attribute> key{
        components.getOrInsertLeaderValue(keyOf(block.getOperation())),
        platform};
    auto [entry, inserted] = graphOf.try_emplace(key, graphs.size());
    if (inserted) {
      graphs.push_back(ComputeGraph{platform, {}});
      classOf.emplace_back();
    }
    ComputeGraph &graph = graphs[entry->second];
    auto [classEntry, classInserted] = classOf[entry->second].try_emplace(
        blockSignature(block), static_cast<unsigned>(graph.classes.size()));
    if (classInserted)
      graph.classes.push_back(BlockClass{});
    graph.classes[classEntry->second].members.push_back(block);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[cinm-inference] " << graphs.size() << " graph(s) on '"
                 << platformName << "'";
    for (const ComputeGraph &graph : graphs)
      llvm::dbgs() << " (" << graph.numBlocks() << " blocks in "
                   << graph.classes.size() << " classes)";
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

/// The two-level solve over one graph: profile each class over the resource
/// menu, allocate the device exactly over the profiles, then stamp each
/// group's winning configuration onto its members and commit them through
/// the single-configuration evaluation path.
static DiagnosedSilenceableFailure
runGraphAllocation(const ComputeGraph &graph, StringRef platformName,
                   InferencePluginFactory makePlugin,
                   const InferenceOptions &opts, StringRef baseDumpDir,
                   StringRef graphName) {
  Location loc = graph.classes.front().representative().getLoc();

  // Profiling: one cost profile per class, on its representative.
  SmallVector<ClassProfile> profiles;
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
    InferenceOptions profileOpts = opts;
    if (!baseDumpDir.empty())
      profileOpts.dumpDir = (std::filesystem::path(baseDumpDir.str()) /
                             graphName.str() / ("class_" + std::to_string(ci)))
                                .string();
    SmallVector<ProfilePoint> points = TRY_GET(
        profileComputeBlock(blockClass.representative(), *plugin, profileOpts));
    profiles.push_back({blockClass.size(), std::move(points)});
  }

  // Allocation: exact solve over the profiles. The budget is the whole
  // device, since each connected component is interpreted as owning the
  // grid; the per-class menus never exceed it.
  std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
  AllocationOptions allocOpts;
  allocOpts.resourceBudget = plugin->sharedResourceMax();
  allocOpts.capacityBytes = plugin->sharedCapacityBytes();
  allocOpts.programReloadMs = opts.programReloadMs;
  std::optional<AllocationResult> alloc = allocateGraph(profiles, allocOpts);
  if (!alloc)
    return emitSilenceableFailure(loc)
           << "no feasible device allocation for this graph: some class fits "
              "no menu configuration";
  LLVM_DEBUG({
    llvm::dbgs() << "[cinm-inference] Graph '" << graphName << "': bottleneck "
                 << alloc->bottleneckMs << " ms, " << alloc->resourceUsed
                 << " / " << allocOpts.resourceBudget << " units pinned\n";
    for (auto [ci, ca] : llvm::enumerate(alloc->perClass))
      for (const GroupAllocation &g : ca.groups)
        llvm::dbgs() << "  class " << ci << ": " << g.size << " member(s) on "
                     << (g.resource ? std::to_string(g.resource)
                                    : std::string("timeshare"))
                     << ", load " << g.loadMs << " ms\n";
  });

  // Finalization: stamp each group's argmin onto its members and commit. The
  // argmin is feasible under the packing by construction -- the allocator
  // admitted the group only if k co-resident copies of this configuration's
  // static footprint fit next to one working region -- so no budgeted
  // re-search is needed for the chosen points. A timeshared group has no
  // reserved set; committing the best point's configuration prices its
  // transient borrow of the device.
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    const ClassAllocation &classAlloc = alloc->perClass[ci];
    const ClassProfile &profile = profiles[ci];
    unsigned member = 0;
    for (const GroupAllocation &group : classAlloc.groups) {
      const ProfilePoint *point = nullptr;
      for (const ProfilePoint &p : profile.points)
        if (group.resource ? p.resource == group.resource
                           : (!point || p.costMs < point->costMs))
          point = &p;
      assert(point && "allocator chose a resource the profile does not have");

      for (unsigned i = 0; i < group.size; ++i, ++member) {
        ComputeBlockOp block = graph.classes[ci].members[member];
        std::unique_ptr<InferencePlugin> memberPlugin =
            makePlugin(graph.platform);
        InferenceOptions memberOpts = opts;
        memberOpts.dumpDir.clear();
        memberOpts.evalSingleSolution = point->config;
        TRY(inferAcceleratorConfig(block, *memberPlugin, memberOpts));
      }
    }
  }
  (void)platformName;
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
inferAcceleratorConfigs(Operation *root, StringRef platformName,
                        InferencePluginFactory makePlugin,
                        InferenceOptions opts) {
  SmallVector<ComputeGraph> graphs = collectComputeGraphs(root, platformName);

  const std::string baseDumpDir = std::move(opts.dumpDir);
  utils::NameInventor namer(root->getContext(), "infer_");

  for (const ComputeGraph &graph : graphs) {
    ComputeBlockOp first = graph.classes.front().representative();
    std::unique_ptr<InferencePlugin> probe = makePlugin(graph.platform);
    if (!probe)
      return emitSilenceableFailure(first.getLoc())
             << "no inference plugin for platform '" << platformName << "'";

    auto scope = first->getParentOfType<SymbolOpInterface>();
    StringRef nameHint = scope && scope.getNameAttr() ? scope.getName() : "op";

    // The two-level solve, when asked for and supported. A user-supplied
    // single solution is a per-block override and bypasses it.
    if (opts.graphAllocation && !opts.evalSingleSolution &&
        !probe->sharedResourceParam().empty() &&
        probe->sharedResourceMax() > 0) {
      StringAttr graphName = namer.getUniqueName(nameHint);
      LLVM_DEBUG(llvm::dbgs()
                 << "===== START GRAPH ALLOCATION " << graphName << " =====\n");
      TRY(runGraphAllocation(graph, platformName, makePlugin, opts, baseDumpDir,
                             graphName));
      continue;
    }

    // Otherwise every block is searched on its own, with the whole device to
    // itself -- the pre-graph behavior.
    for (const BlockClass &blockClass : graph.classes)
      for (ComputeBlockOp block : blockClass.members) {
        std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
        if (!plugin)
          return emitSilenceableFailure(block.getLoc())
                 << "no inference plugin for platform '" << platformName << "'";

        auto blockScope = block->getParentOfType<SymbolOpInterface>();
        StringRef blockHint = blockScope && blockScope.getNameAttr()
                                  ? blockScope.getName()
                                  : "op";
        StringAttr name = namer.getUniqueName(blockHint);
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
