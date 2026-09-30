#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AllocationReport.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"
#include <llvm/Support/FormatVariadic.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Graph collection
// ===----------------------------------------------------------------------===//

CinmPlatformAttrInterface findAvailablePlatform(Operation *op,
                                                StringRef platformName) {
  // The nearest declaration wins: an op that carries the attribute states
  // its complete set of platforms, and the enclosing declarations are only
  // defaults for the ops that carry none. A host-pinned block inside a
  // function that advertises an accelerator platform is *not* a candidate
  // for that platform.
  for (Operation *scope = op; scope; scope = scope->getParentOp()) {
    auto available =
        scope->getAttrOfType<ArrayAttr>(CinmDialect::AVAILABLE_PLATFORMS_NAME);
    if (!available)
      continue;
    for (Attribute attr : available)
      if (auto platform = llvm::dyn_cast<CinmPlatformAttrInterface>(attr))
        if (platform.getName() == platformName)
          return platform;
    return {};
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

/// The nodes of `nodeOfBlock` whose results `block` consumes, directly or
/// through ops the graph does not own (a slice of a producer's result, a
/// reshape, a host-side merge). Tracing back through those intermediates is
/// what makes the edge set reflect the dataflow rather than the syntax.
/// Sorted and deduplicated.
SmallVector<unsigned>
producingNodes(ComputeBlockOp block,
               const DenseMap<Operation *, unsigned> &nodeOfBlock) {
  SmallVector<unsigned> preds;
  SmallVector<Value> worklist(block->getOperands());
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    Operation *def = value.getDefiningOp();
    if (!def) // a block argument: outside the graph, nothing to trace
      continue;
    auto known = nodeOfBlock.find(def);
    if (known != nodeOfBlock.end()) {
      preds.push_back(known->second);
      continue; // a graph node: an edge, not something to see through
    }
    llvm::append_range(worklist, def->getOperands());
  }
  llvm::sort(preds);
  preds.erase(llvm::unique(preds), preds.end());
  return preds;
}

} // namespace

/// How many times `block` runs per inference: the product of the constant
/// trip counts of the loops between it and its function. A loop whose trip
/// count is not a constant counts as one -- the makespan then undercounts
/// it, which is the conservative side for a decision about what to pin.
static int64_t executionsOf(ComputeBlockOp block) {
  int64_t executions = 1;
  for (Operation *op = block->getParentOp();
       op && !isa<FunctionOpInterface>(op); op = op->getParentOp()) {
    if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
      auto constant = [](std::optional<OpFoldResult> bound) {
        return bound ? getConstantIntValue(*bound) : std::nullopt;
      };
      std::optional<int64_t> lb = constant(loop.getSingleLowerBound());
      std::optional<int64_t> ub = constant(loop.getSingleUpperBound());
      std::optional<int64_t> step = constant(loop.getSingleStep());
      if (lb && ub && step && *step > 0 && *ub > *lb)
        executions *= (*ub - *lb + *step - 1) / *step;
    }
  }
  return executions;
}

/// The `sequence` field of a group member's stamp (read by the residency
/// slots of CnmToUPMEM): the loops between the group's block and its
/// function, innermost first, when the program text fixes the order the
/// group's launches come in -- its members all in one basic block, so that a
/// pass over it launches each member once, in stamp order, and every loop
/// around that block with constant bounds, so that a launch's number gives
/// each loop's iteration. `operands` lists the member's operands that carry
/// the loop's induction variable. Null when the order is not fixed.
static DictionaryAttr launchSequenceOf(Builder &builder, ComputeBlockOp member,
                                       ArrayRef<ComputeBlockOp> group) {
  for (ComputeBlockOp other : group)
    if (other->getBlock() != member->getBlock())
      return {};
  SmallVector<Attribute> loops;
  int64_t stride = 1;
  for (Operation *op = member->getParentOp();
       op && !isa<FunctionOpInterface>(op); op = op->getParentOp()) {
    auto loop = dyn_cast<LoopLikeOpInterface>(op);
    if (!loop)
      return {};
    auto constant = [](std::optional<OpFoldResult> bound) {
      return bound ? getConstantIntValue(*bound) : std::nullopt;
    };
    std::optional<Value> iv = loop.getSingleInductionVar();
    std::optional<int64_t> lb = constant(loop.getSingleLowerBound());
    std::optional<int64_t> ub = constant(loop.getSingleUpperBound());
    std::optional<int64_t> step = constant(loop.getSingleStep());
    if (!iv || !lb || !ub || !step || *step <= 0 || *ub <= *lb)
      return {};
    const int64_t trip = (*ub - *lb + *step - 1) / *step;
    SmallVector<int64_t> operands;
    for (auto [k, operand] : llvm::enumerate(member.getOperands()))
      if (operand == *iv)
        operands.push_back(k);
    loops.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("lb", builder.getI64IntegerAttr(*lb)),
        builder.getNamedAttr("step", builder.getI64IntegerAttr(*step)),
        builder.getNamedAttr("trip", builder.getI64IntegerAttr(trip)),
        builder.getNamedAttr("stride", builder.getI64IntegerAttr(stride)),
        builder.getNamedAttr("operands",
                             builder.getDenseI64ArrayAttr(operands)),
    }));
    stride *= trip;
  }
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("loops", builder.getArrayAttr(loops))});
}

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
      graphs.push_back(ComputeGraph{platform, {}, {}});
      classOf.emplace_back();
    }
    ComputeGraph &graph = graphs[entry->second];
    auto [classEntry, classInserted] = classOf[entry->second].try_emplace(
        blockSignature(block), static_cast<unsigned>(graph.classes.size()));
    if (classInserted)
      graph.classes.push_back(BlockClass{});
    BlockClass &blockClass = graph.classes[classEntry->second];
    graph.nodes.push_back(BlockNode{block, classEntry->second,
                                    blockClass.size(), /*predecessors=*/{},
                                    executionsOf(block)});
    blockClass.members.push_back(block);
  }

  // Dependency edges, once every node of a graph is known: which of them
  // produced the values a block consumes. Blocks are added in walk order, so
  // a producer always has the smaller index and the numbering is topological.
  for (ComputeGraph &graph : graphs) {
    DenseMap<Operation *, unsigned> nodeOfBlock;
    for (auto [i, node] : llvm::enumerate(graph.nodes))
      nodeOfBlock[node.block.getOperation()] = i;
    for (BlockNode &node : graph.nodes)
      node.predecessors = producingNodes(node.block, nodeOfBlock);
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
  // dump straight to the base dir -- as does dump-space-only, whose output
  // does not depend on any seed.
  if (!opts.exhaustiveSearch && !opts.sampleN && opts.nSeeds <= 1 &&
      !opts.dumpSpaceOnly)
    path /= "seed_" + std::to_string(opts.rngSeed);
  return path.string();
}

/// One class of an imported allocation (InferenceOptions::allocationIn): the
/// profile points its groups run, host first, and which group each member
/// is on.
struct ImportedClass {
  SmallVector<ProfilePoint> points;
  SmallVector<GroupAllocation> groups;
  SmallVector<unsigned> groupOfMember;
};

/// Read the allocation in `path` for `graph`, named `graphName`; a file that
/// names its graph must name this one. `resourceParam` is the configuration
/// key that names a group's resource; a config that sets it must agree with
/// the group. Returns the reason on failure.
static std::optional<std::string>
readImportedAllocation(StringRef path, const ComputeGraph &graph,
                       StringRef graphName, StringRef resourceParam,
                       std::vector<ImportedClass> &out) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return ("cannot read '" + path + "': " + buffer.getError().message()).str();
  llvm::Expected<llvm::json::Value> root =
      llvm::json::parse((*buffer)->getBuffer());
  if (!root)
    return ("'" + path + "' is not JSON: " + llvm::toString(root.takeError()))
        .str();
  const llvm::json::Array *classes =
      root->getAsObject() ? root->getAsObject()->getArray("classes") : nullptr;
  if (!classes)
    return ("'" + path + "' has no `classes` array").str();
  if (std::optional<StringRef> named = root->getAsObject()->getString("graph");
      named && *named != graphName)
    return ("'" + path + "' is the allocation of '" + *named + "', not '" +
            graphName + "'")
        .str();

  out.assign(graph.classes.size(), {});
  SmallVector<bool> listed(graph.classes.size(), false);
  for (const llvm::json::Value &entry : *classes) {
    const llvm::json::Object *c = entry.getAsObject();
    std::optional<int64_t> ci = c ? c->getInteger("class") : std::nullopt;
    if (!ci || *ci < 0 || *ci >= static_cast<int64_t>(graph.classes.size()))
      return std::string("a class entry has no valid `class` index");
    if (listed[*ci])
      return llvm::formatv("class {0} is listed twice", *ci).str();
    listed[*ci] = true;
    const unsigned size = graph.classes[*ci].size();
    ImportedClass &imported = out[*ci];
    imported.groupOfMember.assign(size, ~0u);
    const llvm::json::Array *groups = c->getArray("groups");
    if (!groups)
      return llvm::formatv("class {0} has no `groups`", *ci).str();
    for (const llvm::json::Value &g : *groups) {
      const llvm::json::Object *group = g.getAsObject();
      const llvm::json::Array *members =
          group ? group->getArray("members") : nullptr;
      if (!members)
        return llvm::formatv("a group of class {0} has no `members`", *ci)
            .str();
      GroupAllocation alloc;
      alloc.onHost = group->getBoolean("on_host").value_or(false);
      alloc.size = members->size();
      const double costMs = group->getNumber("cost_ms").value_or(0.0);
      if (!alloc.onHost) {
        // A timeshared group holds no set: it runs its point on a transient
        // borrow of the device and takes no budget.
        const int64_t resource = group->getInteger("resource").value_or(0);
        if (group->getBoolean("timeshared").value_or(false))
          alloc.pointResource = resource;
        else
          alloc.resource = resource;
        const llvm::json::Object *config = group->getObject("config");
        if (resource <= 0 || !config || config->empty())
          return llvm::formatv("a device group of class {0} needs a positive "
                               "`resource` and a `config`",
                               *ci)
              .str();
        ProfilePoint point;
        point.resource = resource;
        point.costMs = costMs;
        for (const auto &[key, value] : *config) {
          std::optional<int64_t> v = value.getAsInteger();
          if (!v)
            return llvm::formatv("config entry `{0}` of class {1} is not an "
                                 "integer",
                                 key.str(), *ci)
                .str();
          point.config[key.str()] = static_cast<ParmValue>(*v);
        }
        auto it = point.config.find(resourceParam);
        if (it != point.config.end() && it->second != resource)
          return llvm::formatv("class {0}: a group on {1} runs a config with "
                               "{2}={3}",
                               *ci, resource, resourceParam, it->second)
              .str();
        // Groups of a class on the same resource run the same configuration:
        // that is how a group finds its point (pointOf).
        auto same = llvm::find_if(imported.points, [&](const ProfilePoint &p) {
          return !p.onHost && p.resource == resource;
        });
        if (same == imported.points.end())
          imported.points.push_back(std::move(point));
        else if (same->config != point.config)
          return llvm::formatv("class {0} has two groups on {1} with different "
                               "configs",
                               *ci, resource)
              .str();
      } else if (llvm::none_of(imported.points, [](const ProfilePoint &p) {
                   return p.onHost;
                 })) {
        ProfilePoint host;
        host.resource = 0;
        host.costMs = costMs;
        host.onHost = true;
        imported.points.insert(imported.points.begin(), std::move(host));
      }
      const unsigned gi = imported.groups.size();
      for (const llvm::json::Value &m : *members) {
        std::optional<int64_t> mi = m.getAsInteger();
        if (!mi || *mi < 0 || *mi >= size)
          return llvm::formatv("class {0} has {1} members; a group lists "
                               "another",
                               *ci, size)
              .str();
        if (imported.groupOfMember[*mi] != ~0u)
          return llvm::formatv("member {0} of class {1} is in two groups", *mi,
                               *ci)
              .str();
        imported.groupOfMember[*mi] = gi;
      }
      imported.groups.push_back(alloc);
    }
    for (auto [mi, gi] : llvm::enumerate(imported.groupOfMember))
      if (gi == ~0u)
        return llvm::formatv("member {0} of class {1} is in no group", mi, *ci)
            .str();
  }
  for (auto [ci, isListed] : llvm::enumerate(listed))
    if (!isListed)
      return llvm::formatv("class {0} is not listed", ci).str();
  return std::nullopt;
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

  // Every class's reference module, in the form the plugin states its space
  // over: what the menu is read off, and what every search of the class
  // clones its trials from. None for a class the plugin cannot rewrite (the
  // error is emitted); it then has no menu and stays on the host.
  GraphRecord record;
  auto &references = record.references;
  for (const BlockClass &blockClass : graph.classes) {
    std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
    FailureOr<ReferenceModule> reference =
        prepareReferenceModule(blockClass.representative(), *plugin);
    references.push_back(failed(reference)
                             ? std::nullopt
                             : std::optional(std::move(*reference)));
  }
  record.traces.resize(graph.classes.size());
  record.fates.resize(graph.classes.size());
  snapshotGraph(graph, record);
  if (!baseDumpDir.empty())
    writeReferenceModules(std::filesystem::path(baseDumpDir.str()) /
                              graphName.str(),
                          graphName, record);

  // The report describes the run however it ends, so it is set up before
  // anything can end it.
  SmallVector<std::filesystem::path> reportPaths;
  if (!baseDumpDir.empty())
    reportPaths.push_back(std::filesystem::path(baseDumpDir.str()) /
                          graphName.str() / "allocation.json");
  if (!opts.allocationReportDir.empty())
    reportPaths.push_back(std::filesystem::path(opts.allocationReportDir) /
                          (graphName.str() + ".json"));
  auto writeReport = llvm::scope_exit([&] {
    if (reportPaths.empty())
      return;
    std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
    for (const std::filesystem::path &path : reportPaths)
      writeAllocationReport(path, graph, graphName, platformName, *plugin, opts,
                            record);
  });

  if (opts.gateDryRun) {
    // Nothing profiled and nothing offloaded: the program that comes out
    // runs entirely on the host, and the point of the run is the report --
    // the screen's verdicts and the menus a run would have swept.
    for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
      if (!references[ci])
        continue;
      std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
      record.traces[ci] = planProfile(blockClass.representative(),
                                      references[ci]->block, *plugin, opts);
    }
    return DiagnosedSilenceableFailure::success();
  }
  record.profiled = true;

  // Profiling: one cost profile per class, on its representative. A class the
  // platform cannot run at any menu point is not an error at the graph level:
  // its members simply stay on the host (they keep no accelerator annotation,
  // which is what the downstream lowering treats as host execution) and the
  // solve runs over the remaining classes. Only definite failures abort.
  auto &profiles = record.profiles; // one entry per *kept* class
  auto &solveIndexOfClass = record.solveIndexOfClass;
  solveIndexOfClass.assign(graph.classes.size(), -1);
  // Indexed by graph class, unlike `profiles`: a class that found no feasible
  // configuration still ran searches, and what they found is worth keeping.
  auto &traces = record.traces;
  auto &fates = record.fates;

  // The classes are independent searches over their own representatives, so
  // they run concurrently. This is where a whole program's parallelism
  // actually is: one class's sweep only sustains (menu points x batch size)
  // evaluations, which on a many-core machine leaves most of it idle, while
  // a graph offers a dozen of those at once.
  //
  // The budget is NOT divided between the classes, because their costs are
  // nothing like equal -- on llama one class is half the work of the whole
  // graph -- so a share-out starves whichever class is the critical path and
  // gives its threads to classes that finish early anyway. Every class offers
  // all of its points instead, and one shared gate caps how many searches run
  // at once; the permits end up wherever work remains.
  const unsigned baseWorkers =
      opts.numWorkers ? opts.numWorkers
                      : std::max(1u, std::thread::hardware_concurrency());
  const bool threaded = loc.getContext()->isMultithreadingEnabled();
  ProfileGate gate(baseWorkers);
  const unsigned classThreads = threaded ? graph.classes.size() : 1;

  // Collected per class, reduced in class order below: the allocation depends
  // on the order of `profiles`, so the finish order must not reach it.
  struct ClassResult {
    std::optional<SmallVector<ProfilePoint>> points;
    std::optional<DiagnosedSilenceableFailure> definite;
    /// Why a class has no points, in the sweep's own words -- the menu
    /// screen's verdict, say. Kept rather than emitted where it arises: the
    /// sweeps finish in no particular order, and a reader cannot make sense
    /// of diagnostics in that one.
    std::string silenced;
  };
  std::vector<ClassResult> results(graph.classes.size());

  // An imported allocation replaces profiling and the solve: its groups'
  // points are the profiles, and it is committed as it stands.
  std::vector<ImportedClass> imported;
  if (!opts.allocationIn.empty()) {
    std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
    if (std::optional<std::string> error = readImportedAllocation(
            (std::filesystem::path(opts.allocationIn) /
             (graphName.str() + ".json"))
                .string(),
            graph, graphName, plugin->sharedResourceParam(), imported))
      return emitDefiniteFailure(loc) << "allocation-in: " << *error;
    for (auto [ci, cls] : llvm::enumerate(imported))
      results[ci].points = cls.points;
  }
  const bool isImported = !imported.empty();

  auto profileClass = [&](size_t ci) {
    const BlockClass &blockClass = graph.classes[ci];
    if (!references[ci]) {
      results[ci].silenced =
          "no reference module could be prepared for this block";
      return;
    }
    std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
    InferenceOptions profileOpts = opts;
    profileOpts.numWorkers = baseWorkers;
    if (!baseDumpDir.empty())
      profileOpts.dumpDir = (std::filesystem::path(baseDumpDir.str()) /
                             graphName.str() / ("class_" + std::to_string(ci)))
                                .string();
    utils::Maybe<SmallVector<ProfilePoint>> points = profileComputeBlock(
        blockClass.representative(), *plugin, profileOpts, &traces[ci],
        threaded ? &gate : nullptr, &*references[ci]);
    if (auto *fail = std::get_if<DiagnosedSilenceableFailure>(&points)) {
      if (fail->isDefiniteFailure())
        results[ci].definite = std::move(*fail);
      else {
        // Not an error at the graph level, and the warning that says so is
        // emitted below: diagnostics from the sweep would come out in finish
        // order, which is not an order the user can make sense of.
        results[ci].silenced = StringRef(fail->getMessage()).trim().str();
        (void)fail->silence();
      }
      return;
    }
    results[ci].points = std::move(std::get<SmallVector<ProfilePoint>>(points));
  };

  if (isImported) {
    // Nothing to profile.
  } else if (classThreads <= 1) {
    for (size_t ci = 0; ci < graph.classes.size(); ++ci)
      profileClass(ci);
  } else {
    std::atomic<size_t> next{0};
    auto worker = [&] {
      for (size_t ci = next.fetch_add(1, std::memory_order_relaxed);
           ci < graph.classes.size();
           ci = next.fetch_add(1, std::memory_order_relaxed))
        profileClass(ci);
    };
    std::vector<std::thread> threads;
    threads.reserve(classThreads - 1);
    for (unsigned t = 1; t < classThreads; ++t)
      threads.emplace_back(worker);
    worker();
    for (std::thread &t : threads)
      t.join();
  }

  // What the host would take for one execution of a class (cinm::hostSeconds),
  // as a profile point the allocation may choose
  // (InferenceOptions::allowHostPlacement): zero resource, no residency.
  auto hostPointOf =
      [&](cinm::ComputeBlockOp block) -> std::optional<ProfilePoint> {
    cinm::OffloadFootprint f = cinm::measureOffloadFootprint(block);
    auto host = cinm::HostPlatformAttr::getInScope(block);
    if (!f.known || !host)
      return std::nullopt;
    const double ms =
        cinm::hostSeconds(f, host.getModel(), opts.hostAchievedFraction) * 1e3;
    if (!(ms > 0.0))
      return std::nullopt;
    ProfilePoint point;
    point.resource = 0;
    point.costMs = ms;
    point.onHost = true;
    return point;
  };

  // Placement is the allocation's to decide only under the objective that
  // can price it: the throughput solve takes the busiest device set's load,
  // and host work loads no set.
  const bool placementIsSolved =
      opts.allowHostPlacement && opts.latencyObjective;

  // Reduce in class order, so the profile list, the diagnostics and the
  // failure that wins are all the ones a serial sweep would have produced.
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    if (results[ci].definite)
      return std::move(*results[ci].definite);
    if (!results[ci].points) {
      // With host placement the class is not dropped: it enters the solve
      // able only to stay where it is, so its cost is on the critical path
      // like everything else.
      if (opts.allowHostPlacement && opts.latencyObjective) {
        if (std::optional<ProfilePoint> host =
                hostPointOf(blockClass.representative())) {
          results[ci].points = SmallVector<ProfilePoint>{*host};
          fates[ci].reason = results[ci].silenced;
        }
      }
    }
    // A class that does not enter the solve stays on the host, and says why.
    auto staysOnHost = [&](ClassFate::Kind kind, std::string reason) {
      fates[ci] = {kind, reason};
      blockClass.representative().emitWarning()
          << reason << "; it stays on the host, along with the "
          << (blockClass.size() - 1) << " other block(s) of its class";
    };
    if (!results[ci].points) {
      staysOnHost(ClassFate::Unprofiled,
                  results[ci].silenced.empty()
                      ? ("no feasible '" + platformName +
                         "' configuration for this block")
                            .str()
                      : results[ci].silenced);
      continue;
    }
    SmallVector<ProfilePoint> &pts = *results[ci].points;
    const ProfilePoint *bestPt = &*llvm::min_element(
        pts, [](const auto &a, const auto &b) { return a.costMs < b.costMs; });

    // The menu screen again, with the device's side priced by the search
    // instead of bounded by its roofline.
    if (opts.screenMenuAgainstHost && !placementIsSolved && !isImported) {
      cinm::OffloadFootprint f =
          cinm::measureOffloadFootprint(blockClass.representative());
      auto host =
          cinm::HostPlatformAttr::getInScope(blockClass.representative());
      const double hostMs = f.known && host
                                ? cinm::hostSeconds(f, host.getModel(),
                                                    opts.hostAchievedFraction) *
                                      1e3
                                : 0.0;
      if (hostMs > 0.0 && bestPt->costMs >= hostMs) {
        staysOnHost(ClassFate::LosesToHost,
                    llvm::formatv("the search found nothing on '{0}' that "
                                  "beats the host: its best, {1} device(s) "
                                  "at {2:F3} ms, against the host's {3:F3} ms",
                                  platformName, bestPt->resource,
                                  bestPt->costMs, hostMs)
                        .str());
        continue;
      }
    }

    // The transfer-bound heuristic, from before there was a host cost model
    // to compare against (see InferenceOptions::hostTransferBoundShare): a
    // class whose best point is the smallest menu value gains nothing from
    // more devices, and when that point is also mostly transfer the device
    // buys it essentially nothing at all.
    if (opts.hostTransferBoundShare > 0 && !isImported) {
      if (bestPt->resource == pts.front().resource &&
          bestPt->transferShare >= opts.hostTransferBoundShare) {
        staysOnHost(ClassFate::TransferBound,
                    llvm::formatv("transfer-bound on '{0}' ({1}% of its best "
                                  "point's cost is data movement, and more "
                                  "devices do not improve it)",
                                  platformName,
                                  static_cast<int>(bestPt->transferShare * 100))
                        .str());
        continue;
      }
    }
    if (placementIsSolved && !isImported) {
      std::optional<ProfilePoint> host =
          hostPointOf(blockClass.representative());
      // A class the host cannot be priced for would enter the solve with
      // device points only, and be offloaded whatever they cost -- a device
      // point is only evidence against a host alternative. When a host is
      // in scope and only the footprint is missing, the class stays there.
      if (!host &&
          cinm::HostPlatformAttr::getInScope(blockClass.representative()) &&
          !cinm::measureOffloadFootprint(blockClass.representative()).known) {
        staysOnHost(ClassFate::HostUnpriced,
                    "the host cost of this block cannot be read (its "
                    "footprint is unknown), so no device point can be "
                    "weighed against it");
        continue;
      }
      if (host)
        // First: points ascend in resource, and the allocation starts from
        // the cheapest one it can hold everyone on.
        results[ci].points->insert(results[ci].points->begin(), *host);
    }
    solveIndexOfClass[ci] = static_cast<int>(profiles.size());
    // The load of a member is its cost times how often it runs: a block
    // inside a rolled loop runs once per iteration.
    int64_t executions = 0;
    for (const BlockNode &node : graph.nodes)
      if (node.classIndex == ci)
        executions += node.executions;
    profiles.push_back({blockClass.size(), std::move(*results[ci].points),
                        double(executions) / double(blockClass.size())});
  }
  if (profiles.empty())
    return DiagnosedSilenceableFailure::success(); // whole graph on the host

  // Allocation: exact solve over the profiles. The budget is the whole
  // device, since each connected component is interpreted as owning the
  // grid; the per-class menus never exceed it. The co-residency packing is
  // bounded by every memory level the platform declares -- levels a
  // configuration pins nothing in never bind, so there is nothing to select.
  std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
  AllocationOptions allocOpts;
  allocOpts.resourceBudget = plugin->sharedResourceMax();
  for (CinmLevelDefAttr level : graph.platform.getLevels())
    allocOpts.capacities.push_back(
        {level.getName().getValue().str(), level.getSizeInBytes()});
  allocOpts.programReloadMs = opts.programReloadMs;
  record.allocOpts = allocOpts;

  std::optional<AllocationResult> &alloc = record.alloc;
  SmallVector<int> solveIndexOfNode(graph.nodes.size(), -1);
  // The dependency edges, in the topological order the makespan requires.
  // Built for either objective: the latency solve optimises over them, and
  // the throughput solve is *scored* over them afterwards so both objectives
  // are reported for whichever allocation was chosen. Nodes of skipped
  // (host) classes leave the graph, but the ordering they carried must not:
  // a consumer inherits the device predecessors of a skipped producer. Nodes
  // are topological, so one forward pass settles it.
  SmallVector<GraphNode> nodes;
  {
    SmallVector<SmallVector<unsigned>> skippedPreds(graph.nodes.size());
    for (auto [ni, node] : llvm::enumerate(graph.nodes)) {
      SmallVector<unsigned> preds;
      for (unsigned p : node.predecessors) {
        if (solveIndexOfNode[p] >= 0)
          preds.push_back(static_cast<unsigned>(solveIndexOfNode[p]));
        else
          llvm::append_range(preds, skippedPreds[p]);
      }
      llvm::sort(preds);
      preds.erase(llvm::unique(preds), preds.end());
      if (solveIndexOfClass[node.classIndex] < 0) {
        skippedPreds[ni] = std::move(preds);
        continue;
      }
      solveIndexOfNode[ni] = static_cast<int>(nodes.size());
      nodes.push_back(
          {static_cast<unsigned>(solveIndexOfClass[node.classIndex]),
           node.memberIndex, std::move(preds), node.executions});
    }
  }
  if (isImported) {
    AllocationResult result;
    result.perClass.resize(profiles.size());
    for (auto [ci, solveIndex] : llvm::enumerate(solveIndexOfClass))
      if (solveIndex >= 0) {
        result.perClass[solveIndex].groups = imported[ci].groups;
        for (const GroupAllocation &g : imported[ci].groups)
          result.resourceUsed += g.onHost ? 0 : g.resource;
      }
    SmallVector<unsigned> classOfSolveIndex(profiles.size());
    for (auto [ci, solveIndex] : llvm::enumerate(solveIndexOfClass))
      if (solveIndex >= 0)
        classOfSolveIndex[solveIndex] = ci;
    for (const GraphNode &node : nodes)
      result.groupOfNode.push_back(imported[classOfSolveIndex[node.classIndex]]
                                       .groupOfMember[node.memberIndex]);
    if (result.resourceUsed > allocOpts.resourceBudget)
      return emitDefiniteFailure(loc)
             << "allocation-in: the groups take " << result.resourceUsed
             << " of a budget of " << allocOpts.resourceBudget;
    const AllocationScore score = scoreAllocation(profiles, nodes, result);
    result.objectiveMs =
        opts.latencyObjective ? score.latencyMs : score.throughputMs;
    alloc = std::move(result);
  } else if (opts.latencyObjective) {
    alloc = allocateGraphForLatency(profiles, nodes, allocOpts);
  } else {
    alloc = allocateGraph(profiles, allocOpts);
  }
  if (!alloc)
    return emitSilenceableFailure(loc)
           << "no feasible device allocation for this graph: some class fits "
              "no menu configuration";
  LLVM_DEBUG({
    llvm::dbgs() << "[cinm-inference] Graph '" << graphName
                 << "': " << (opts.latencyObjective ? "makespan" : "bottleneck")
                 << " " << alloc->objectiveMs << " ms, " << alloc->resourceUsed
                 << " / " << allocOpts.resourceBudget << " units pinned\n";
    for (auto [ci, ca] : llvm::enumerate(alloc->perClass))
      for (const GroupAllocation &g : ca.groups)
        llvm::dbgs() << "  class " << ci << ": " << g.size << " member(s) on "
                     << (g.resource ? std::to_string(g.resource)
                                    : std::string("timeshare"))
                     << ", load " << g.loadMs << " ms\n";
  });

  // Score the allocation under both objectives, not just the one solved for:
  // the off-diagonal is what says whether the choice mattered.
  record.score = scoreAllocation(profiles, nodes, *alloc);

  // Finalization: stamp each group's argmin onto its members and commit. The
  // argmin is feasible under the packing by construction -- the allocator
  // admitted the group only if k co-resident copies of this configuration's
  // static footprint fit next to one working region -- so no budgeted
  // re-search is needed for the chosen points. A timeshared group has no
  // reserved set; committing the best point's configuration prices its
  // transient borrow of the device.
  // Which group each member landed on. The latency solve says so per node,
  // since its members are not interchangeable; the throughput solve leaves
  // that free, so members fill the groups in order.
  auto &groupOfMember = record.groupOfMember;
  groupOfMember.resize(graph.classes.size());
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes))
    groupOfMember[ci].resize(blockClass.size(), 0);
  if (alloc->groupOfNode.empty()) {
    for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
      if (solveIndexOfClass[ci] < 0)
        continue;
      const ClassAllocation &classAlloc =
          alloc->perClass[solveIndexOfClass[ci]];
      unsigned member = 0;
      for (auto [gi, group] : llvm::enumerate(classAlloc.groups))
        for (unsigned i = 0; i < group.size; ++i, ++member)
          groupOfMember[ci][member] = gi;
    }
  } else {
    for (auto [ni, node] : llvm::enumerate(graph.nodes))
      if (solveIndexOfNode[ni] >= 0)
        groupOfMember[node.classIndex][node.memberIndex] =
            alloc->groupOfNode[solveIndexOfNode[ni]];
  }

  // Record what the solve decided about each block, on the block. The
  // lowering reads the residency fields back (`slot`, `slots`, `sequence`:
  // CnmToUPMEM's residency slots); the rest is what lets the allocation
  // report be read against the IR it describes -- which of 644 compute
  // blocks is the class-12 outlier, which blocks share a device set. Host
  // classes are stamped too, so that the IR itself says which blocks fell
  // back.
  {
    Builder builder(loc.getContext());
    for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
      // The members of a group share a device set and hold their static
      // operands resident side by side: `slot` is the member's position in
      // that layout and `slots` its width, the k the allocator packed
      // (maxCoResidents). The lowering sizes the resident buffers by
      // `slots` and lands each member's scatter in its own slot. Members are
      // numbered in block order, which is the order their launches come in.
      SmallVector<unsigned> groupSize, slotOfMember(blockClass.members.size());
      SmallVector<SmallVector<ComputeBlockOp>> membersOfGroup;
      if (solveIndexOfClass[ci] >= 0)
        for (auto [mi, member] : llvm::enumerate(blockClass.members)) {
          unsigned gi = groupOfMember[ci][mi];
          if (gi >= groupSize.size()) {
            groupSize.resize(gi + 1, 0);
            membersOfGroup.resize(gi + 1);
          }
          slotOfMember[mi] = groupSize[gi]++;
          membersOfGroup[gi].push_back(member);
        }
      for (auto [mi, member] : llvm::enumerate(blockClass.members)) {
        SmallVector<NamedAttribute> fields{
            builder.getNamedAttr("graph", builder.getStringAttr(graphName)),
            builder.getNamedAttr("class", builder.getI64IntegerAttr(ci)),
            builder.getNamedAttr("member", builder.getI64IntegerAttr(mi)),
        };
        const bool placedOnHost =
            solveIndexOfClass[ci] >= 0 && alloc->perClass[solveIndexOfClass[ci]]
                                              .groups[groupOfMember[ci][mi]]
                                              .onHost;
        if (placedOnHost)
          fields.push_back(
              builder.getNamedAttr("placement", builder.getStringAttr("host")));
        if (solveIndexOfClass[ci] >= 0 && !placedOnHost) {
          unsigned gi = groupOfMember[ci][mi];
          fields.push_back(
              builder.getNamedAttr("group", builder.getI64IntegerAttr(gi)));
          fields.push_back(builder.getNamedAttr(
              "slot", builder.getI64IntegerAttr(slotOfMember[mi])));
          fields.push_back(builder.getNamedAttr(
              "slots", builder.getI64IntegerAttr(groupSize[gi])));
          if (DictionaryAttr sequence =
                  launchSequenceOf(builder, member, membersOfGroup[gi]))
            fields.push_back(builder.getNamedAttr("sequence", sequence));
        }
        member->setAttr(CinmDialect::GRAPH_ALLOC_NAME,
                        builder.getDictionaryAttr(fields));
      }
    }
  }

  // Materialize each PINNED group's device set once, at the top of its
  // container function, with frees at every function exit -- residency is
  // per workload lifetime, and function-top placement is its stand-in until
  // a real init phase exists. The handle is forwarded into each member
  // below; the member's
  // lowering then uses it instead of allocating (CnmToUPMEM's
  // findForwardedWorkgroup). Timeshared groups have no set of their own and
  // keep the per-launch allocation, which prices exactly the eviction the
  // allocator chose for them. Targets whose plugin does not materialize
  // (the default hook) fall back to per-block allocation unchanged.
  SmallVector<SmallVector<Value>> wgOfGroup(graph.classes.size());
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    if (solveIndexOfClass[ci] < 0)
      continue;
    const ClassAllocation &classAlloc = alloc->perClass[solveIndexOfClass[ci]];
    wgOfGroup[ci].assign(classAlloc.groups.size(), Value());
    for (auto [gi, group] : llvm::enumerate(classAlloc.groups)) {
      if (!group.resource)
        continue;
      // Any member locates the container function; a graph is a connected
      // dataflow component, so they all share one.
      ComputeBlockOp first;
      for (auto [mi, member] : llvm::enumerate(blockClass.members))
        if (groupOfMember[ci][mi] == gi) {
          first = member;
          break;
        }
      if (!first)
        continue; // empty group: nothing to own the set
      auto container = first->getParentOfType<FunctionOpInterface>();
      if (!container || container.getFunctionBody().empty())
        continue;

      const ProfilePoint *point =
          pointOf(profiles[solveIndexOfClass[ci]], group);
      OpBuilder builder(container->getContext());
      builder.setInsertionPointToStart(&container.getFunctionBody().front());
      Value wg = plugin->materializeWorkgroupAlloc(builder, first.getLoc(),
                                                   point->config);
      if (!wg)
        continue;
      if (!isa<WorkgroupTypeInterface>(wg.getType()))
        return emitDefiniteFailure(
            first.getLoc(),
            "materializeWorkgroupAlloc returned a value whose type does not "
            "implement WorkgroupTypeInterface; members could never "
            "recognize it as their workgroup");
      for (Block &blk : container.getFunctionBody())
        if (Operation *term = blk.getTerminator();
            term && term->hasTrait<OpTrait::ReturnLike>()) {
          builder.setInsertionPoint(term);
          plugin->materializeWorkgroupFree(builder, first.getLoc(), wg);
        }
      wgOfGroup[ci][gi] = wg;
      LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] class " << ci << " group "
                              << gi << ": device set hoisted to top of @"
                              << container.getName() << "\n");
    }
  }

  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    if (solveIndexOfClass[ci] < 0)
      continue;
    const ClassAllocation &classAlloc = alloc->perClass[solveIndexOfClass[ci]];
    const ClassProfile &profile = profiles[solveIndexOfClass[ci]];
    for (auto [mi, memberRef] : llvm::enumerate(blockClass.members)) {
      ComputeBlockOp block = memberRef; // op handles are cheap to copy
      const GroupAllocation &group = classAlloc.groups[groupOfMember[ci][mi]];
      // The allocation left this one where it is: no configuration to
      // commit, and the block stays a host block (see
      // InferenceOptions::allowHostPlacement).
      if (group.onHost)
        continue;
      const ProfilePoint *point = pointOf(profile, group);

      // Forward the group's set into the member: one more operand, one more
      // block argument (compute_block zips them 1:1). The commit below
      // clones the block into a trial module, so the argument is present
      // during the member's own lowering, which is where it takes effect.
      if (Value wg = wgOfGroup[ci][groupOfMember[ci][mi]]) {
        block->insertOperands(block->getNumOperands(), {wg});
        block.getBody().front().addArgument(wg.getType(), block.getLoc());
      }

      std::unique_ptr<InferencePlugin> memberPlugin =
          makePlugin(graph.platform);
      InferenceOptions memberOpts = opts;
      memberOpts.dumpDir.clear();
      memberOpts.evalSingleSolution = point->config;
      TRY(inferAcceleratorConfig(block, *memberPlugin, memberOpts));
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
    // single solution is a per-block override and bypasses it, as does
    // dump-space-only: the space dump is a per-block artifact, so the
    // per-block loop below is the path that produces it.
    if (opts.graphAllocation && !opts.evalSingleSolution &&
        !opts.dumpSpaceOnly && !probe->sharedResourceParam().empty() &&
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
