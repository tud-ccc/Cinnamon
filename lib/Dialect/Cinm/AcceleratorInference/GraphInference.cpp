#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmUtils.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.h"
#include "cinm-mlir/Utils/Scheduling/SchedulingSupport.h"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

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
                                    blockClass.size(), /*predecessors=*/{}});
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

/// Wrap `s` in double quotes, doubling any embedded quotes (RFC 4180).
static std::string csvQuote(StringRef s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += '"';
    out += c;
  }
  out += '"';
  return out;
}

/// Write the measured cost profiles of one graph to `path` as CSV: one row
/// per (class, menu point), plus one measurement-less row per class that no
/// menu point could run, so the file describes every class of the graph.
/// This is exactly what the allocator solves over, laid out for offline
/// analysis of the choice it made.
static void dumpProfilesCSV(const std::filesystem::path &path,
                            const ComputeGraph &graph,
                            ArrayRef<ClassProfile> profiles,
                            ArrayRef<int> solveIndexOfClass) {
  // Level columns are the platform's declared levels, but a plugin may report
  // residency in a level the platform does not declare -- such a level never
  // binds the packing, yet dropping it here would lose measured data -- so the
  // column set is the union, platform levels first.
  SmallVector<std::string> levels;
  for (CinmLevelDefAttr level : graph.platform.getLevels())
    levels.push_back(level.getName().getValue().str());
  for (const ClassProfile &profile : profiles)
    for (const ProfilePoint &point : profile.points)
      for (const LevelResidency &entry : point.residency.levels)
        if (!llvm::is_contained(levels, entry.level))
          levels.push_back(entry.level);

  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path);
  if (!out)
    return;
  // raw_cost_ms is what this point's own pinned search measured; cost_ms is
  // what the point offers after lower-envelope repair, and repaired_from
  // names the smaller resource whose incumbent it carries (empty when the
  // point kept its own). transfer_share is the incumbent's per-inference
  // data-movement share (amortized weight scatters excluded); empty when it
  // was not measured.
  out << "class,debug_tag,location,multiplicity,resource,cost_ms,"
         "raw_cost_ms,repaired_from,transfer_share,weight_scatter_ms,config";
  for (const std::string &level : levels)
    out << ",static_" << level << ",dyn_" << level;
  out << "\n";

  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    ComputeBlockOp rep = blockClass.representative();
    auto tag = rep->getAttrOfType<StringAttr>(CinmDialect::DEBUG_TAG_NAME);
    std::string location;
    llvm::raw_string_ostream(location) << rep.getLoc();
    auto classCols = [&]() {
      out << ci << "," << csvQuote(tag ? tag.getValue() : StringRef()) << ","
          << csvQuote(location) << "," << blockClass.size() << ",";
    };

    if (solveIndexOfClass[ci] < 0) {
      // A class that stays on the host is kept for the record, with every
      // measured column empty.
      classCols();
      out << ",,,,,,";
      for (size_t i = 0, e = 2 * levels.size(); i < e; ++i)
        out << ",";
      out << "\n";
      continue;
    }

    for (const ProfilePoint &point : profiles[solveIndexOfClass[ci]].points) {
      // The configuration is one `dim=value;...` column rather than one column
      // per dimension: the space is stated per class, so the classes of one
      // graph need not agree on their dimensions. Sorted, since a StringMap
      // has no order of its own.
      SmallVector<StringRef> dims;
      for (const auto &entry : point.config)
        dims.push_back(entry.getKey());
      llvm::sort(dims);
      std::string config;
      llvm::raw_string_ostream cfg(config);
      llvm::interleave(
          dims, cfg,
          [&](StringRef dim) { cfg << dim << "=" << point.config.lookup(dim); },
          ";");

      classCols();
      out << point.resource << "," << point.costMs << "," << point.rawCostMs
          << ",";
      if (point.repairedFrom)
        out << point.repairedFrom;
      out << ",";
      if (point.transferShare >= 0)
        out << point.transferShare;
      out << "," << point.residency.weightScatterMs << "," << csvQuote(config);
      for (const std::string &level : levels) {
        const LevelResidency *entry = point.residency.find(level);
        out << ",";
        if (entry)
          out << entry->staticBytes;
        out << ",";
        if (entry)
          out << entry->dynBytes;
      }
      out << "\n";
    }
  }
}

/// Write every search the profiling ran, one row per (class, menu point,
/// repeat). With profileSeeds == 1 this is profiles.csv's cost column again;
/// past that, the spread within one (class, resource) is the search noise the
/// profile's shape has to be read against -- a difference between two menu
/// points smaller than the spread at either of them says nothing.
static void dumpProfileSeedsCSV(const std::filesystem::path &path,
                                StringRef graphName,
                                ArrayRef<SmallVector<ProfileSample>> samples) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path);
  if (!out)
    return;
  out << "graph,class,resource,seed,cost_ms\n";
  for (auto [ci, classSamples] : llvm::enumerate(samples))
    for (const ProfileSample &sample : classSamples)
      out << csvQuote(graphName) << "," << ci << "," << sample.resource << ","
          << sample.seed << "," << sample.costMs << "\n";
}

/// The profile point a group runs: a pinned group replays the point measured
/// at its allocated resource, a timeshared one its best point overall.
static const ProfilePoint *pointOf(const ClassProfile &profile,
                                   const GroupAllocation &group) {
  const ProfilePoint *point = nullptr;
  for (const ProfilePoint &p : profile.points)
    if (group.resource ? p.resource == group.resource
                       : (!point || p.costMs < point->costMs))
      point = &p;
  assert(point && "allocator chose a resource the profile does not have");
  return point;
}

/// Write the shape of one graph's solved allocation to `path` as a
/// single-row CSV: what the graph was made of (blocks, classes), how the
/// allocator carved the device up (groups, pinned and timeshared, resource
/// spent), and what it achieved.
///
/// The host/device split counts the blocks of *this* graph, which are only
/// the ones that target `platformName`: a block goes to the host column when
/// no menu configuration could run it. Blocks the program pins to the host
/// up front are not part of the graph at all and are counted nowhere here.
static void dumpAllocationCSV(
    const std::filesystem::path &path, const ComputeGraph &graph,
    StringRef graphName, StringRef platformName, const InferenceOptions &opts,
    const AllocationOptions &allocOpts, const AllocationResult &alloc,
    ArrayRef<ClassProfile> profiles, ArrayRef<int> solveIndexOfClass,
    const AllocationScore &score) {
  unsigned deviceBlocks = 0, groups = 0, pinnedGroups = 0;
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes))
    if (solveIndexOfClass[ci] >= 0)
      deviceBlocks += blockClass.size();
  for (const ClassAllocation &classAlloc : alloc.perClass)
    for (const GroupAllocation &group : classAlloc.groups) {
      ++groups;
      if (group.resource)
        ++pinnedGroups;
    }

  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path);
  if (!out)
    return;
  // objective_ms is the value of the objective this run solved for;
  // throughput_ms and latency_ms score the same allocation under *both*, so
  // the cost of having optimised the other one is readable off one row.
  // latency_ms is empty when a set is timeshared (see scoreAllocation).
  out << "graph,platform,objective,objective_ms,throughput_ms,latency_ms,"
         "n_blocks,n_blocks_device,"
         "n_blocks_host,n_classes,n_classes_device,n_classes_host,n_groups,"
         "n_groups_pinned,n_groups_timeshared,resource_used,resource_budget\n";
  out << csvQuote(graphName) << "," << csvQuote(platformName) << ","
      << (opts.latencyObjective ? "latency" : "throughput") << ","
      << alloc.objectiveMs << "," << score.throughputMs << ",";
  if (std::isfinite(score.latencyMs))
    out << score.latencyMs;
  out << "," << graph.numBlocks() << "," << deviceBlocks << ","
      << (graph.numBlocks() - deviceBlocks) << "," << graph.classes.size()
      << "," << profiles.size() << ","
      << (graph.classes.size() - profiles.size()) << "," << groups << ","
      << pinnedGroups << "," << (groups - pinnedGroups) << ","
      << alloc.resourceUsed << "," << allocOpts.resourceBudget << "\n";
}

/// Write one row per device set the allocator carved out. `resource` is what
/// the set reserves (0 for a timeshared group, which reserves nothing);
/// `point_resource` is the profile point it runs, which is what joins a row
/// to profiles.csv. `load_ms` is the set's per-inference work -- the term the
/// throughput objective takes the max over -- and exceeds the point's cost
/// by the reload and rescatter a timeshared group pays.
static void dumpGroupsCSV(const std::filesystem::path &path,
                          const ComputeGraph &graph, StringRef graphName,
                          const AllocationResult &alloc,
                          ArrayRef<ClassProfile> profiles,
                          ArrayRef<int> solveIndexOfClass) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path);
  if (!out)
    return;
  out << "graph,class,group,size,resource,point_resource,cost_ms,load_ms,"
         "timeshared\n";
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    if (solveIndexOfClass[ci] < 0)
      continue;
    const ClassProfile &profile = profiles[solveIndexOfClass[ci]];
    const ClassAllocation &classAlloc = alloc.perClass[solveIndexOfClass[ci]];
    for (auto [gi, group] : llvm::enumerate(classAlloc.groups)) {
      const ProfilePoint *point = pointOf(profile, group);
      out << csvQuote(graphName) << "," << ci << "," << gi << "," << group.size
          << "," << group.resource << "," << point->resource << ","
          << point->costMs << "," << group.loadMs << ","
          << (group.resource ? 0 : 1) << "\n";
    }
  }
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

  // Profiling: one cost profile per class, on its representative. A class the
  // platform cannot run at any menu point is not an error at the graph level:
  // its members simply stay on the host (they keep no accelerator annotation,
  // which is what the downstream lowering treats as host execution) and the
  // solve runs over the remaining classes. Only definite failures abort.
  SmallVector<ClassProfile> profiles; // one entry per *kept* class
  SmallVector<int> solveIndexOfClass(graph.classes.size(), -1);
  // Indexed by graph class, unlike `profiles`: a class that found no feasible
  // configuration still ran searches, and what they cost is worth keeping.
  SmallVector<SmallVector<ProfileSample>> samples(graph.classes.size());

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
  };
  std::vector<ClassResult> results(graph.classes.size());

  auto profileClass = [&](size_t ci) {
    const BlockClass &blockClass = graph.classes[ci];
    std::unique_ptr<InferencePlugin> plugin = makePlugin(graph.platform);
    InferenceOptions profileOpts = opts;
    profileOpts.numWorkers = baseWorkers;
    if (!baseDumpDir.empty())
      profileOpts.dumpDir = (std::filesystem::path(baseDumpDir.str()) /
                             graphName.str() / ("class_" + std::to_string(ci)))
                                .string();
    utils::Maybe<SmallVector<ProfilePoint>> points =
        profileComputeBlock(blockClass.representative(), *plugin, profileOpts,
                            &samples[ci], threaded ? &gate : nullptr);
    if (auto *fail = std::get_if<DiagnosedSilenceableFailure>(&points)) {
      if (fail->isDefiniteFailure())
        results[ci].definite = std::move(*fail);
      else
        // Not an error at the graph level, and the warning that says so is
        // emitted below: diagnostics from the sweep would come out in finish
        // order, which is not an order the user can make sense of.
        (void)fail->silence();
      return;
    }
    results[ci].points = std::move(std::get<SmallVector<ProfilePoint>>(points));
  };

  if (classThreads <= 1) {
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

  // Reduce in class order, so the profile list, the diagnostics and the
  // failure that wins are all the ones a serial sweep would have produced.
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    if (results[ci].definite)
      return std::move(*results[ci].definite);
    if (!results[ci].points) {
      blockClass.representative().emitWarning()
          << "no feasible '" << platformName
          << "' configuration for this block; it stays on the host, along "
             "with the "
          << (blockClass.size() - 1) << " other block(s) of its class";
      continue;
    }
    // The transfer-bound gate (see InferenceOptions::hostTransferBoundShare):
    // a class whose best point is the smallest menu value gains nothing from
    // more devices, and when that point is also mostly transfer the device
    // buys it essentially nothing at all -- the conjunction keeps
    // compute-bound classes that merely scale poorly (attention-shaped
    // matmuls) on the device. This is a heuristic in lieu of a host cost
    // model (future work); the evidence behind it is the profile shape
    // itself, see docs/SearchStrategyPlan.md.
    if (opts.hostTransferBoundShare > 0) {
      SmallVector<ProfilePoint> &pts = *results[ci].points;
      const ProfilePoint *bestPt =
          &*llvm::min_element(pts, [](const auto &a, const auto &b) {
            return a.costMs < b.costMs;
          });
      if (bestPt->resource == pts.front().resource &&
          bestPt->transferShare >= opts.hostTransferBoundShare) {
        blockClass.representative().emitWarning()
            << "transfer-bound on '" << platformName << "' ("
            << static_cast<int>(bestPt->transferShare * 100)
            << "% of its best point's cost is data movement, and more "
               "devices do not improve it); it stays on the host, along "
               "with the "
            << (blockClass.size() - 1) << " other block(s) of its class";
        continue;
      }
    }
    solveIndexOfClass[ci] = static_cast<int>(profiles.size());
    profiles.push_back({blockClass.size(), std::move(*results[ci].points)});
  }
  if (!baseDumpDir.empty()) {
    auto dir = std::filesystem::path(baseDumpDir.str()) / graphName.str();
    dumpProfilesCSV(dir / "profiles.csv", graph, profiles, solveIndexOfClass);
    dumpProfileSeedsCSV(dir / "profile_seeds.csv", graphName, samples);
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

  std::optional<AllocationResult> alloc;
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
           node.memberIndex, std::move(preds)});
    }
  }
  if (opts.latencyObjective)
    alloc = allocateGraphForLatency(profiles, nodes, allocOpts);
  else
    alloc = allocateGraph(profiles, allocOpts);
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

  if (!baseDumpDir.empty()) {
    auto dir = std::filesystem::path(baseDumpDir.str()) / graphName.str();
    // Score the allocation under both objectives, not just the one solved
    // for: the off-diagonal is what says whether the choice mattered.
    const AllocationScore score = scoreAllocation(profiles, nodes, *alloc);
    dumpAllocationCSV(dir / "allocation.csv", graph, graphName, platformName,
                      opts, allocOpts, *alloc, profiles, solveIndexOfClass,
                      score);
    dumpGroupsCSV(dir / "groups.csv", graph, graphName, *alloc, profiles,
                  solveIndexOfClass);
  }

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
  SmallVector<SmallVector<unsigned>> groupOfMember(graph.classes.size());
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

  // Record what the solve decided about each block, on the block. Nothing
  // reads it back: it is what lets the CSV dumps be read against the IR they
  // describe -- which of 644 compute blocks is the class-12 outlier, which
  // blocks share a device set. Host classes are stamped too, since "which
  // ones fell back" is exactly the question the dumps leave open.
  {
    Builder builder(loc.getContext());
    for (auto [ci, blockClass] : llvm::enumerate(graph.classes))
      for (auto [mi, member] : llvm::enumerate(blockClass.members)) {
        SmallVector<NamedAttribute> fields{
            builder.getNamedAttr("graph", builder.getStringAttr(graphName)),
            builder.getNamedAttr("class", builder.getI64IntegerAttr(ci)),
            builder.getNamedAttr("member", builder.getI64IntegerAttr(mi)),
        };
        if (solveIndexOfClass[ci] >= 0)
          fields.push_back(builder.getNamedAttr(
              "group", builder.getI64IntegerAttr(groupOfMember[ci][mi])));
        member->setAttr(CinmDialect::GRAPH_ALLOC_NAME,
                        builder.getDictionaryAttr(fields));
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
