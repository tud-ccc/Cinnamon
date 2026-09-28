//===- AllocationReport.cpp - What a graph allocation did, as JSON -------===//
//
// See AllocationReport.h, and cinm_experiments.profiles for the reader.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AllocationReport.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/OperatorDescription.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Location.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#include <cmath>

namespace mlir::cinm {

namespace json = llvm::json;

/// A finite number, or null: JSON has no infinity.
static json::Value finiteOr(double value) {
  return std::isfinite(value) ? json::Value(value) : json::Value(nullptr);
}

/// `file:line:col` where there is one, the printed location otherwise.
static std::string locationOf(Operation *op) {
  if (auto file = op->getLoc()->findInstanceOf<FileLineColLoc>())
    return llvm::formatv("{0}:{1}:{2}", file.getFilename().getValue(),
                         file.getLine(), file.getColumn())
        .str();
  std::string loc;
  llvm::raw_string_ostream os(loc);
  op->getLoc().print(os);
  return loc;
}

/// The loops between `block` and its function, innermost first, with their
/// trip counts where the bounds are constant (null where they are not).
static json::Array loopsOf(ComputeBlockOp block) {
  json::Array loops;
  for (Operation *op = block->getParentOp();
       op && !isa<FunctionOpInterface>(op); op = op->getParentOp()) {
    auto loop = dyn_cast<LoopLikeOpInterface>(op);
    if (!loop)
      continue;
    auto constant = [](std::optional<OpFoldResult> bound) {
      return bound ? getConstantIntValue(*bound) : std::nullopt;
    };
    std::optional<int64_t> lb = constant(loop.getSingleLowerBound());
    std::optional<int64_t> ub = constant(loop.getSingleUpperBound());
    std::optional<int64_t> step = constant(loop.getSingleStep());
    json::Value trip = nullptr;
    if (lb && ub && step && *step > 0)
      trip = *ub > *lb ? (*ub - *lb + *step - 1) / *step : 0;
    loops.push_back(json::Object{{"op", op->getName().getStringRef().str()},
                                 {"trip", std::move(trip)}});
  }
  return loops;
}

static json::Object describeFootprint(const OffloadFootprint &f) {
  if (!f.known)
    return json::Object{{"known", false}};
  return json::Object{
      {"known", true},
      {"work_ops", f.work},
      {"static_bytes", f.staticBytes},
      {"dynamic_bytes", f.dynamicBytes},
      {"dynamic_out_bytes", f.dynamicOutBytes},
      {"mul_dtype", f.mulType ? json::Value(describeElementType(f.mulType))
                              : json::Value(nullptr)},
      {"acc_dtype", f.accType ? json::Value(describeElementType(f.accType))
                              : json::Value(nullptr)},
  };
}

static json::Object describeHost(const HostModel &host) {
  return json::Object{
      {"ops_per_s", host.opsPerSecond},
      {"dram_bytes_per_s", host.dramBytesPerSecond},
      {"scalar_op_ns", host.scalarOpNs},
      {"vector_op_ns", host.vectorOpNs},
      {"vector_bytes", host.vectorBytes},
      {"stream_bytes_per_s", host.streamBytesPerSecond},
      {"copy_bytes_per_s", host.copyBytesPerSecond},
  };
}

static json::Value describeConfig(const llvm::StringMap<ParmValue> &config) {
  // Sorted, since a StringMap has no order of its own.
  SmallVector<StringRef> dims;
  for (const auto &entry : config)
    dims.push_back(entry.getKey());
  llvm::sort(dims);
  json::Object out;
  for (StringRef dim : dims)
    out[dim] = static_cast<int64_t>(config.lookup(dim));
  return out;
}

/// One device point of a class: everything that happened at one value of
/// its menu.
static json::Object describeMenuPoint(const MenuPointTrace &point,
                                      bool inProfile,
                                      const std::filesystem::path &graphDir) {
  json::Object entry{{"resource", point.resource}, {"where", "device"}};

  if (point.verdict)
    entry["roofline"] = json::Object{
        {"ms", point.verdict->deviceMs},
        {"transfer_ms", point.verdict->transferMs},
        {"ops_per_s", point.verdict->deviceOpsPerSecond},
        {"transfer_ms_streamed", point.verdict->transferMsIfNothingResident},
    };
  else
    entry["roofline"] = nullptr;
  entry["screen"] = point.verdict
                        ? json::Value(point.verdict->kept ? "kept" : "dropped")
                        : json::Value(nullptr);
  entry["selected"] = point.profiled;

  // Priced by the best evidence it has: a search that found something, else
  // the device roofline the screen read, else nothing at all.
  const bool searched = llvm::any_of(point.searches, [](const auto &search) {
    return search.costMs.has_value();
  });
  entry["priced_by"] = searched        ? "search"
                       : point.verdict ? "device_roofline"
                                       : "none";

  json::Array searches;
  for (const SearchOutcome &search : point.searches) {
    json::Object s{{"repeat", static_cast<int64_t>(search.repeat)},
                   {"rng_seed", search.rngSeed}};
    s["cost_ms"] =
        search.costMs ? json::Value(*search.costMs) : json::Value(nullptr);
    if (!search.failure.empty())
      s["failure"] = search.failure;
    searches.push_back(std::move(s));
  }
  entry["searches"] = std::move(searches);

  if (const std::optional<ProfilePoint> &p = point.point) {
    entry["cost_ms"] = p->costMs;
    entry["config"] = describeConfig(p->config);
    entry["transfer_share"] = p->transferShare >= 0
                                  ? json::Value(p->transferShare)
                                  : json::Value(nullptr);
    entry["weight_scatter_ms"] = p->residency.weightScatterMs;
    json::Object residency;
    for (const LevelResidency &level : p->residency.levels)
      residency[level.level] = json::Object{{"static_bytes", level.staticBytes},
                                            {"dyn_bytes", level.dynBytes}};
    entry["residency"] = std::move(residency);
  } else {
    entry["cost_ms"] = nullptr;
  }
  entry["dominated_by"] =
      point.dominatedBy ? json::Value(point.dominatedBy) : json::Value(nullptr);
  entry["in_profile"] = inProfile;
  if (!point.dumpDir.empty())
    entry["dump"] = std::filesystem::path(point.dumpDir)
                        .lexically_relative(graphDir)
                        .string();
  return entry;
}

static StringRef fateName(const GraphRecord &record, ClassFate::Kind kind) {
  if (!record.profiled)
    return "dry_run";
  switch (kind) {
  case ClassFate::Solved:
    return "solved";
  case ClassFate::Unprofiled:
    return "unprofiled";
  case ClassFate::LosesToHost:
    return "loses_to_host";
  case ClassFate::TransferBound:
    return "transfer_bound";
  }
  llvm_unreachable("unknown class fate");
}

/// Read what the report says about `graph`'s IR into `record`, while the IR
/// is still what the run started from (GraphRecord::nodes). The references
/// must be prepared: the operator is read off them.
void snapshotGraph(const ComputeGraph &graph, GraphRecord &record) {
  for (const BlockNode &node : graph.nodes) {
    json::Array preds;
    for (unsigned p : node.predecessors)
      preds.push_back(static_cast<int64_t>(p));
    record.nodes.push_back(json::Object{
        {"class", static_cast<int64_t>(node.classIndex)},
        {"member", static_cast<int64_t>(node.memberIndex)},
        {"loc", locationOf(node.block)},
        {"executions", node.executions},
        {"loops", loopsOf(node.block)},
        {"predecessors", std::move(preds)},
    });
  }

  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    ComputeBlockOp rep = blockClass.representative();
    json::Object c;
    c["class"] = static_cast<int64_t>(ci);
    auto tag = rep->getAttrOfType<StringAttr>(CinmDialect::DEBUG_TAG_NAME);
    c["debug_tag"] = tag ? json::Value(tag.getValue()) : json::Value(nullptr);
    c["loc"] = locationOf(rep);
    c["members"] = static_cast<int64_t>(blockClass.size());
    int64_t executions = 0;
    for (const BlockNode &node : graph.nodes)
      if (node.classIndex == ci)
        executions += node.executions;
    c["executions"] = executions;
    OffloadFootprint footprint = measureOffloadFootprint(rep);
    c["footprint"] = describeFootprint(footprint);
    c["operator"] = record.references[ci]
                        ? describeComputeBlock(record.references[ci]->block)
                        : json::Value(nullptr);
    c["reference"] = nullptr; // writeReferenceModules, when there is a dump
    record.classes.push_back(std::move(c));
    record.footprints.push_back(footprint);
    auto host = HostPlatformAttr::getInScope(rep);
    record.hosts.push_back(host ? std::optional(host.getModel())
                                : std::nullopt);
  }
}

void writeReferenceModules(const std::filesystem::path &graphDir,
                           StringRef graphName, GraphRecord &record) {
  for (auto [ci, reference] : llvm::enumerate(record.references)) {
    if (!reference)
      continue;
    const std::string function = (graphName + "_class" + Twine(ci)).str();
    const std::filesystem::path relative =
        std::filesystem::path("class_" + std::to_string(ci)) / "reference.mlir";
    ModuleOp source = reference->module.get();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(source->clone()));
    copy->walk([&](func::FuncOp func) { func.setSymName(function); });

    std::error_code ec;
    std::filesystem::create_directories((graphDir / relative).parent_path(),
                                        ec);
    llvm::raw_fd_ostream os((graphDir / relative).string(), ec);
    if (ec) {
      llvm::errs() << "could not write " << (graphDir / relative).string()
                   << ": " << ec.message() << "\n";
      continue;
    }
    copy->print(os);
    record.classes[ci]["reference"] = json::Object{
        {"path", relative.string()},
        {"function", function},
    };
  }
}

/// Write everything one graph's allocation did to `path` as JSON: the machine
/// it priced against, the options it ran under, the graph, and per class the
/// operator, every point of its menu with how it was priced and what became
/// of it, and the groups the allocation gave it.
void writeAllocationReport(const std::filesystem::path &path,
                           const ComputeGraph &graph, StringRef graphName,
                           StringRef platformName, InferencePlugin &plugin,
                           const InferenceOptions &opts,
                           const GraphRecord &record) {
  const std::filesystem::path graphDir = path.parent_path();
  const bool placementIsSolved =
      opts.allowHostPlacement && opts.latencyObjective;

  json::Object root;
  root["schema"] = 1;
  root["graph"] = graphName;
  root["platform"] = platformName;

  if (!record.hosts.empty() && record.hosts.front()) {
    json::Object h = describeHost(*record.hosts.front());
    h["achieved_fraction"] = opts.hostAchievedFraction;
    root["host"] = std::move(h);
  } else {
    root["host"] = nullptr;
  }

  json::Array levels;
  for (CinmLevelDefAttr level : graph.platform.getLevels())
    levels.push_back(json::Object{{"name", level.getName().getValue()},
                                  {"bytes", level.getSizeInBytes()}});
  root["device"] = json::Object{
      {"resource_param", plugin.sharedResourceParam()},
      {"resource_max", plugin.sharedResourceMax()},
      {"levels", std::move(levels)},
  };

  root["options"] = json::Object{
      {"objective", opts.latencyObjective ? "latency" : "throughput"},
      {"screen_menu", opts.screenMenuAgainstHost},
      {"host_achieved_fraction", opts.hostAchievedFraction},
      {"allow_host_placement", opts.allowHostPlacement},
      {"placement_is_solved", placementIsSolved},
      {"max_menu_points", opts.maxMenuPoints},
      {"profile_repair", opts.profileRepair},
      {"profile_seeds", opts.profileSeeds},
      {"host_transfer_bound_share", opts.hostTransferBoundShare},
      {"program_reload_ms", opts.programReloadMs},
      {"max_evals", opts.maxEvals},
      {"n_init", opts.nInit},
      {"rng_seed", opts.rngSeed},
      {"exhaustive_search", opts.exhaustiveSearch},
      {"sample_n", static_cast<int64_t>(opts.sampleN)},
      {"stamp_configs", opts.stampConfigs},
      {"gate_dry_run", opts.gateDryRun},
  };

  root["nodes"] = json::Value(json::Array(record.nodes));

  json::Array classes;
  for (auto [ci, blockClass] : llvm::enumerate(graph.classes)) {
    json::Object c = record.classes[ci];
    const OffloadFootprint &footprint = record.footprints[ci];
    const std::optional<HostModel> &host = record.hosts[ci];
    const double hostMs = footprint.known && host
                              ? hostRooflineSeconds(footprint, *host) * 1e3
                              : 0.0;

    const ClassFate &fate = record.fates[ci];
    c["fate"] = fateName(record, fate.kind);
    c["reason"] =
        fate.reason.empty() ? json::Value(nullptr) : json::Value(fate.reason);

    const int solveIndex =
        record.solveIndexOfClass.empty() ? -1 : record.solveIndexOfClass[ci];
    const ClassProfile *profile =
        solveIndex >= 0 ? &record.profiles[solveIndex] : nullptr;
    auto inProfile = [&](int64_t resource, bool onHost) {
      return profile && llvm::any_of(profile->points, [&](const auto &p) {
               return p.onHost == onHost && (onHost || p.resource == resource);
             });
    };

    // Point 0 is the host: what it would take for one execution, as a
    // roofline. The menu screen compares against it slowed down to what a
    // real host achieves; the screen after profiling, and the allocation when
    // it decides placement, against the roofline itself.
    json::Array points;
    if (hostMs > 0.0) {
      json::Object hostPoint{{"resource", 0},
                             {"where", "host"},
                             {"priced_by", "host_roofline"},
                             {"cost_ms", hostMs},
                             {"in_profile", inProfile(0, true)}};
      hostPoint["screen_cost_ms"] =
          record.traces[ci].screenHostMs > 0.0
              ? json::Value(record.traces[ci].screenHostMs)
              : json::Value(nullptr);
      const bool computeBound =
          footprint.work / host->opsPerSecond >=
          (footprint.staticBytes + footprint.dynamicBytes) /
              host->dramBytesPerSecond;
      hostPoint["bound"] = computeBound ? "compute" : "bandwidth";
      points.push_back(std::move(hostPoint));
    }
    for (const MenuPointTrace &point : record.traces[ci].menu)
      points.push_back(
          describeMenuPoint(point, inProfile(point.resource, false), graphDir));
    c["points"] = std::move(points);

    if (profile && record.alloc) {
      const ClassAllocation &classAlloc = record.alloc->perClass[solveIndex];
      json::Array groups;
      for (auto [gi, group] : llvm::enumerate(classAlloc.groups)) {
        const ProfilePoint *point = pointOf(*profile, group);
        json::Array members;
        if (ci < record.groupOfMember.size())
          for (auto [mi, g] : llvm::enumerate(record.groupOfMember[ci]))
            if (g == gi)
              members.push_back(static_cast<int64_t>(mi));
        groups.push_back(json::Object{
            {"group", static_cast<int64_t>(gi)},
            {"size", static_cast<int64_t>(group.size)},
            {"on_host", group.onHost},
            {"timeshared", !group.onHost && group.resource == 0},
            {"resource", group.resource},
            {"point_resource", point->onHost ? 0 : point->resource},
            {"cost_ms", point->costMs},
            {"load_ms", group.loadMs},
            {"members", std::move(members)},
        });
      }
      c["groups"] = std::move(groups);
    } else {
      c["groups"] = nullptr;
    }
    classes.push_back(std::move(c));
  }
  root["classes"] = std::move(classes);

  if (record.alloc && record.allocOpts) {
    json::Array capacities;
    for (const LevelCapacity &level : record.allocOpts->capacities)
      capacities.push_back(
          json::Object{{"level", level.level}, {"bytes", level.bytes}});
    json::Object a{
        {"objective_ms", record.alloc->objectiveMs},
        {"resource_used", record.alloc->resourceUsed},
        {"resource_budget", record.allocOpts->resourceBudget},
        {"capacities", std::move(capacities)},
    };
    // Both objectives, whichever one was solved for: the off-diagonal is
    // what says whether the choice mattered. Latency is null when a set is
    // timeshared (see scoreAllocation).
    a["throughput_ms"] =
        record.score ? finiteOr(record.score->throughputMs) : nullptr;
    a["latency_ms"] =
        record.score ? finiteOr(record.score->latencyMs) : nullptr;
    root["allocation"] = std::move(a);
  } else {
    root["allocation"] = nullptr;
  }

  std::error_code ec;
  std::filesystem::create_directories(graphDir, ec);
  llvm::raw_fd_ostream os(path.string(), ec);
  if (ec) {
    llvm::errs() << "could not write " << path.string() << ": " << ec.message()
                 << "\n";
    return;
  }
  os << llvm::formatv("{0:2}", json::Value(std::move(root))) << "\n";
}

} // namespace mlir::cinm
