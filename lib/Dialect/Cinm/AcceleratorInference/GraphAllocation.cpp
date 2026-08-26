#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int64_t kNoBudget = std::numeric_limits<int64_t>::max();
constexpr unsigned kNoNode = std::numeric_limits<unsigned>::max();

/// One way a group of co-resident members may be provisioned: a profile
/// point (pinned) or the timeshare pseudo-allocation. `loadPerMember` is the
/// per-inference cost one member contributes; a group of k costs k times it
/// (identical programs run sequentially on their set). `kMax` is the
/// co-residency cap: how many members may share a set at all under this
/// option's memory footprints.
struct GroupOption {
  int64_t resource;     // 0 = unpinned
  double loadPerMember; // ms
  unsigned kMax;        // co-residency cap (<= class multiplicity)
};

/// All provisioning options of one class.
struct ClassOptions {
  unsigned multiplicity;
  SmallVector<GroupOption> options;
};

/// Largest k such that k members may share a set at `point`'s configuration:
/// in every capacity-bounded memory level, their k pinned footprints plus
/// the one shared working region must fit, k * static + dyn <= capacity.
/// Levels the configuration pins nothing in never bind; the check is off
/// when no capacities are declared.
unsigned maxCoResidents(const ProfilePoint &point,
                        ArrayRef<LevelCapacity> capacities,
                        unsigned multiplicity) {
  int64_t k = multiplicity;
  for (const LevelCapacity &capacity : capacities) {
    const LevelResidency *level = point.residency.find(capacity.level);
    if (!level || level->staticBytes <= 0)
      continue;
    int64_t room = capacity.bytes - level->dynBytes;
    k = std::min<int64_t>(
        k, room < level->staticBytes ? 0 : room / level->staticBytes);
  }
  return static_cast<unsigned>(std::max<int64_t>(0, k));
}

/// Minimum total pinned resource with which `cls`'s members can all be
/// grouped so that every group's load stays <= target. The DP peels a group
/// off the front: interchangeability means only counts matter.
/// Returns kNoBudget when impossible.
int64_t minBudgetFor(const ClassOptions &cls, double target,
                     SmallVector<GroupAllocation> *outGroups = nullptr) {
  const unsigned n = cls.multiplicity;
  SmallVector<int64_t> dp(n + 1, kNoBudget);
  // choice[j]: (option index, group size) that realizes dp[j].
  SmallVector<std::pair<int, unsigned>> choice(n + 1, {-1, 0});
  dp[0] = 0;
  for (unsigned j = 1; j <= n; ++j) {
    for (auto [oi, o] : llvm::enumerate(cls.options)) {
      unsigned kMax = std::min(o.kMax, j);
      for (unsigned k = 1; k <= kMax; ++k) {
        if (double(k) * o.loadPerMember > target)
          break; // larger k only increases the load
        if (dp[j - k] == kNoBudget)
          continue;
        int64_t budget = dp[j - k] + o.resource;
        if (budget < dp[j]) {
          dp[j] = budget;
          choice[j] = {static_cast<int>(oi), k};
        }
      }
    }
  }
  if (outGroups && dp[n] != kNoBudget) {
    for (unsigned j = n; j > 0;) {
      auto [oi, k] = choice[j];
      const GroupOption &o = cls.options[oi];
      outGroups->push_back({k, o.resource, double(k) * o.loadPerMember});
      j -= k;
    }
  }
  return dp[n];
}

} // namespace

std::optional<AllocationResult> allocateGraph(ArrayRef<ClassProfile> classes,
                                              const AllocationOptions &opts) {
  // Expand profiles into group options and collect every achievable group
  // load: k * loadPerMember for k up to the co-residency cap. The optimum's
  // bottleneck is the max of its groups' loads, so it is one of these values;
  // searching only them makes the parametric solve exact.
  SmallVector<ClassOptions> all;
  SmallVector<double> candidates;
  for (const ClassProfile &cls : classes) {
    ClassOptions co;
    co.multiplicity = cls.multiplicity;
    double bestPinned = kInf, bestScatter = 0;
    for (const ProfilePoint &p : cls.points) {
      co.options.push_back(
          {p.resource, p.costMs,
           maxCoResidents(p, opts.capacities, cls.multiplicity)});
      if (p.costMs < bestPinned) {
        bestPinned = p.costMs;
        bestScatter = p.residency.weightScatterMs;
      }
    }
    if (opts.allowTimeshare && bestPinned < kInf) {
      // Unpinned: borrow the best point's device count transiently; pay the
      // program switch and the weight re-scatter every inference. Nothing
      // stays resident, so the capacity check does not constrain it.
      co.options.push_back({0, bestPinned + opts.programReloadMs + bestScatter,
                            cls.multiplicity});
    }
    for (const GroupOption &o : co.options)
      for (unsigned k = 1; k <= o.kMax; ++k)
        candidates.push_back(double(k) * o.loadPerMember);
    all.push_back(std::move(co));
  }

  llvm::sort(candidates);
  candidates.erase(llvm::unique(candidates), candidates.end());

  // Feasibility of a bottleneck target is monotone: more permissive targets
  // only widen every class's option set. Binary search the smallest feasible
  // candidate.
  auto feasible = [&](double target) -> bool {
    int64_t total = 0;
    for (const ClassOptions &cls : all) {
      int64_t need = minBudgetFor(cls, target);
      if (need == kNoBudget)
        return false;
      total += need;
      if (total > opts.resourceBudget)
        return false;
    }
    return true;
  };

  size_t lo = 0, hi = candidates.size();
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (feasible(candidates[mid]))
      hi = mid;
    else
      lo = mid + 1;
  }
  if (lo == candidates.size())
    return std::nullopt;

  const double target = candidates[lo];
  AllocationResult result;
  for (const ClassOptions &cls : all) {
    ClassAllocation alloc;
    int64_t used = minBudgetFor(cls, target, &alloc.groups);
    (void)used;
    for (const GroupAllocation &g : alloc.groups) {
      result.objectiveMs = std::max(result.objectiveMs, g.loadMs);
      result.resourceUsed += g.resource;
    }
    result.perClass.push_back(std::move(alloc));
  }
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Allocation: bottleneck "
                          << result.objectiveMs << " ms, "
                          << result.resourceUsed << " / " << opts.resourceBudget
                          << " resource units\n");
  return result;
}

// ===----------------------------------------------------------------------===//
// Latency: critical-path greedy
// ===----------------------------------------------------------------------===//

namespace {

/// A device set under construction: which class it serves, which profile
/// point provisions it, and which nodes live on it (ascending, i.e. the
/// order they execute in).
struct LatencyGroup {
  unsigned classIndex;
  unsigned point; ///< index into ClassProfile::points
  SmallVector<unsigned> members;
};

/// Makespan of one inference under `groups`: a longest path over the
/// dependency edges plus the serialization edges a set imposes on the nodes
/// sharing it. `nodes` is topologically ordered, so one forward sweep does
/// it -- the serialization edges cannot break that, since they run from an
/// earlier member of a set to a later one and members are stored ascending.
double makespanOf(ArrayRef<ClassProfile> classes, ArrayRef<GraphNode> nodes,
                  ArrayRef<LatencyGroup> groups) {
  SmallVector<double> cost(nodes.size(), 0.0);
  SmallVector<unsigned> prevOnSet(nodes.size(), kNoNode);
  for (const LatencyGroup &group : groups) {
    double each = classes[group.classIndex].points[group.point].costMs;
    for (auto [i, member] : llvm::enumerate(group.members)) {
      cost[member] = each;
      if (i)
        prevOnSet[member] = group.members[i - 1];
    }
  }

  SmallVector<double> finish(nodes.size(), 0.0);
  double makespan = 0;
  for (auto [i, node] : llvm::enumerate(nodes)) {
    double start = 0;
    for (unsigned pred : node.predecessors)
      start = std::max(start, finish[pred]);
    if (prevOnSet[i] != kNoNode)
      start = std::max(start, finish[prevOnSet[i]]);
    finish[i] = start + cost[i];
    makespan = std::max(makespan, finish[i]);
  }
  return makespan;
}

/// The nodes with no slack: those on some longest path. Both a forward and a
/// backward sweep, over the same edges makespanOf walks.
llvm::BitVector criticalNodes(ArrayRef<ClassProfile> classes,
                              ArrayRef<GraphNode> nodes,
                              ArrayRef<LatencyGroup> groups) {
  SmallVector<double> cost(nodes.size(), 0.0);
  SmallVector<unsigned> prevOnSet(nodes.size(), kNoNode);
  SmallVector<unsigned> nextOnSet(nodes.size(), kNoNode);
  for (const LatencyGroup &group : groups) {
    double each = classes[group.classIndex].points[group.point].costMs;
    for (auto [i, member] : llvm::enumerate(group.members)) {
      cost[member] = each;
      if (i) {
        prevOnSet[member] = group.members[i - 1];
        nextOnSet[group.members[i - 1]] = member;
      }
    }
  }

  SmallVector<SmallVector<unsigned>> successors(nodes.size());
  SmallVector<double> finish(nodes.size(), 0.0);
  double makespan = 0;
  for (auto [i, node] : llvm::enumerate(nodes)) {
    double start = 0;
    for (unsigned pred : node.predecessors) {
      start = std::max(start, finish[pred]);
      successors[pred].push_back(i);
    }
    if (prevOnSet[i] != kNoNode)
      start = std::max(start, finish[prevOnSet[i]]);
    finish[i] = start + cost[i];
    makespan = std::max(makespan, finish[i]);
  }

  // Latest finish without pushing the makespan out. Successors and the next
  // member on a set both have larger indices, so one reverse sweep does it.
  SmallVector<double> latest(nodes.size(), makespan);
  for (unsigned i = nodes.size(); i-- > 0;) {
    for (unsigned succ : successors[i])
      latest[i] = std::min(latest[i], latest[succ] - cost[succ]);
    if (nextOnSet[i] != kNoNode)
      latest[i] =
          std::min(latest[i], latest[nextOnSet[i]] - cost[nextOnSet[i]]);
  }

  llvm::BitVector critical(nodes.size());
  const double epsilon = 1e-9 * std::max(1.0, makespan);
  for (unsigned i = 0; i < nodes.size(); ++i)
    if (latest[i] - finish[i] <= epsilon)
      critical.set(i);
  return critical;
}

int64_t budgetOf(ArrayRef<ClassProfile> classes,
                 ArrayRef<LatencyGroup> groups) {
  int64_t total = 0;
  for (const LatencyGroup &group : groups)
    total += classes[group.classIndex].points[group.point].resource;
  return total;
}

/// Cheapest point of `cls` that can hold `k` co-resident members, or nullopt
/// if none can. Points ascend in resource, and a larger set means a smaller
/// per-unit footprint, so the first match is also the cheapest.
std::optional<unsigned> cheapestPointFor(const ClassProfile &cls,
                                         ArrayRef<LevelCapacity> capacities,
                                         unsigned k) {
  for (auto [pi, p] : llvm::enumerate(cls.points))
    if (maxCoResidents(p, capacities, k) >= k)
      return static_cast<unsigned>(pi);
  return std::nullopt;
}

} // namespace

std::optional<AllocationResult>
allocateGraphForLatency(ArrayRef<ClassProfile> classes,
                        ArrayRef<GraphNode> nodes,
                        const AllocationOptions &opts) {
  // Start from maximal merging: one set per class, holding all its members at
  // the smallest size that can hold them. This is the cheapest allocation
  // there is, and on a layered model it is already most of the answer --
  // members that are sequentially dependent never overlap, so sharing a set
  // costs them nothing.
  SmallVector<LatencyGroup> groups;
  SmallVector<SmallVector<unsigned>> membersOf(classes.size());
  for (auto [ni, node] : llvm::enumerate(nodes))
    membersOf[node.classIndex].push_back(ni);
  for (auto [ci, cls] : llvm::enumerate(classes)) {
    SmallVector<unsigned> &members = membersOf[ci];
    if (members.empty())
      continue;
    // If the whole class cannot be co-resident, chunk it into as few sets as
    // its best point allows.
    unsigned cap = 0;
    for (const ProfilePoint &p : cls.points)
      cap = std::max(cap, maxCoResidents(p, opts.capacities, members.size()));
    if (cap == 0)
      return std::nullopt;
    for (unsigned at = 0; at < members.size(); at += cap) {
      unsigned k = std::min<unsigned>(cap, members.size() - at);
      std::optional<unsigned> point = cheapestPointFor(cls, opts.capacities, k);
      if (!point)
        return std::nullopt;
      groups.push_back(
          {static_cast<unsigned>(ci), *point,
           SmallVector<unsigned>(llvm::ArrayRef(members).slice(at, k))});
    }
  }
  int64_t used = budgetOf(classes, groups);
  if (used > opts.resourceBudget)
    return std::nullopt;

  // Greedy: apply the move with the best makespan reduction per extra device
  // unit until nothing improves. Every accepted move spends budget, so this
  // terminates against opts.resourceBudget.
  while (true) {
    const double current = makespanOf(classes, nodes, groups);
    const llvm::BitVector critical = criticalNodes(classes, nodes, groups);
    SmallVector<LatencyGroup> best;
    double bestScore = 0;
    // Fallback for the case a pure improvement rule cannot see past: when a
    // set's cost is masked by an equally slow sibling, widening either alone
    // gains nothing and the greedy stalls one move short of widening both.
    // So a *grow* that does not make things worse is taken when nothing
    // better is on offer, provided the set carries a node with no slack --
    // spare units spent there can still pay off, spent anywhere else they
    // are pinned for nothing. (Splits are not eligible: a new set is real
    // waste when it buys nothing.)
    SmallVector<LatencyGroup> filler;
    int64_t fillerSpend = 0;

    auto consider = [&](SmallVector<LatencyGroup> trial, bool isGrow,
                        bool isCritical) {
      int64_t spend = budgetOf(classes, trial) - used;
      if (spend <= 0 || used + spend > opts.resourceBudget)
        return;
      double gain = current - makespanOf(classes, nodes, trial);
      if (gain > 0) {
        double score = gain / double(spend);
        if (score > bestScore) {
          bestScore = score;
          best = std::move(trial);
        }
        return;
      }
      if (gain == 0 && isGrow && isCritical &&
          (filler.empty() || spend < fillerSpend)) {
        fillerSpend = spend;
        filler = std::move(trial);
      }
    };

    for (auto [gi, group] : llvm::enumerate(groups)) {
      const ClassProfile &cls = classes[group.classIndex];
      bool onPath = llvm::any_of(
          group.members, [&](unsigned member) { return critical[member]; });
      // Grow: the same members on a larger set.
      for (unsigned pi = group.point + 1; pi < cls.points.size(); ++pi) {
        if (maxCoResidents(cls.points[pi], opts.capacities,
                           group.members.size()) < group.members.size())
          continue;
        SmallVector<LatencyGroup> trial(groups);
        trial[gi].point = pi;
        consider(std::move(trial), /*isGrow=*/true, onPath);
      }
      // Split: one member peeled off onto a set of its own. Which member
      // matters -- only one on the critical path pays off -- and so does how
      // wide the new set is, since a split that lands on a slower size buys
      // nothing. Both are enumerated.
      if (group.members.size() < 2)
        continue;
      for (unsigned mi = 0; mi < group.members.size(); ++mi) {
        for (auto [pi, p] : llvm::enumerate(cls.points)) {
          if (maxCoResidents(p, opts.capacities, 1) < 1)
            continue;
          SmallVector<LatencyGroup> trial(groups);
          unsigned member = trial[gi].members[mi];
          trial[gi].members.erase(trial[gi].members.begin() + mi);
          trial.push_back(
              {group.classIndex, static_cast<unsigned>(pi), {member}});
          consider(std::move(trial), /*isGrow=*/false, onPath);
        }
      }
    }

    if (bestScore > 0)
      groups = std::move(best);
    else if (!filler.empty())
      groups = std::move(filler);
    else
      break;
    used = budgetOf(classes, groups);
  }

  // Report: groups in class order, and the group each node landed on.
  AllocationResult result;
  result.perClass.resize(classes.size());
  result.groupOfNode.assign(nodes.size(), 0);
  for (const LatencyGroup &group : groups) {
    const ProfilePoint &point = classes[group.classIndex].points[group.point];
    ClassAllocation &alloc = result.perClass[group.classIndex];
    unsigned index = alloc.groups.size();
    alloc.groups.push_back({static_cast<unsigned>(group.members.size()),
                            point.resource,
                            double(group.members.size()) * point.costMs});
    for (unsigned member : group.members)
      result.groupOfNode[member] = index;
  }
  result.objectiveMs = makespanOf(classes, nodes, groups);
  result.resourceUsed = used;
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Allocation: makespan "
                          << result.objectiveMs << " ms, "
                          << result.resourceUsed << " / " << opts.resourceBudget
                          << " resource units\n");
  return result;
}

AllocationScore scoreAllocation(ArrayRef<ClassProfile> classes,
                                ArrayRef<GraphNode> nodes,
                                const AllocationResult &result) {
  AllocationScore score;

  // Throughput needs no schedule: a set's load is its size times its point's
  // cost, whichever members it holds, and the objective is the worst set.
  for (const ClassAllocation &classAlloc : result.perClass)
    for (const GroupAllocation &group : classAlloc.groups)
      score.throughputMs = std::max(score.throughputMs, group.loadMs);

  // Latency needs one. Rebuild the sets as the makespan walk wants them:
  // the profile point each was provisioned at, and its members ascending.
  SmallVector<LatencyGroup> groups;
  // Where a class's groups start in `groups`, so the member deal below can
  // find them, and how many of this class's nodes have been dealt so far.
  SmallVector<unsigned> firstGroupOfClass(result.perClass.size(), 0);
  for (auto [ci, classAlloc] : llvm::enumerate(result.perClass)) {
    firstGroupOfClass[ci] = groups.size();
    for (const GroupAllocation &group : classAlloc.groups) {
      // A timeshared set runs no fixed point, so there is no per-node cost
      // and no makespan to report for this allocation.
      if (group.resource == 0) {
        score.latencyMs = std::numeric_limits<double>::quiet_NaN();
        return score;
      }
      // Groups carry the resource they were provisioned at, not the index of
      // the point that provisioned them; points are unique in resource and
      // ascending, so the resource recovers it.
      const auto &points = classes[ci].points;
      auto it = llvm::find_if(points, [&](const ProfilePoint &p) {
        return p.resource == group.resource;
      });
      if (it == points.end()) {
        score.latencyMs = std::numeric_limits<double>::quiet_NaN();
        return score;
      }
      groups.push_back({static_cast<unsigned>(ci),
                        static_cast<unsigned>(it - points.begin()),
                        {}});
    }
  }

  if (!result.groupOfNode.empty()) {
    // The latency solve already decided; nodes are walked in order, so each
    // set's members come out ascending as LatencyGroup requires.
    for (auto [ni, node] : llvm::enumerate(nodes)) {
      if (ni >= result.groupOfNode.size())
        break;
      unsigned slot =
          firstGroupOfClass[node.classIndex] + result.groupOfNode[ni];
      if (slot < groups.size())
        groups[slot].members.push_back(ni);
    }
  } else {
    // Round-robin deal, skipping sets already at their size (they need not
    // be equal). See the header for why consecutive nodes are spread.
    SmallVector<unsigned> nextOfClass(result.perClass.size(), 0);
    for (auto [ni, node] : llvm::enumerate(nodes)) {
      const unsigned ci = node.classIndex;
      if (ci >= result.perClass.size())
        continue;
      ArrayRef<GroupAllocation> classGroups = result.perClass[ci].groups;
      if (classGroups.empty())
        continue;
      for (unsigned tried = 0; tried < classGroups.size(); ++tried) {
        const unsigned g = (nextOfClass[ci] + tried) % classGroups.size();
        LatencyGroup &target = groups[firstGroupOfClass[ci] + g];
        if (target.members.size() < classGroups[g].size) {
          target.members.push_back(ni);
          nextOfClass[ci] = (g + 1) % classGroups.size();
          break;
        }
      }
    }
  }

  score.latencyMs = makespanOf(classes, nodes, groups);
  return score;
}

} // namespace mlir::cinm
