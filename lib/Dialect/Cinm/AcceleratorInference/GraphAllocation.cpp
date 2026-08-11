#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int64_t kNoBudget = std::numeric_limits<int64_t>::max();

/// One way a group of co-resident members may be provisioned: a profile
/// point (pinned) or the timeshare pseudo-allocation. `loadPerMember` is the
/// per-inference cost one member contributes; a group of k costs k times it
/// (identical programs run sequentially on their set).
struct GroupOption {
  int64_t resource;     // 0 = unpinned
  double loadPerMember; // ms
  int64_t staticBytes;  // per device unit, per member; co-residents' sum
  int64_t dynBytes;     // per device unit, shared across members (max)
};

/// All provisioning options of one class.
struct ClassOptions {
  unsigned multiplicity;
  SmallVector<GroupOption> options;
};

/// Largest k such that k members may share a set under option `o`: their k
/// pinned footprints plus the one shared working region must fit the unit,
/// k * static + dyn <= capacity. Unlimited when the capacity check is off.
unsigned maxCoResidents(const GroupOption &o, int64_t capacityBytes,
                        unsigned multiplicity) {
  if (capacityBytes <= 0 || o.staticBytes <= 0)
    return multiplicity;
  int64_t room = capacityBytes - o.dynBytes;
  if (room < o.staticBytes)
    return 0;
  return static_cast<unsigned>(
      std::min<int64_t>(multiplicity, room / o.staticBytes));
}

/// Minimum total pinned resource with which `cls`'s members can all be
/// grouped so that every group's load stays <= target. The DP peels a group
/// off the front: interchangeability means only counts matter.
/// Returns kNoBudget when impossible.
int64_t minBudgetFor(const ClassOptions &cls, double target,
                     int64_t capacityBytes,
                     SmallVector<GroupAllocation> *outGroups = nullptr) {
  const unsigned n = cls.multiplicity;
  SmallVector<int64_t> dp(n + 1, kNoBudget);
  // choice[j]: (option index, group size) that realizes dp[j].
  SmallVector<std::pair<int, unsigned>> choice(n + 1, {-1, 0});
  dp[0] = 0;
  for (unsigned j = 1; j <= n; ++j) {
    for (auto [oi, o] : llvm::enumerate(cls.options)) {
      unsigned kMax = maxCoResidents(o, capacityBytes, j);
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
      co.options.push_back({p.resource, p.costMs, p.residency.staticMramBytes,
                            p.residency.dynMramBytes});
      if (p.costMs < bestPinned) {
        bestPinned = p.costMs;
        bestScatter = p.residency.weightScatterMs;
      }
    }
    if (opts.allowTimeshare && bestPinned < kInf) {
      // Unpinned: borrow the best point's device count transiently; pay the
      // program switch and the weight re-scatter every inference. Nothing
      // stays resident, so the capacity check does not constrain it.
      co.options.push_back(
          {0, bestPinned + opts.programReloadMs + bestScatter, 0, 0});
    }
    for (const GroupOption &o : co.options) {
      unsigned kMax = maxCoResidents(o, opts.capacityBytes, co.multiplicity);
      for (unsigned k = 1; k <= kMax; ++k)
        candidates.push_back(double(k) * o.loadPerMember);
    }
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
      int64_t need = minBudgetFor(cls, target, opts.capacityBytes);
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
    int64_t used = minBudgetFor(cls, target, opts.capacityBytes, &alloc.groups);
    (void)used;
    for (const GroupAllocation &g : alloc.groups) {
      result.bottleneckMs = std::max(result.bottleneckMs, g.loadMs);
      result.resourceUsed += g.resource;
    }
    result.perClass.push_back(std::move(alloc));
  }
  LLVM_DEBUG(llvm::dbgs() << "[cinm-inference] Allocation: bottleneck "
                          << result.bottleneckMs << " ms, "
                          << result.resourceUsed << " / " << opts.resourceBudget
                          << " resource units\n");
  return result;
}

} // namespace mlir::cinm
