//===- GraphAllocationTest.cpp - Exact graph-level allocation -------------===//
//
// allocateGraph divides the device among a graph's classes by a parametric
// min-max solve. The closed-form tests pin the canonical scenarios (a 2MM
// chain choosing partitioning over timesharing, QKV projections packing into
// one set while their weights fit); the differential test checks exactness
// against a brute-force enumeration of every grouping and provisioning on
// small random instances.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/GraphAllocation.h"

#include <gtest/gtest.h>

#include <functional>
#include <random>

using namespace mlir;
using cinm::AllocationOptions;
using cinm::AllocationResult;
using cinm::ClassProfile;
using cinm::ProfilePoint;

namespace {

ProfilePoint point(int64_t d, double cost, int64_t staticBytes = 0,
                   int64_t dynBytes = 0, double scatterMs = 0) {
  ProfilePoint p;
  p.resource = d;
  p.costMs = cost;
  p.residency.staticMramBytes = staticBytes;
  p.residency.dynMramBytes = dynBytes;
  p.residency.weightScatterMs = scatterMs;
  return p;
}

/// Brute-force optimum: recursively assign every member of every class to a
/// group with a chosen option, tracking budget; return the best achievable
/// bottleneck. Exponential; only for tiny instances.
double bruteForce(ArrayRef<ClassProfile> classes,
                  const AllocationOptions &opts) {
  // Enumerate per class: partitions of n into groups with an option each.
  struct Option {
    int64_t resource;
    double loadPerMember;
    int64_t staticBytes, dynBytes;
  };
  std::vector<std::vector<Option>> options(classes.size());
  for (auto [ci, cls] : llvm::enumerate(classes)) {
    for (const ProfilePoint &p : cls.points)
      options[ci].push_back({p.resource, p.costMs, p.residency.staticMramBytes,
                             p.residency.dynMramBytes});
    if (opts.allowTimeshare && !cls.points.empty()) {
      double best = std::numeric_limits<double>::infinity();
      double scatter = 0;
      for (const ProfilePoint &p : cls.points)
        if (p.costMs < best) {
          best = p.costMs;
          scatter = p.residency.weightScatterMs;
        }
      options[ci].push_back({0, best + opts.programReloadMs + scatter, 0, 0});
    }
  }

  double best = std::numeric_limits<double>::infinity();
  // For each class, enumerate groupings recursively; combine across classes.
  std::function<void(size_t, int64_t, double)> perClass =
      [&](size_t ci, int64_t budgetLeft, double maxLoad) {
        if (ci == classes.size()) {
          best = std::min(best, maxLoad);
          return;
        }
        std::function<void(unsigned, int64_t, double)> group =
            [&](unsigned remaining, int64_t budget, double load) {
              if (budget < 0)
                return;
              if (remaining == 0) {
                perClass(ci + 1, budget, load);
                return;
              }
              for (const Option &o : options[ci])
                for (unsigned k = 1; k <= remaining; ++k) {
                  if (opts.capacityBytes > 0 && o.staticBytes > 0 &&
                      int64_t(k) * o.staticBytes + o.dynBytes >
                          opts.capacityBytes)
                    continue;
                  group(remaining - k, budget - o.resource,
                        std::max(load, double(k) * o.loadPerMember));
                }
            };
        group(classes[ci].multiplicity, budgetLeft, maxLoad);
      };
  perClass(0, opts.resourceBudget, 0);
  return best;
}

TEST(GraphAllocation, TwoMMSequentialPartitions) {
  // Two different-shape gemms (two singleton classes), grid of 2048. Pinned
  // side by side they run in a pipeline; timesharing the whole grid would pay
  // two program reloads per inference against sub-ms kernels, so
  // partitioning must win by orders of magnitude.
  SmallVector<ClassProfile> classes;
  classes.push_back({1, {point(1024, 0.9), point(2048, 0.5)}});
  classes.push_back({1, {point(1024, 1.1), point(2048, 0.6)}});
  AllocationOptions opts;
  opts.resourceBudget = 2048;
  opts.programReloadMs = 40.0;

  auto result = cinm::allocateGraph(classes, opts);
  ASSERT_TRUE(result);
  // Both pinned at 1024: bottleneck = 1.1. All alternatives (either op
  // timeshared) cost >= 40 ms.
  EXPECT_DOUBLE_EQ(result->bottleneckMs, 1.1);
  EXPECT_EQ(result->resourceUsed, 2048);
  for (const auto &cls : result->perClass) {
    ASSERT_EQ(cls.groups.size(), 1u);
    EXPECT_EQ(cls.groups[0].resource, 1024);
  }
  EXPECT_DOUBLE_EQ(bruteForce(classes, opts), result->bottleneckMs);
}

TEST(GraphAllocation, TimeshareWinsWhenReloadIsFree) {
  // Same instance, but reload/scatter cost zero and a tiny grid: pinning
  // both at 1024 is impossible (budget 1024), and with free reloads
  // timesharing the full grid beats pinning small.
  SmallVector<ClassProfile> classes;
  classes.push_back({1, {point(1024, 0.9)}});
  classes.push_back({1, {point(1024, 1.1)}});
  AllocationOptions opts;
  opts.resourceBudget = 1024;
  opts.programReloadMs = 0.0;

  auto result = cinm::allocateGraph(classes, opts);
  ASSERT_TRUE(result);
  EXPECT_DOUBLE_EQ(result->bottleneckMs, bruteForce(classes, opts));
  // One op pinned, the other timeshared (or both timeshared): bottleneck 1.1.
  EXPECT_DOUBLE_EQ(result->bottleneckMs, 1.1);
}

TEST(GraphAllocation, QKVSharesOneSetWhileWeightsFit) {
  // Three same-shape projections (one class of 3). Weights fit 3x on one
  // set: one group of three sharing a program, load 3L, leaving the rest of
  // the grid free -- unless splitting into more sets is better, which it is
  // when budget allows (3 sets of 1 run concurrently at load L each).
  SmallVector<ClassProfile> classes;
  classes.push_back(
      {3,
       {point(512, 2.0, /*static*/ 1000, /*dyn*/ 100, /*scatter*/ 5.0),
        point(1024, 1.0, 1000, 100, 5.0)}});
  AllocationOptions opts;
  opts.resourceBudget = 4096;
  opts.capacityBytes = 3500; // fits 3 x 1000 + 100

  auto result = cinm::allocateGraph(classes, opts);
  ASSERT_TRUE(result);
  // Budget allows three separate sets at 1024: bottleneck 1.0.
  EXPECT_DOUBLE_EQ(result->bottleneckMs, 1.0);
  EXPECT_DOUBLE_EQ(bruteForce(classes, opts), result->bottleneckMs);

  // Tight budget: only 1024 units in total. One set of 1024 shared by all
  // three (load 3.0) beats three sets of ~341 (not on the menu) and beats
  // timesharing (reload 40). The weights fit (3*1000 + 100 <= 3500).
  opts.resourceBudget = 1024;
  result = cinm::allocateGraph(classes, opts);
  ASSERT_TRUE(result);
  EXPECT_DOUBLE_EQ(result->bottleneckMs, 3.0);
  ASSERT_EQ(result->perClass[0].groups.size(), 1u);
  EXPECT_EQ(result->perClass[0].groups[0].size, 3u);
  EXPECT_DOUBLE_EQ(bruteForce(classes, opts), result->bottleneckMs);

  // Shrink MRAM so only two fit per set: the class must split 2+1, and the
  // budget only carries one 1024 set plus one 512 set.
  opts.capacityBytes = 2200; // 2*1000+100 fits, 3*1000+100 does not
  opts.resourceBudget = 1536;
  result = cinm::allocateGraph(classes, opts);
  ASSERT_TRUE(result);
  EXPECT_DOUBLE_EQ(bruteForce(classes, opts), result->bottleneckMs);
  unsigned total = 0;
  for (const auto &g : result->perClass[0].groups) {
    EXPECT_LE(g.size, 2u);
    total += g.size;
  }
  EXPECT_EQ(total, 3u);
}

TEST(GraphAllocation, InfeasibleWithoutTimeshareAndBudget) {
  SmallVector<ClassProfile> classes;
  classes.push_back({1, {point(1024, 1.0)}});
  classes.push_back({1, {point(1024, 1.0)}});
  AllocationOptions opts;
  opts.resourceBudget = 1024;
  opts.allowTimeshare = false;

  EXPECT_FALSE(cinm::allocateGraph(classes, opts));
}

TEST(GraphAllocation, MatchesBruteForceOnRandomInstances) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> nClasses(1, 3), mult(1, 3), nPoints(1, 3);
  std::uniform_real_distribution<double> cost(0.5, 20.0);
  std::uniform_int_distribution<int64_t> staticB(0, 1200);

  for (int iter = 0; iter < 200; ++iter) {
    SmallVector<ClassProfile> classes;
    int nc = nClasses(rng);
    for (int c = 0; c < nc; ++c) {
      ClassProfile cls;
      cls.multiplicity = mult(rng);
      int np = nPoints(rng);
      for (int p = 0; p < np; ++p)
        cls.points.push_back(
            point(256 * (p + 1), cost(rng), staticB(rng), 100, cost(rng) / 4));
      classes.push_back(std::move(cls));
    }
    AllocationOptions opts;
    opts.resourceBudget = 256 * (rng() % 8);
    opts.capacityBytes = (rng() % 2) ? 2500 : 0;
    opts.programReloadMs = (rng() % 2) ? 40.0 : 0.5;
    opts.allowTimeshare = rng() % 2;

    auto result = cinm::allocateGraph(classes, opts);
    double oracle = bruteForce(classes, opts);
    if (std::isinf(oracle)) {
      EXPECT_FALSE(result) << "iter " << iter;
      continue;
    }
    ASSERT_TRUE(result) << "iter " << iter;
    EXPECT_DOUBLE_EQ(result->bottleneckMs, oracle) << "iter " << iter;
    EXPECT_LE(result->resourceUsed, opts.resourceBudget) << "iter " << iter;
  }
}

} // namespace
