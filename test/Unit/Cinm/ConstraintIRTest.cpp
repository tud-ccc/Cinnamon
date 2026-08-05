//===- ConstraintIRTest.cpp - The constraint DSL against brute force ------===//
//
// What the search space *contains* is not observable from a pass pipeline --
// only what the search eventually picks is -- so these check it directly:
// build a space with SpaceBuilder, enumerate everything the encoding offers,
// and compare against the same predicate applied by hand to the declared box.
//
// Two properties per case, and the second is the one that has caught bugs:
//
//  - **Correctness.** The accepted set must equal the brute-force set. Too few
//    is a constraint that over-rejects; too many is one that is not being
//    enforced at all.
//  - **Absorption.** For the cases marked so, the encoding must offer *only*
//    the valid configurations -- `totalSize() == |valid|`. That is the whole
//    point of docs/ConstraintAnalysisDesign.md's component enumeration, and it
//    is where a constraint the enumerator silently stops enforcing shows up: it
//    is absorbed (so the predicate is dropped) while no longer pruning, which
//    correctness alone would catch only if the predicate had been kept.
//
//===----------------------------------------------------------------------===//
//===- ConstraintIRTest.cpp ----------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <gtest/gtest.h>

#include <functional>
#include <set>
#include <vector>

using namespace mlir::cinm;

namespace {

using Point = std::vector<ParmValue>;
using PointSet = std::set<Point>;

PointSet bruteForce(const std::vector<std::vector<ParmValue>> &domains,
                    const std::function<bool(const Point &)> &ok) {
  PointSet out;
  std::vector<size_t> idx(domains.size(), 0);
  while (true) {
    Point pt;
    for (size_t i = 0; i < domains.size(); ++i)
      pt.push_back(domains[i][idx[i]]);
    if (ok(pt))
      out.insert(pt);

    size_t d = 0;
    for (; d < domains.size(); ++d) {
      if (++idx[d] < domains[d].size())
        break;
      idx[d] = 0;
    }
    if (d == domains.size())
      break;
  }
  return out;
}

PointSet accepted(const ConfigSpace &space) {
  PointSet out;
  Configuration conf(space.size());
  for (size_t i = 0; i < space.totalSize(); ++i) {
    space.at(i, conf);
    if (space.isValid(conf))
      out.insert(Point(conf.begin(), conf.end()));
  }
  return out;
}

enum Absorption { Absorbed, Filtered };

std::vector<ParmValue> range(ParmValue lo, ParmValue hi) {
  std::vector<ParmValue> out;
  for (ParmValue v = lo; v <= hi; ++v)
    out.push_back(v);
  return out;
}

std::vector<ParmValue> divisorsOfN(ParmValue n) {
  std::vector<ParmValue> out;
  for (ParmValue v = 1; v <= n; ++v)
    if (n % v == 0)
      out.push_back(v);
  return out;
}

void check(const ConfigSpace &space, const PointSet &expected,
           Absorption absorption) {
  PointSet got = accepted(space);

  EXPECT_EQ(got, expected);

  if (absorption == Absorbed)
    EXPECT_EQ(space.totalSize(), expected.size());

  Configuration conf(space.size());
  for (size_t i = 0; i < space.totalSize(); ++i) {
    space.at(i, conf);
    EXPECT_EQ(space.indexOf(conf), i)
        << "Round-trip failure at flat index " << i;
  }
}

} // namespace

TEST(ConstraintIRTest, GatedEquality) {
  SpaceBuilder b;
  auto fuse = b.intRange("fuse", 0, 2);
  auto a = b.divisorsOf("a", 64);
  auto c = b.divisorsOf("c", 64);
  b.require(implies(fuse >= 1, a == c));

  ConfigSpace space;
  b.buildInto(space);

  check(space,
        bruteForce({range(0, 2), divisorsOfN(64), divisorsOfN(64)},
                   [](const Point &p) { return p[0] < 1 || p[1] == p[2]; }),
        Absorbed);
}

TEST(ConstraintIRTest, NestedGates) {
  SpaceBuilder b;
  auto fuse = b.intRange("fuse", 0, 2);
  auto a = b.divisorsOf("a", 64);
  auto c = b.divisorsOf("c", 64);
  auto aw = b.divisorsOf("aw", a);
  auto cw = b.divisorsOf("cw", c);

  b.require(implies(fuse >= 1, a == c));
  b.require(implies(fuse >= 2, aw == cw));

  ConfigSpace space;
  b.buildInto(space);

  check(space,
        bruteForce({range(0, 2), divisorsOfN(64), divisorsOfN(64),
                    divisorsOfN(64), divisorsOfN(64)},
                   [](const Point &p) {
                     if (p[1] % p[3] || p[2] % p[4])
                       return false;
                     if (p[0] >= 1 && p[1] != p[2])
                       return false;
                     return !(p[0] >= 2 && p[3] != p[4]);
                   }),
        Absorbed);
}

TEST(ConstraintIRTest, GateWithProductEquality) {
  SpaceBuilder b;
  auto fuse = b.intRange("fuse", 0, 1);
  auto a = b.divisorsOf("a", 64);
  auto c = b.divisorsOf("c", 64);
  auto n = b.intRange("n", 1, 64);
  b.require(a * n == 64);
  b.require(implies(fuse >= 1, a == c));
  ConfigSpace space;
  b.buildInto(space);

  check(
      space,
      bruteForce({range(0, 1), divisorsOfN(64), divisorsOfN(64), range(1, 64)},
                 [](const Point &p) {
                   if (p[1] * p[3] != 64)
                     return false;
                   return p[0] < 1 || p[1] == p[2];
                 }),
      Absorbed);
}

TEST(ConstraintIRTest, GateOnInequality) {
  SpaceBuilder b;
  auto fuse = b.intRange("fuse", 0, 1);
  auto a = b.divisorsOf("a", 64);
  auto c = b.divisorsOf("c", 64);
  b.require(implies(fuse >= 1, a * c <= 64));
  ConfigSpace space;
  b.buildInto(space);

  // An inequality consequent is not a product equality, and nothing else here
  // links the variables, so there is no component to absorb it into. Correct,
  // just filtered rather than encoded.
  check(
      space,
      bruteForce({range(0, 1), divisorsOfN(64), divisorsOfN(64)},
                 [](const Point &p) { return p[0] < 1 || p[1] * p[2] <= 64; }),
      Filtered);
}

//===----------------------------------------------------------------------===//
// Division is exact division
//===----------------------------------------------------------------------===//

TEST(ConstraintIRTest, ExactDivision) {
  SpaceBuilder b;
  auto v = b.intRange("b", 1, 1024);
  b.require(IntExpr(1024) / v == 1);
  ConfigSpace space;
  b.buildInto(space);

  // Only b == 1024. Under a truncating quotient this would also accept every
  // b in (512, 1024], none of which divides 1024.
  check(space,
        bruteForce({range(1, 1024)},
                   [](const Point &p) {
                     return 1024 % p[0] == 0 && 1024 / p[0] == 1;
                   }),
        Filtered);
}

TEST(ConstraintIRTest, ExactDivisionUnderGuard) {
  SpaceBuilder b;
  auto fuse = b.intRange("fuse", 0, 1);
  auto v = b.intRange("b", 1, 1024);
  b.require(implies(fuse >= 1, IntExpr(1024) / v == 1));
  ConfigSpace space;
  b.buildInto(space);

  // No divisibility side condition is extracted from under a guard, so the
  // evaluation is the only thing enforcing exactness here. It is still
  // absorbed: matchProductEquality cross-multiplies the consequent to
  // `1024 == b`, which the enumerator solves.
  check(space,
        bruteForce({range(0, 1), range(1, 1024)},
                   [](const Point &p) {
                     if (p[0] < 1)
                       return true;
                     return 1024 % p[1] == 0 && 1024 / p[1] == 1;
                   }),
        Absorbed);
}

TEST(ConstraintIRTest, InexactDivisionInAntecedent) {
  SpaceBuilder b;
  auto v = b.intRange("b", 1, 16);
  auto w = b.intRange("w", 1, 4);
  b.require(implies(IntExpr(16) / v == 1, w == 4));
  ConfigSpace space;
  b.buildInto(space);

  // An inexact division in the *antecedent* falsifies the antecedent, which
  // satisfies the implication. Masking at the root instead of per comparison
  // would reject these lanes -- the opposite verdict.
  check(space,
        bruteForce({range(1, 16), range(1, 4)},
                   [](const Point &p) {
                     const bool ante = 16 % p[0] == 0 && 16 / p[0] == 1;
                     return !ante || p[1] == 4;
                   }),
        Absorbed);
}

//===----------------------------------------------------------------------===//
// divides() tests where `/` asserts
//===----------------------------------------------------------------------===//

TEST(ConstraintIRTest, DividesIsStructuralAtTheTop) {
  SpaceBuilder b;
  auto a = b.divisorsOf("a", 64);
  auto c = b.intRange("c", 1, 64);
  b.require(divides(c, a));
  ConfigSpace space;
  b.buildInto(space);

  // Unconditional, so it is reified exactly as `a / c` would have been: a
  // structural relation folded into the encoding, not a filter.
  check(space,
        bruteForce({divisorsOfN(64), range(1, 64)},
                   [](const Point &p) { return p[0] % p[1] == 0; }),
        Absorbed);
}

TEST(ConstraintIRTest, DividesUnderGuardOnlyTests) {
  SpaceBuilder b;
  auto a = b.divisorsOf("a", 16);
  auto c = b.intRange("c", 1, 16);
  auto w = b.intRange("w", 1, 4);
  b.require(implies(divides(c, a), w == 4));
  ConfigSpace space;
  b.buildInto(space);

  // The point of the operator: `c` is *not* forced to divide `a`. Every pair
  // survives; only the ones where it happens to divide constrain `w`.
  check(
      space,
      bruteForce({divisorsOfN(16), range(1, 16), range(1, 4)},
                 [](const Point &p) { return p[0] % p[1] != 0 || p[2] == 4; }),
      Filtered);
}
