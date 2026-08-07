#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <string>
#include <vector>

namespace mlir::cinm::constraints {

// ===----------------------------------------------------------------------===//
// Solving
// ===----------------------------------------------------------------------===//
//
// The constraint IR handed to a finite-domain solver, which enumerates the
// feasible set directly instead of the framework enumerating a superset and
// filtering it.
//
// This is the whole of what used to be planning: there is no partition to
// choose, no component to enumerate, no relation to classify as static,
// structural or dynamic, and no budget deciding between them. A constraint is
// either expressible in the IR -- in which case it is posted and propagated --
// or it is an opaque C++ lambda, which the space still filters with afterwards
// (see ConfigSpace::constraints).
//
// What the solver is given is exactly what the vectorized evaluator in
// ConstraintIR.cpp computes, node for node. In particular `a / b` denotes the
// quotient *and* asserts that b divides a, so a comparison containing an
// inexact division is false rather than a comparison against a truncated
// quotient. That side condition is conjoined at the comparison that contains
// it, never at the root: an inexact division in an implication's antecedent
// has to falsify the antecedent -- which *satisfies* the implication -- and
// posting it globally would reject the configuration instead.

struct SolveOptions {
  /// Abandon the search after this many nodes. Zero disables the limit.
  ///
  /// A space too large to enumerate is a modelling mistake to report, not a
  /// case to degrade gracefully for: there is no coarser encoding to fall back
  /// on now that the feasible set *is* the representation.
  uint64_t nodeLimit = 200'000'000;
  /// Abandon the search after this many solutions. Zero disables the limit.
  size_t solutionLimit = 50'000'000;
  /// Search worker threads. More than one makes the order in which solutions
  /// are found nondeterministic, which costs nothing here because the result
  /// is sorted before it is returned.
  unsigned threads = 1;
};

struct SolveResult {
  /// Every configuration satisfying the posted constraints, sorted
  /// lexicographically.
  ///
  /// Sorting is not incidental. It makes a configuration's index a property of
  /// the space rather than of the search that found it, so changing the
  /// branching heuristic -- or the thread count -- renumbers nothing.
  std::vector<Configuration> solutions;

  uint64_t nodes = 0;
  uint64_t failures = 0;

  /// False when a budget stopped the search. `solutions` is then a prefix of
  /// the feasible set, and nothing may be concluded from what is absent from
  /// it.
  bool complete = true;

  /// Set when the model could not be built at all -- an overflowing product is
  /// the case that actually happens, since a bound the solver has to represent
  /// can exceed what an int variable holds even though no *solution* does.
  std::string error;

  bool failed() const { return !error.empty(); }
};

/// Enumerate every configuration over `params` that satisfies every constraint.
///
/// Opaque predicates are not represented here at all -- they have no IR to post
/// -- and remain the space's business to filter with.
SolveResult solveSpace(llvm::ArrayRef<SearchParam> params,
                       llvm::ArrayRef<ConstraintNodePtr> constraints,
                       const SolveOptions &opts = {});

} // namespace mlir::cinm::constraints
