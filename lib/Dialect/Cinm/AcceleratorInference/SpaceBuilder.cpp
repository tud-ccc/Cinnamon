#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintGecode.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"
#include "cinm-mlir/Utils/Permutation.h"

#include <algorithm>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Parallel.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

#define DEBUG_TYPE "cinm-inference"

using namespace mlir::cinm::constraints;
namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Planning report
// ===----------------------------------------------------------------------===//

/// What building the space decided, in the form the space carries away with it.
///
/// A space that reports 3M points is the result of a solve, and the result
/// alone does not say what went into it: which constraints the solver was
/// given, which ones it could not be given, and how much work it took to
/// enumerate what came out. Recording that is the difference between an
/// experiment's artefacts describing a space and merely sizing it.
struct SpaceBuilder::PlanMetadata final : SpaceMetadata {
  /// Where a constraint ended up. What matters is whether it is enforced by
  /// construction or still costs a test per configuration.
  struct Constraint {
    std::string description;
    /// "solved"        posted to the solver, so no configuration violates it
    /// "static-filter" applied to a domain at declaration, same effect
    /// "filter"        an opaque lambda, tested per configuration
    std::string disposition;
    /// False for a predicate registered as an opaque lambda, which has no IR
    /// to post and therefore can only ever filter.
    bool analysable = true;
    /// Number of solutions this constraint filtered out.
    /// This is -1 for solved/static-filter constraints.
    /// Sensitive to constraint declaration order.
    int numRemoved = -1;
  };

  std::vector<Constraint> constraints;

  /// Product of the declared domain sizes, against the number of
  /// configurations the solver returned. Their ratio is the density -- which
  /// is now a property of the space rather than of a merging policy, since
  /// there is no partition left to choose.
  double cartesian = 1;
  size_t solutions = 0;

  /// What the search cost, and whether it finished. An incomplete solve means
  /// the space is a prefix of the feasible set and nothing may be concluded
  /// from what is missing from it.
  uint64_t nodes = 0;
  uint64_t failures = 0;
  bool complete = true;

  void printJSONMembers(std::ostream &os) const override;
};

namespace {
void printJSONString(std::ostream &os, llvm::StringRef s) {
  os << '"';
  for (char c : s) {
    if (c == '"' || c == '\\')
      os << '\\';
    os << c;
  }
  os << '"';
}
} // namespace

void SpaceBuilder::PlanMetadata::printJSONMembers(std::ostream &os) const {
  os << "  \"solver\": {\"cartesian\": " << cartesian
     << ", \"solutions\": " << solutions << ", \"nodes\": " << nodes
     << ", \"failures\": " << failures
     << ", \"complete\": " << (complete ? "true" : "false") << "},\n";

  os << "  \"constraints\": [\n";
  for (size_t i = 0; i < constraints.size(); ++i) {
    const Constraint &c = constraints[i];
    os << "    {\"constraint\": ";
    printJSONString(os, c.description);
    os << ", \"disposition\": ";
    printJSONString(os, c.disposition);
    if (!c.analysable)
      os << ", \"analysable\": false";
    if (c.numRemoved != -1)
      os << ", \"numRemoved\": " << c.numRemoved;
    os << "}" << (i + 1 < constraints.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder — dimension declaration
// ===----------------------------------------------------------------------===//

IntVar SpaceBuilder::intRange(llvm::StringRef name, ParmValue lo,
                              ParmValue hi) {
  assert(lo > 0 && "search parameters are positive integers");
  IntVar v(name, hi);
  dims_.push_back({v.name_, v.idx_, DimEntry::IntRange, lo, hi, {}});
  return v;
}

IntVar SpaceBuilder::pow2Range(llvm::StringRef name, ParmValue expLo,
                               ParmValue expHi) {
  assert(expLo >= 0 && "search parameters are positive integers");
  IntVar v(name, ParmValue{1} << expHi);
  dims_.push_back({v.name_, v.idx_, DimEntry::Pow2, expLo, expHi, {}});
  return v;
}

PermVar SpaceBuilder::permutation(llvm::StringRef name, unsigned n) {
  // Every one of the n dimensions holds a place in [1, n]; what stops them
  // being n independent numbers is the distinctness the solver posts for a
  // parameter of this kind. See ParmKind<Permutation>.
  auto hi = static_cast<ParmValue>(n);
  PermVar v(name, hi);
  dims_.push_back({v.name_, v.idx_, DimEntry::Permutation, 1, hi, {}, n});
  return v;
}

PermVar SpaceBuilder::permutation(llvm::StringRef name,
                                  llvm::ArrayRef<BoolExpr> active) {
  PermVar v = permutation(name, active.size());

  // How many items this configuration actually orders. `asInt(a) == 0` is the
  // negation of `a`, which is why no Not node is needed: there is no other
  // caller for one.
  llvm::SmallVector<IntExpr> flags;
  for (const BoolExpr &a : active)
    flags.push_back(asInt(a));
  IntExpr count = sum(flags);

  // The active items take places 1..count and the inactive ones the rest.
  // With distinctness that is already exactly count! assignments *up to* where
  // the inactive items go, which is what the third rule then pins: they sit in
  // index order, so a configuration does not appear once per rearrangement of
  // items that take no place at all.
  //
  // These are the constraints the caller does not write. They are about the
  // encoding, and the encoding is not the caller's business.
  for (size_t i = 0; i < active.size(); ++i) {
    require(implies(flags[i] == 1, v.axis(i) <= count),
            (name + ": item " + std::to_string(i) +
             " takes one of the first (number active) places when it is active")
                .str());
    require(implies(flags[i] == 0, v.axis(i) > count),
            (name + ": item " + std::to_string(i) +
             " takes a place past the active ones when it is inactive")
                .str());
    for (size_t j = i + 1; j < active.size(); ++j)
      require(
          implies(flags[i] == 0, implies(flags[j] == 0, v.axis(i) < v.axis(j))),
          (name + ": items " + std::to_string(i) + " and " + std::to_string(j) +
           " keep index order while both are inactive (symmetry)")
              .str());
  }
  return v;
}

IntVar SpaceBuilder::divisorsOf(llvm::StringRef name, ParmValue n) {
  IntVar v(name, n);
  dims_.push_back({v.name_, v.idx_, DimEntry::DivisorsOfConst, 1, n, {n}});
  return v;
}

IntVar SpaceBuilder::divisorsOf(llvm::StringRef name, IntVar src) {
  IntVar v(name, src.maxVal());
  dims_.push_back({v.name_, v.idx_, DimEntry::IntRange, 1, src.maxVal(), {}});
  require(divides(v, src));
  return v;
}

SpaceBuilder::DimEntry &SpaceBuilder::findEntry(const IntVar &v) {
  for (auto &e : dims_)
    if (e.idx == v.idx_)
      return e;
  llvm_unreachable("SpaceVar not found in SpaceBuilder");
}

IntVar SpaceBuilder::findVarByName(llvm::StringRef name) const {
  for (const auto &e : dims_)
    if (e.name == name.str())
      return IntVar(e.name, e.hi, e.idx);
  llvm_unreachable("dim name not found in SpaceBuilder");
}

int SpaceBuilder::dimIndexByName(llvm::StringRef name) const {
  for (int i = 0; i < (int)dims_.size(); ++i)
    if (dims_[i].name == name.str())
      return i;
  return -1;
}

bool SpaceBuilder::pin(llvm::StringRef name, ParmValue value) {
  int idx = dimIndexByName(name);
  if (idx < 0 || dims_[idx].kind == DimEntry::Permutation)
    return false;
  require(findVarByName(name) == value, (name + " pinned").str());
  return true;
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder — constraint declaration
// ===----------------------------------------------------------------------===//
void SpaceBuilder::require(Constraint pred, llvm::StringRef description) {
  predicates_.push_back({.description = description.str(),
                         .pred = std::move(pred),
                         .node = nullptr});
}

void SpaceBuilder::require(const ConstraintNodePtr &node,
                           llvm::StringRef description) {
  extractDivConstraints(node);
  // A bare arithmetic expression contributes only its divisibility conditions
  // (that is the `require(a / b)` spelling); only a boolean node is a
  // predicate.
  if (!ConstraintNode::isBoolKind(node->kind))
    return;
  // The tree is the constraint. It carries no lambda: buildInto posts it to the
  // solver, which is the only thing that ever enforces it.
  predicates_.push_back({.description = description.empty()
                                            ? describeNode(*node)
                                            : description.str(),
                         .pred = nullptr,
                         .node = node});
}

void SpaceBuilder::extractDivConstraints(const ConstraintNodePtr &node) {
  if (!node)
    return;
  // Not under a guard. What this function reifies is unconditional, so a `/`
  // inside an implication would impose its divisibility on the very
  // configurations the guard exists to exclude.
  if (node->kind == ConstraintNode::Kind::Implies)
    return;
  if (node->kind == ConstraintNode::Kind::Div)
    require(divides(node->operands()[1], node->operands()[0]));
  for (const auto &child : node->operands())
    extractDivConstraints(child);
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder::buildInto
// ===----------------------------------------------------------------------===//

void SpaceBuilder::buildInto(ConfigSpace &space, unsigned nWorkers) {
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space] building config space:\n");

  auto report = std::make_unique<PlanMetadata>();

  // Phase 1: build each SearchParam, deduplicate + apply static filters,
  // addDim.
  for (auto &entry : dims_) {
    SearchParam param = [&]() -> SearchParam {
      switch (entry.kind) {
      case DimEntry::IntRange:
      case DimEntry::DivisorsOfConst:
        return makeRange(entry.name, entry.lo, entry.hi);
      case DimEntry::Pow2:
        return makePow2Range(entry.name, entry.lo, entry.hi);
      case DimEntry::Permutation:
        return makePermutation(entry.name, entry.permutationSize);
      }
      llvm_unreachable("unknown DimKind");
    }();
    // How the domain is stored and how its values are meant are independent;
    // the declaration carries the second on the handle.

    std::sort(entry.divisorFilters.begin(), entry.divisorFilters.end());
    entry.divisorFilters.erase(
        std::unique(entry.divisorFilters.begin(), entry.divisorFilters.end()),
        entry.divisorFilters.end());
    for (ParmValue n : entry.divisorFilters) {
      param.keepDivisorsOf(n);
      report->constraints.push_back(
          {std::to_string(n) + " % " + entry.name + " == 0", "static-filter",
           true});
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[cinm-space]   dim '" << entry.name << "': ";
      switch (entry.kind) {
      case DimEntry::IntRange:
        llvm::dbgs() << "int[" << entry.lo << ".." << entry.hi << "]";
        break;
      case DimEntry::DivisorsOfConst:
        llvm::dbgs() << "divisors[" << entry.lo << ".." << entry.hi << "]";
        break;
      case DimEntry::Pow2:
        llvm::dbgs() << "pow2[2^" << entry.lo << "..2^" << entry.hi << "]";
        break;
      case DimEntry::Permutation:
        llvm::dbgs() << "orderings of " << entry.permutationSize
                     << " (one place in 1.." << entry.hi << " per item)";
        break;
      }
      if (!entry.divisorFilters.empty()) {
        llvm::dbgs() << "  filters=divisorsOf{";
        for (size_t i = 0; i < entry.divisorFilters.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
          llvm::dbgs() << entry.divisorFilters[i];
        }
        llvm::dbgs() << "}";
      }
      llvm::dbgs() << "\n";
    });

    *entry.idx = space.addParam(std::move(param));
  }

  // Per *parameter*, and its values rather than its dimensions: the density
  // below is meant to say how much the constraints written here cut the space,
  // so it must not also take credit for distinctness. That constraint is the
  // encoding's own bookkeeping -- it is what makes n dimensions an ordering in
  // the first place -- and counting the n^n encodings it rules out would
  // inflate every density by n^(n-1) per ordering.
  for (const SearchParam &param : space.params)
    report->cartesian *= param.numValues();

  // Phase 2: collect what the solver is to be given.

  std::vector<ConstraintNodePtr> nodes;
  for (const auto &entry : predicates_)
    if (entry.node)
      nodes.push_back(entry.node);

  // Phase 3: solve. This is the whole of what used to be planning.
  constraints::SolveOptions opts;
  opts.threads = nWorkers;

  auto t0 = std::chrono::steady_clock::now();
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space] Solving " << nodes.size()
                          << " constraints on " << opts.threads
                          << " threads\n");
  constraints::SolveResult solved =
      constraints::solveSpace(space.params, nodes, opts);
  if (solved.failed())
    llvm::report_fatal_error(llvm::Twine("cinm search space: ") + solved.error);
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space] - Done solving constraints in "
                          << elapsed.count() << " ms\n");

  // A truncated space is not a smaller space, it is a different one: the
  // configurations missing from it are missing because the search ran out of
  // budget where it happened to be, which is not a property anything
  // downstream can account for. Everything a search then reports would be
  // about a space nobody chose, so this stops rather than continues.
  if (!solved.complete)
    llvm::report_fatal_error(
        llvm::Twine("cinm search space: the solver hit its budget after ") +
        llvm::Twine(solved.solutions.size()) + " configurations and " +
        llvm::Twine(solved.nodes) +
        " nodes, so the space is a prefix of the feasible set rather than the "
        "feasible set. Constrain the parameters further, or raise the limits "
        "in constraints::SolveOptions if the space really is this large.");

  LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   solved: "
                          << solved.solutions.size()
                          << " feasible configurations (" << solved.nodes
                          << " search nodes, " << solved.failures
                          << " failed nodes)\n");

  report->solutions = solved.solutions.size();
  report->nodes = solved.nodes;
  report->failures = solved.failures;
  report->complete = solved.complete;

  // Phase 4: apply the predicates the solver was not given. Doing it here and
  // not on the space is what makes the space's contents exactly its feasible
  // set, so nothing downstream ever has to re-check a configuration.
  for (auto &entry : predicates_) {
    if (!entry.node) {
      auto &solutions = solved.solutions;
      // this wasn't part of the solve (it's an opaque predicate)
      auto newEnd = std::remove_if(solutions.begin(), solutions.end(),
                                   [&](Configuration conf) -> bool {
                                     ConfWrapper wrapper(space, conf);
                                     return !entry.pred(wrapper);
                                   });
      int numRemoved = solutions.end() - newEnd;
      solutions.erase(newEnd, solutions.end());

      report->constraints.push_back(
          {entry.description, "filter", false, numRemoved});
    } else {
      report->constraints.push_back({entry.description, "solved", true});
    }
  }

  // The feasible set is a subset of the values the parameters range over, so
  // this holds by construction -- and it is worth asserting because the two
  // sides are counted by completely different code. It is what caught the
  // Cartesian size being a product over *dimensions*: distinctness made the
  // encoding box bigger than the values, and a density came out above 1.
  assert(static_cast<double>(solved.solutions.size()) <= report->cartesian &&
         "more feasible configurations than the parameters have values");

  space.setSolutions(std::move(solved.solutions));
  space.metadata = std::move(report);
}

} // namespace mlir::cinm
