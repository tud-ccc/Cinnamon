#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintGecode.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"
#include "cinm-mlir/Utils/Permutation.h"

#include <algorithm>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <set>

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
  std::optional<int64_t> count = factorial(n);
  assert(count && "too many dimensions to enumerate their permutations");
  // The rank is one-based like every other parameter; the decoder is
  // zero-based. Which end converts is stated where the value is consumed.
  auto hi = static_cast<ParmValue>(*count);
  PermVar v(name, hi);
  dims_.push_back({v.name_, v.idx_, DimEntry::Permutation, 1, hi, {}, n});
  return v;
}

PermVar SpaceBuilder::permutation(llvm::StringRef name,
                                  llvm::ArrayRef<BoolExpr> active) {
  PermVar v = permutation(name, active.size());

  // How many items this configuration actually orders.
  llvm::SmallVector<IntExpr> flags;
  for (const BoolExpr &a : active)
    flags.push_back(asInt(a));
  IntExpr count = sum(flags);

  // The encoding is a rank over the active items, so a configuration ordering
  // k of them has k! orderings and every larger rank names nothing. Expanded
  // over the domain of `count` rather than tabulated against it: there are
  // `active.size() + 1` cases, and that is a handful.
  //
  // This is the constraint the caller does not write. Under a positional
  // encoding it would be a distinctness plus a rule putting the inactive items
  // last, posted from this same place -- which is the point of posting it here.
  IntExpr rank(v.node());
  for (ParmValue k = 0; static_cast<size_t>(k) <= active.size(); ++k) {
    std::optional<int64_t> orders = factorial(k);
    assert(orders && "too many items to enumerate their orderings");
    auto numOrders = static_cast<ParmValue>(*orders);
    require(implies(count == k, rank <= numOrders),
            (name + ": at most " + std::to_string(numOrders) +
             " ordering(s) when " + std::to_string(k) + " item(s) are active")
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
  multiples_.push_back({v.name_, src.name_});
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

// ===----------------------------------------------------------------------===//
// SpaceBuilder — constraint declaration
// ===----------------------------------------------------------------------===//

void SpaceBuilder::mustDivide(IntVar v, ParmValue n) {
  if (!ShapedType::isDynamic(n))
    findEntry(v).divisorFilters.push_back(n);
}

void SpaceBuilder::mustDivide(IntVar parent, IntVar child) {
  multiples_.push_back({parent.name_, child.name_});
}

void SpaceBuilder::require(VecConstraint pred, llvm::StringRef description) {
  predicates_.push_back({.description = description.str(),
                         .pred = std::move(pred),
                         .node = nullptr});
}

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
  // A divisibility test *at the top* of a require is unconditional, so it can
  // be reified like the one `/` asserts -- as a domain filter or a structural
  // relation, rather than a predicate that filters after the fact. Only here:
  // nested under a guard (or anywhere else) it stays a test, which is the whole
  // point of having it. addDivConstraint registers whatever it settles on,
  // including a predicate when the shapes allow nothing better.
  if (node->kind == ConstraintNode::Kind::Divides) {
    addDivConstraint(/*num=*/node->operands()[1], /*den=*/node->operands()[0]);
    return;
  }
  std::string desc =
      description.empty() ? describeNode(*node) : description.str();
  require(toVecConstraint(node), desc);
  // Keep the tree so buildInto can analyse it; require(VecConstraint) has
  // just pushed the entry.
  predicates_.back().node = node;
}

void SpaceBuilder::extractDivConstraints(const ConstraintNodePtr &node) {
  if (!node)
    return;
  // Not under a guard. What this function reifies is unconditional, so a `/`
  // inside an implication would impose its divisibility on the very
  // configurations the guard exists to exclude.
  //
  // Nothing is lost but pruning, and not even all of that. `a / b` is exact by
  // evaluation -- a lane whose division does not come out exact makes the
  // enclosing comparison false, see evalBoolNodeVec -- so a guarded division
  // still means what it says. And matchProductEquality cross-multiplies the
  // consequent anyway, so `implies(g, extent / block == 1)` still reaches the
  // enumerator as the gated equality `extent == block`. What goes is the static
  // domain filter, which is exactly the part that would have been wrong.
  if (node->kind == ConstraintNode::Kind::Implies)
    return;
  if (node->kind == ConstraintNode::Kind::Div)
    addDivConstraint(node->operands()[0], node->operands()[1]);
  for (const auto &child : node->operands())
    extractDivConstraints(child);
}

void SpaceBuilder::addDivConstraint(const ConstraintNodePtr &num,
                                    const ConstraintNodePtr &den) {
  using Kind = ConstraintNode::Kind;

  // const / var: the divisor can only ever take values dividing the constant,
  // so this is a static domain filter rather than a runtime check.
  if (num->kind == Kind::Const && den->kind == Kind::Var) {
    mustDivide(findVarByName(den->varName()), num->constValue());
    return;
  }
  // var / var: structural, recorded for planning to fold into a component.
  if (num->kind == Kind::Var && den->kind == Kind::Var) {
    mustDivide(findVarByName(den->varName()), findVarByName(num->varName()));
    return;
  }

  std::string desc = describeNode(*den) + " | " + describeNode(*num);
  if (den->kind == Kind::Mul) {
    // (B * C) | A  ⟹  B * C <= A as well; keeping the bound makes the
    // predicate reject the degenerate cases the divisibility test alone lets
    // through.
    require(VecConstraint(
                [num, den](const ConfigurationVector &c, arma::urowvec &valid) {
                  const ParmVector nv = evalNodeVec(*num, c);
                  const ParmVector dv = evalNodeVec(*den, c);
                  valid %= (dv <= nv) % vecDivides(dv, nv);
                }),
            desc);
    return;
  }
  require(VecConstraint(
              [num, den](const ConfigurationVector &c, arma::urowvec &valid) {
                valid %= vecDivides(evalNodeVec(*den, c), evalNodeVec(*num, c));
              }),
          desc);
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder::buildInto
// ===----------------------------------------------------------------------===//

void SpaceBuilder::buildInto(ConfigSpace &space) {
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
        llvm::dbgs() << "permutations of " << entry.permutationSize
                     << " (ranks 1.." << entry.hi << ")";
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

    *entry.idx = space.addDim(std::move(param));
  }

  for (const SearchParam &param : space.params)
    report->cartesian *= static_cast<double>(param.cardinality());

  // Phase 2: collect what the solver is to be given. Divisibility declared
  // through mustDivide never passes through a Div node, so it arrives here as
  // a relation rather than inside an expression; deduplicate it first, since
  // the same pair is commonly declared by several callers.
  std::sort(multiples_.begin(), multiples_.end());
  multiples_.erase(std::unique(multiples_.begin(), multiples_.end()),
                   multiples_.end());

  std::vector<constraints::DivisibilityRelation> divisibility;
  for (const auto &m : multiples_) {
    size_t parent = findVarByName(m.parent).idx();
    size_t child = findVarByName(m.child).idx();
    assert(parent < space.size() && child < space.size() &&
           "mustDivide names a dimension that was never declared");
    divisibility.push_back({parent, child});
    report->constraints.push_back({m.parent + " | " + m.child, "solved", true});
  }

  std::vector<ConstraintNodePtr> nodes;
  for (const auto &entry : predicates_)
    if (entry.node)
      nodes.push_back(entry.node);

  // Phase 3: solve. This is the whole of what used to be planning.
  constraints::SolveResult solved =
      constraints::solveSpace(space.params, nodes, divisibility);
  if (solved.failed())
    llvm::report_fatal_error(llvm::Twine("cinm search space: ") + solved.error);

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
                          << solved.solutions.size() << " configurations from "
                          << solved.nodes << " nodes, " << solved.failures
                          << " failures\n");

  report->solutions = solved.solutions.size();
  report->nodes = solved.nodes;
  report->failures = solved.failures;
  report->complete = solved.complete;
  space.setSolutions(std::move(solved.solutions));

  // Phase 4: register every predicate on the space.
  //
  // The ones the solver was given cannot reject anything any more, so this is
  // not how they are enforced. They are kept because isValid() and
  // debugIsValid() are the only way to ask *why* a hand-built configuration is
  // not in the space, and because a disagreement between a posted constraint
  // and the vectorized evaluator then shows up as configurations the space
  // contains but reports invalid -- which is a translation bug, and one worth
  // finding by construction rather than by wondering why a search is small.
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   predicates: " << predicates_.size()
                          << "\n");
  for (auto &entry : predicates_) {
    if (!entry.node)
      report->constraints.push_back({entry.description, "filter", false});
    else
      report->constraints.push_back({entry.description, "solved", true});
    std::visit(
        [&](auto &pred) {
          space.addConstraint(std::move(pred), entry.description);
        },
        entry.pred);
  }

  space.metadata = std::move(report);
}

} // namespace mlir::cinm
