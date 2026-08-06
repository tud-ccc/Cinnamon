#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"
#include "cinm-mlir/Utils/Permutation.h"

#include <algorithm>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <set>

#define DEBUG_TYPE "cinm-inference"

using namespace mlir::cinm::constraints;
namespace mlir::cinm {

namespace detail {
void assertArithmeticOperands(const ConstraintNodePtr &lhs,
                              const ConstraintNodePtr &rhs,
                              llvm::StringRef op) {
  for (const ConstraintNodePtr &side : {lhs, rhs}) {
    auto found = findNonArithmeticVar(*side);
    if (!found)
      continue;
    llvm::report_fatal_error(llvm::Twine("search parameter '") + found->second +
                             "' is a " + paramKindName(found->first) +
                             ", whose values are a numbering rather than a "
                             "quantity; it cannot appear under '" +
                             op +
                             "' (only '==' and '!=' are meaningful on it)");
  }
}
} // namespace detail

// ===----------------------------------------------------------------------===//
// Planning report
// ===----------------------------------------------------------------------===//

/// What planning decided, in the form the space carries away with it.
///
/// A space that reports 3M points is the product of a series of decisions --
/// which parameters were enumerated jointly, which constraints that let it drop
/// as predicates, which grouping was attempted and abandoned -- and none of
/// them are recoverable from the result. Recording them is the difference
/// between an experiment's artefacts describing a space and merely sizing it.
struct SpaceBuilder::PlanMetadata final : SpaceMetadata {
  /// A set of parameters enumerated jointly.
  struct Component {
    std::vector<std::string> params;
    /// Tuples enumerated, against the product of the parameters' domains.
    /// Their ratio is what the component bought.
    size_t solutions = 0;
    size_t cartesian = 1;
    /// Set when the enumeration hit a budget: the component absorbed nothing
    /// and every relation in it fell back to the pairwise and predicate paths.
    /// `solutions` is meaningless then.
    bool abandoned = false;
  };

  /// Where a constraint ended up. What matters is whether it still costs a
  /// test per configuration, and if not, what absorbed it.
  struct Constraint {
    std::string description;
    /// "folded", "filter", "static-filter", "structural-pair".
    std::string disposition;
    /// Index into `components` for a folded constraint, -1 otherwise.
    int component = -1;
    /// False for a predicate registered as an opaque lambda, which planning
    /// cannot read and therefore can never fold.
    bool analysable = true;
  };

  std::vector<Component> components;
  std::vector<Constraint> constraints;
  /// Which component absorbed a given predicate tree, so that buildInto can
  /// pair it with the description the predicate was registered under.
  llvm::DenseMap<const constraints::ConstraintNode *, int> foldedInto;

  void printJSONMembers(std::ostream &os) const override;

  int componentOf(llvm::ArrayRef<size_t> dims, const ConfigSpace &space);
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

int SpaceBuilder::PlanMetadata::componentOf(llvm::ArrayRef<size_t> dims,
                                            const ConfigSpace &space) {
  Component comp;
  for (size_t d : dims) {
    comp.params.push_back(space[d].name);
    comp.cartesian *= space[d].cardinality();
  }
  components.push_back(std::move(comp));
  return static_cast<int>(components.size()) - 1;
}

void SpaceBuilder::PlanMetadata::printJSONMembers(std::ostream &os) const {
  os << "  \"components\": [\n";
  for (size_t i = 0; i < components.size(); ++i) {
    const Component &c = components[i];
    os << "    {\"params\": [";
    for (size_t j = 0; j < c.params.size(); ++j) {
      if (j)
        os << ", ";
      printJSONString(os, c.params[j]);
    }
    os << "], \"cartesian\": " << c.cartesian;
    if (c.abandoned)
      os << ", \"abandoned\": true";
    else
      os << ", \"solutions\": " << c.solutions;
    os << "}" << (i + 1 < components.size() ? "," : "") << "\n";
  }
  os << "  ],\n";

  os << "  \"constraints\": [\n";
  for (size_t i = 0; i < constraints.size(); ++i) {
    const Constraint &c = constraints[i];
    os << "    {\"constraint\": ";
    printJSONString(os, c.description);
    os << ", \"disposition\": ";
    printJSONString(os, c.disposition);
    if (c.component >= 0)
      os << ", \"component\": " << c.component;
    if (!c.analysable)
      os << ", \"analysable\": false";
    os << "}" << (i + 1 < constraints.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder — dimension declaration
// ===----------------------------------------------------------------------===//

SpaceVar SpaceBuilder::intRange(llvm::StringRef name, ParmValue lo,
                                ParmValue hi) {
  assert(lo > 0 && "search parameters are positive integers");
  SpaceVar v(name, hi);
  dims_.push_back({v, DimEntry::IntRange, lo, hi, {}});
  return v;
}

SpaceVar SpaceBuilder::pow2Range(llvm::StringRef name, ParmValue expLo,
                                 ParmValue expHi) {
  assert(expLo >= 0 && "search parameters are positive integers");
  SpaceVar v(name, ParmValue{1} << expHi);
  dims_.push_back({v, DimEntry::Pow2, expLo, expHi, {}});
  return v;
}

SpaceVar SpaceBuilder::permutation(llvm::StringRef name, unsigned n) {
  std::optional<int64_t> count = factorial(n);
  assert(count && "too many dimensions to enumerate their permutations");
  // The rank is one-based like every other parameter; the decoder is
  // zero-based. Which end converts is stated where the value is consumed.
  auto hi = static_cast<ParmValue>(*count);
  SpaceVar v(name, hi, ParamKind::Permutation);
  dims_.push_back({v, DimEntry::Permutation, 1, hi, {}, n});
  return v;
}

SpaceVar SpaceBuilder::divisorsOf(llvm::StringRef name, ParmValue n) {
  SpaceVar v(name, n);
  dims_.push_back({v, DimEntry::DivisorsOfConst, 1, n, {n}});
  return v;
}

SpaceVar SpaceBuilder::divisorsOf(llvm::StringRef name, SpaceVar src) {
  SpaceVar v(name, src.maxVal());
  dims_.push_back({v, DimEntry::IntRange, 1, src.maxVal(), {}});
  multiples_.push_back({v.name_, src.name_});
  return v;
}

SpaceBuilder::DimEntry &SpaceBuilder::findEntry(const SpaceVar &v) {
  for (auto &e : dims_)
    if (e.var.idx_ == v.idx_)
      return e;
  llvm_unreachable("SpaceVar not found in SpaceBuilder");
}

SpaceVar SpaceBuilder::findVarByName(llvm::StringRef name) const {
  for (const auto &e : dims_)
    if (e.var.name_ == name.str())
      return e.var;
  llvm_unreachable("dim name not found in SpaceBuilder");
}

int SpaceBuilder::dimIndexByName(llvm::StringRef name) const {
  for (int i = 0; i < (int)dims_.size(); ++i)
    if (dims_[i].var.name_ == name.str())
      return i;
  return -1;
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder — constraint declaration
// ===----------------------------------------------------------------------===//

void SpaceBuilder::mustDivide(SpaceVar v, ParmValue n) {
  if (!ShapedType::isDynamic(n))
    findEntry(v).divisorFilters.push_back(n);
}

void SpaceBuilder::mustDivide(SpaceVar parent, SpaceVar child) {
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

void SpaceBuilder::reportConstraintAnalysis(const ConfigSpace &space) const {
  std::vector<std::string> names;
  for (const auto &p : space.params)
    names.push_back(p.name);

  // Joint number of values the variables of a monomial can take together --
  // the size of the sub-space the encoding would have to enumerate for it.
  auto jointCardinality = [&](const Monomial &m) -> size_t {
    std::set<size_t> distinct(m.vars.begin(), m.vars.end());
    size_t n = 1;
    for (size_t v : distinct)
      n *= space[v].cardinality();
    return n;
  };

  for (const auto &entry : predicates_) {
    if (!entry.node)
      continue; // opaque predicate, nothing to analyse
    if (entry.node->kind == ConstraintNode::Kind::Implies) {
      // The guard is what decides whether the consequent is in force, so
      // report the consequent's own shape: a product equality there is one the
      // enumerator can solve, once the guard is settled.
      const bool solvable =
          matchProductEquality(*entry.node->operands()[1]).has_value();
      llvm::dbgs() << "[cinm-analysis]   gated by "
                   << describeNode(*entry.node->operands()[0]) << ": "
                   << describeNode(*entry.node->operands()[1]) << " ("
                   << (solvable ? "equality once the guard is settled"
                                : "not an equality")
                   << ")\n";
      continue;
    }
    auto eq = matchProductEquality(*entry.node);
    if (!eq) {
      llvm::dbgs() << "[cinm-analysis]   not an equality: " << entry.description
                   << "\n";
      continue;
    }

    const size_t lhsCard = jointCardinality(eq->lhs);
    const size_t rhsCard = jointCardinality(eq->rhs);
    llvm::dbgs() << "[cinm-analysis]   equality: "
                 << describeMonomial(eq->lhs, names)
                 << " == " << describeMonomial(eq->rhs, names) << "\n";
    llvm::dbgs() << "[cinm-analysis]     joint cardinality: lhs=" << lhsCard
                 << " rhs=" << rhsCard << "\n";

    // Overlapping variables mean neither side determines the other.
    std::set<size_t> lhsVars(eq->lhs.vars.begin(), eq->lhs.vars.end());
    bool overlap = llvm::any_of(
        eq->rhs.vars, [&](size_t v) { return lhsVars.count(v) != 0; });
    if (overlap) {
      llvm::dbgs() << "[cinm-analysis]     -> not solvable: a variable occurs "
                      "on both sides\n";
      continue;
    }
    if (eq->lhs.vars.empty() && eq->rhs.vars.empty()) {
      llvm::dbgs() << "[cinm-analysis]     -> constant equality\n";
      continue;
    }

    // Enumerate the smaller side, solve for the larger. Reporting only: what
    // the enumerator actually does is pick a variable against the current
    // partial assignment, see ComponentEnumerator::selectNext.
    const bool solveRhs = rhsCard >= lhsCard;
    const Monomial &key = solveRhs ? eq->lhs : eq->rhs;
    const Monomial &solved = solveRhs ? eq->rhs : eq->lhs;
    llvm::dbgs() << "[cinm-analysis]     -> enumerate {"
                 << describeMonomial(key, names) << "} ("
                 << (solveRhs ? lhsCard : rhsCard) << " combos), solve for {"
                 << describeMonomial(solved, names) << "} (removes "
                 << (solveRhs ? rhsCard : lhsCard) << "x enumeration)\n";
  }
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
// Component planning
// ===----------------------------------------------------------------------===//
//
// Structural constraints are folded into the flat index rather than filtered
// afterwards. Variables linked by a divisibility or product relation form a
// connected component; every satisfying tuple of that component is enumerated
// once, and the component then occupies a single slot sized by the solution
// count.

namespace {

/// `dividend % divisor == 0`.
struct DivRel {
  size_t divisor, dividend;
};

/// `lhsCoeff * prod(lhsVars) == rhsCoeff * prod(rhsVars)`.
struct ProdEq {
  int64_t lhsCoeff, rhsCoeff;
  llvm::SmallVector<size_t, 4> lhsVars, rhsVars;
};

/// A product equality that only holds when its guard does — the consequent of
/// an `implies(...)`, kept alongside the antecedent it hangs off.
///
/// The implication as a whole is enforced through `bounds_` like any other
/// comparison, and that alone is already correct. This exists for the other
/// half: once the guard is *settled* true, the consequent can determine a
/// variable outright, exactly as an unconditional equality does. Without it a
/// gated equality could only ever reject an assignment after the fact.
struct GatedProdEq {
  const ConstraintNode *guard;
  ProdEq eq;
};

/// The distinct values a dimension can take, sorted, for membership tests.
struct Domain {
  std::vector<ParmValue> values; ///< ascending
  bool contains(ParmValue v) const {
    return std::binary_search(values.begin(), values.end(), v);
  }
};

/// Every search parameter mentioned anywhere in `node`.
void collectVars(const ConstraintNode &node, std::set<size_t> &out) {
  if (node.kind == ConstraintNode::Kind::Var) {
    out.insert(node.varIdx());
    return;
  }
  for (const auto &child : node.operands())
    collectVars(*child, out);
}

class ComponentEnumerator {
public:
  ComponentEnumerator(llvm::ArrayRef<size_t> dims,
                      llvm::ArrayRef<Domain> domains,
                      llvm::ArrayRef<DivRel> divs, llvm::ArrayRef<ProdEq> prods,
                      llvm::ArrayRef<GatedProdEq> gated,
                      llvm::ArrayRef<const ConstraintNode *> bounds, size_t cap)
      : dims_(dims), domains_(domains), divs_(divs), prods_(prods),
        gated_(gated), bounds_(bounds), cap_(cap),
        assigned_(dims.size(), false), values_(dims.size(), 0),
        inProdEq_(dims.size(), false), isGuardVar_(dims.size(), false) {
    for (size_t pos = 0; pos < dims.size(); ++pos)
      posOfDim_[dims[pos]] = pos;
    for (const ProdEq &e : prods_)
      markProdEqVars(e);
    for (const GatedProdEq &g : gated_) {
      markProdEqVars(g.eq);
      std::set<size_t> guardVars;
      collectVars(*g.guard, guardVars);
      for (size_t v : guardVars)
        if (auto it = posOfDim_.find(v); it != posOfDim_.end())
          isGuardVar_[it->second] = true;
    }
    gatedActive_.resize(gated_.size(), false);
    refreshGatedActive();
  }

  /// Enumerate every satisfying tuple. False if a budget was exceeded, in
  /// which case `out` is meaningless and the caller must fall back.
  bool run(std::vector<std::vector<ParmValue>> &out) {
    out_ = &out;
    return recurse();
  }

private:
  size_t posOf(size_t dim) const { return posOfDim_.lookup(dim); }
  bool isAssigned(size_t dim) const { return assigned_[posOf(dim)]; }
  int64_t valueOf(size_t dim) const { return values_[posOf(dim)]; }

  void markProdEqVars(const ProdEq &e) {
    for (llvm::ArrayRef<size_t> side :
         {llvm::ArrayRef<size_t>(e.lhsVars), llvm::ArrayRef<size_t>(e.rhsVars)})
      for (size_t v : side)
        if (auto it = posOfDim_.find(v); it != posOfDim_.end())
          inProdEq_[it->second] = true;
  }

  /// Which gated equalities are in force under the current partial assignment.
  ///
  /// "In force" is boolMustHold, not boolMayHold: acting on a guard that merely
  /// *might* hold would impose its consequent on the completions where it does
  /// not, which is the one way this could produce a wrong answer rather than a
  /// slow one.
  ///
  /// Recomputed when a guard variable is assigned or unassigned rather than
  /// when it is read. The answer only depends on those variables, and it is
  /// read once per candidate value of every remaining dimension -- doing it
  /// per read costs a full node walk per gated equality per lookup, which is
  /// most of the enumeration's time once a space has any guards at all.
  void refreshGatedActive() {
    if (gated_.empty())
      return;
    VarBounds vb = varBounds();
    for (size_t i = 0; i < gated_.size(); ++i)
      gatedActive_[i] = boolMustHold(*gated_[i].guard, vb);
  }

  /// Run `fn` over the product equalities in force under the current partial
  /// assignment -- the unconditional ones, plus every gated one whose guard is
  /// settled true -- stopping at the first that returns true.
  template <class Fn> bool anyActiveProdEq(Fn fn) const {
    for (const ProdEq &e : prods_)
      if (fn(e))
        return true;
    for (size_t i = 0; i < gated_.size(); ++i)
      if (gatedActive_[i] && fn(gated_[i].eq))
        return true;
    return false;
  }

  /// Product of the assigned variables on one side, or nullopt if any is still
  /// unassigned. `skip` excludes the variable being solved for.
  std::optional<int64_t> sideProduct(llvm::ArrayRef<size_t> vars, int64_t coeff,
                                     size_t skip, bool skipOne) const {
    int64_t acc = coeff;
    bool skipped = false;
    for (size_t v : vars) {
      if (skipOne && v == skip && !skipped) {
        skipped = true;
        continue;
      }
      if (!isAssigned(v))
        return std::nullopt;
      acc *= valueOf(v);
    }
    return acc;
  }

  /// If exactly one variable of `eq` is unassigned and it is `dim`, compute the
  /// only value it can take. Returns nullopt when `dim` is not determined.
  std::optional<int64_t> solveFor(const ProdEq &eq, size_t dim) const {
    size_t occurrences = 0, unassignedCount = 0;
    for (llvm::ArrayRef<size_t> side : {llvm::ArrayRef<size_t>(eq.lhsVars),
                                        llvm::ArrayRef<size_t>(eq.rhsVars)})
      for (size_t v : side) {
        if (v == dim)
          ++occurrences;
        if (!isAssigned(v))
          ++unassignedCount;
      }
    // Solving needs the target to appear linearly and be the only unknown.
    if (occurrences != 1 || unassignedCount != 1)
      return std::nullopt;

    const bool onLhs = llvm::is_contained(eq.lhsVars, dim);
    auto known = sideProduct(onLhs ? eq.lhsVars : eq.rhsVars,
                             onLhs ? eq.lhsCoeff : eq.rhsCoeff, dim, true);
    auto other = sideProduct(onLhs ? eq.rhsVars : eq.lhsVars,
                             onLhs ? eq.rhsCoeff : eq.lhsCoeff, dim, false);
    if (!known || !other || *known == 0 || *other % *known != 0)
      return std::nullopt;
    return *other / *known;
  }

  /// Bounds for a variable given the current partial assignment: a point once
  /// assigned, otherwise its whole domain. Variables outside this component
  /// are unknown, which disables pruning for any expression mentioning them.
  VarBounds varBounds() const {
    return [this](size_t v) -> Interval {
      auto it = posOfDim_.find(v);
      if (it == posOfDim_.end())
        return {0, 0, false};
      const size_t pos = it->second;
      if (assigned_[pos])
        return {values_[pos], values_[pos], true};
      const auto &vals = domains_[pos].values;
      if (vals.empty())
        return {0, 0, false};
      return {vals.front(), vals.back(), true};
    };
  }

  /// True unless some inequality is already unsatisfiable for every completion
  /// of the current partial assignment -- in which case the whole subtree is
  /// dead and need not be walked.
  bool boundsMayHold() const {
    if (bounds_.empty())
      return true;
    VarBounds vb = varBounds();
    for (const ConstraintNode *c : bounds_)
      if (!boolMayHold(*c, vb))
        return false;
    return true;
  }

  /// Every relation whose variables are all assigned must hold.
  bool checkComplete() const {
    for (const DivRel &d : divs_) {
      if (!isAssigned(d.divisor) || !isAssigned(d.dividend))
        continue;
      int64_t div = valueOf(d.divisor);
      if (div == 0 || valueOf(d.dividend) % div != 0)
        return false;
    }
    for (const ProdEq &e : prods_) {
      auto l = sideProduct(e.lhsVars, e.lhsCoeff, 0, false);
      auto r = sideProduct(e.rhsVars, e.rhsCoeff, 0, false);
      if (l && r && *l != *r)
        return false;
    }
    return true;
  }

  /// Candidate values for the next dimension, narrowed by whatever is already
  /// known. Narrowing is what keeps this linear in the solution count instead
  /// of the product of the domains.
  llvm::SmallVector<ParmValue, 16> candidatesFor(size_t pos) const {
    const size_t dim = dims_[pos];
    const Domain &dom = domains_[pos];
    llvm::SmallVector<ParmValue, 16> out;

    // Determined by a product equality: exactly one value can work.
    if (anyActiveProdEq([&](const ProdEq &e) {
          auto v = solveFor(e, dim);
          if (!v)
            return false;
          if (*v >= std::numeric_limits<ParmValue>::min() &&
              *v <= std::numeric_limits<ParmValue>::max() &&
              dom.contains(static_cast<ParmValue>(*v)))
            out.push_back(static_cast<ParmValue>(*v));
          return true;
        }))
      return out;
    // Constrained to divide an already-known dividend: walk its divisors
    // rather than the (often much larger) declared domain.
    for (const DivRel &d : divs_) {
      if (d.divisor != dim || !isAssigned(d.dividend))
        continue;
      const int64_t n = valueOf(d.dividend);
      if (n <= 0)
        return out;
      for (int64_t i = 1; i * i <= n; ++i) {
        if (n % i)
          continue;
        for (int64_t cand : {i, n / i})
          if (cand <= std::numeric_limits<ParmValue>::max() &&
              dom.contains(static_cast<ParmValue>(cand)))
            out.push_back(static_cast<ParmValue>(cand));
      }
      llvm::sort(out);
      out.erase(std::unique(out.begin(), out.end()), out.end());
      return out;
    }
    out.assign(dom.values.begin(), dom.values.end());
    return out;
  }

  /// Which dimension to assign next, chosen against the *current* partial
  /// assignment rather than a fixed order. Ordering statically by domain size
  /// is a trap: it puts the variable a product equality determines (typically
  /// the one with the widest domain, e.g. `dpus`) last, so variables that
  /// appear in no product equality get enumerated before the equality can
  /// reject the prefix. On a batch_gemv space that is 4.2M dead leaves instead
  /// of 1.6k feasible prefixes.
  size_t selectNext() const {
    // 1. Unit propagation: a variable some equality already pins down. Taking
    //    it now is free and prunes immediately.
    for (size_t pos = 0; pos < dims_.size(); ++pos) {
      if (assigned_[pos])
        continue;
      if (anyActiveProdEq([&](const ProdEq &e) {
            return solveFor(e, dims_[pos]).has_value();
          }))
        return pos;
    }
    // 2. A variable some implication is guarded on. Its own domain is
    //    typically tiny (a mode switch), and until it is settled every
    //    equality hanging off it is inert -- so deferring it wastes the whole
    //    subtree, in the same way that deferring the variable an equality
    //    determines wastes it (see the note above).
    // 3. Then drive towards (1): a variable that participates in an equality,
    //    narrowest domain first. 4. Only once none are left do the purely
    //    divisibility-constrained variables get enumerated.
    for (int tier = 0; tier < 3; ++tier) {
      size_t best = SIZE_MAX, bestSize = SIZE_MAX;
      for (size_t pos = 0; pos < dims_.size(); ++pos) {
        if (assigned_[pos])
          continue;
        const bool wanted = tier == 0   ? isGuardVar_[pos]
                            : tier == 1 ? !isGuardVar_[pos] && inProdEq_[pos]
                                        : !isGuardVar_[pos] && !inProdEq_[pos];
        if (!wanted)
          continue;
        if (domains_[pos].values.size() < bestSize) {
          bestSize = domains_[pos].values.size();
          best = pos;
        }
      }
      if (best != SIZE_MAX)
        return best;
    }
    return SIZE_MAX; // everything assigned
  }

  bool recurse() {
    // A node budget bounds the work even when the heuristic picks badly, so a
    // pathological space falls back to predicates instead of hanging.
    if (++nodes_ > kNodeBudget)
      return false;

    const size_t pos = selectNext();
    if (pos == SIZE_MAX) {
      if (out_->size() >= cap_)
        return false;
      std::vector<ParmValue> tuple(dims_.size());
      for (size_t i = 0; i < dims_.size(); ++i)
        tuple[i] = values_[i];
      out_->push_back(std::move(tuple));
      return true;
    }
    for (ParmValue v : candidatesFor(pos)) {
      values_[pos] = v;
      assigned_[pos] = true;
      if (isGuardVar_[pos])
        refreshGatedActive();
      bool feasible = checkComplete() && boundsMayHold();
      bool ok = feasible ? recurse() : true;
      assigned_[pos] = false;
      if (isGuardVar_[pos])
        refreshGatedActive();
      if (!ok)
        return false; // budget exceeded, abort the whole enumeration
    }
    return true;
  }

  /// Search nodes visited before giving up. Generous enough that any space
  /// the heuristic handles well finishes far below it, small enough that a
  /// space it handles badly fails in seconds rather than hanging.
  static constexpr size_t kNodeBudget = 50'000'000;

  llvm::ArrayRef<size_t> dims_;
  llvm::ArrayRef<Domain> domains_;
  llvm::ArrayRef<DivRel> divs_;
  llvm::ArrayRef<ProdEq> prods_;
  /// The subset of `bounds_` that is an implication whose consequent is a
  /// product equality, pre-matched so a settled guard can determine a variable.
  llvm::ArrayRef<GatedProdEq> gated_;
  /// Comparisons and implications over this component's variables, used to
  /// prune. They are also checked at full assignment -- where every interval
  /// is a point, so the check is exact -- so a component enforces them
  /// outright rather than leaving them to be filtered later.
  llvm::ArrayRef<const ConstraintNode *> bounds_;
  size_t cap_;
  std::vector<bool> assigned_;
  std::vector<ParmValue> values_;
  /// Whether dims_[pos] appears in any product equality; drives selectNext().
  std::vector<bool> inProdEq_;
  /// Whether dims_[pos] appears in some implication's antecedent; likewise.
  std::vector<bool> isGuardVar_;
  /// Whether gated_[i]'s guard is settled true; see refreshGatedActive().
  std::vector<bool> gatedActive_;
  size_t nodes_ = 0;
  llvm::DenseMap<size_t, size_t> posOfDim_;
  std::vector<std::vector<ParmValue>> *out_ = nullptr;
};

/// Union-find over dimension indices.
class DisjointSets {
public:
  explicit DisjointSets(size_t n) : parent_(n) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }
  size_t find(size_t x) {
    while (parent_[x] != x)
      x = parent_[x] = parent_[parent_[x]];
    return x;
  }
  void unite(size_t a, size_t b) { parent_[find(a)] = find(b); }

private:
  std::vector<size_t> parent_;
};

} // namespace

void SpaceBuilder::planComponents(
    ConfigSpace &space, PlanMetadata &report,
    std::set<std::pair<std::string, std::string>> &absorbedMultiples,
    std::set<const ConstraintNode *> &absorbedPredicates) {
  /// Enumerating more tuples than this is taken as evidence that the component
  /// is not actually pruning, and the relations are left to the old paths.
  constexpr size_t kSolutionCap = 4'000'000;

  const size_t nDims = space.params.size();
  if (nDims == 0)
    return;

  // Collect the relations that can be folded into the encoding.
  std::vector<DivRel> divs;
  std::vector<std::pair<std::string, std::string>> divNames;
  for (const auto &m : multiples_) {
    int p = dimIndexByName(m.parent), c = dimIndexByName(m.child);
    if (p < 0 || c < 0)
      continue;
    divs.push_back({static_cast<size_t>(p), static_cast<size_t>(c)});
    divNames.push_back({m.parent, m.child});
  }

  /// The equality reading of a comparison, if it has one and it is worth
  /// keeping: a variable occurring on both sides determines nothing, so such
  /// an equality is left to the ordinary predicate path.
  ///
  /// A variable repeated *within* one side is kept. It cannot be solved for
  /// -- solveFor requires a single occurrence -- but sideProduct still
  /// evaluates it correctly, so the relation prunes even where it cannot
  /// determine.
  auto asSolvableProdEq =
      [](const ConstraintNode &node) -> std::optional<ProdEq> {
    auto eq = matchProductEquality(node);
    if (!eq)
      return std::nullopt;
    std::set<size_t> lhsSet(eq->lhs.vars.begin(), eq->lhs.vars.end());
    if (llvm::any_of(eq->rhs.vars,
                     [&](size_t v) { return lhsSet.count(v) != 0; }))
      return std::nullopt;
    ProdEq p;
    p.lhsCoeff = eq->lhs.coeff;
    p.rhsCoeff = eq->rhs.coeff;
    p.lhsVars.assign(eq->lhs.vars.begin(), eq->lhs.vars.end());
    p.rhsVars.assign(eq->rhs.vars.begin(), eq->rhs.vars.end());
    return p;
  };

  std::vector<ProdEq> prods;
  std::vector<const ConstraintNode *> prodNodes;
  std::vector<GatedProdEq> gated;
  for (const auto &entry : predicates_) {
    if (!entry.node)
      continue;
    if (entry.node->kind == ConstraintNode::Kind::Implies) {
      // The implication itself is enforced as a bound below, like any other
      // comparison. What is recorded here is the extra power a settled guard
      // buys: its consequent can then determine a variable.
      if (auto eq = asSolvableProdEq(*entry.node->operands()[1]))
        gated.push_back({entry.node->operands()[0].get(), std::move(*eq)});
      continue;
    }
    if (auto eq = asSolvableProdEq(*entry.node)) {
      prods.push_back(std::move(*eq));
      prodNodes.push_back(entry.node.get());
    }
  }

  if (divs.empty() && prods.empty() && gated.empty())
    return;

  // Connected components over the variables the relations link.
  DisjointSets sets(nDims);
  auto uniteAll = [&](llvm::ArrayRef<size_t> vars) {
    for (size_t i = 1; i < vars.size(); ++i)
      sets.unite(vars[0], vars[i]);
  };
  for (const DivRel &d : divs)
    sets.unite(d.divisor, d.dividend);
  for (const ProdEq &p : prods) {
    llvm::SmallVector<size_t, 8> all(p.lhsVars.begin(), p.lhsVars.end());
    all.append(p.rhsVars.begin(), p.rhsVars.end());
    uniteAll(all);
  }
  // An implication links its guard's variables to its consequent's. Without
  // this the guard keeps a slot of its own and the equalities it switches on
  // are never absorbed -- the component would enumerate them as if they always
  // held, or not at all.
  for (const auto &entry : predicates_) {
    if (!entry.node || entry.node->kind != ConstraintNode::Kind::Implies)
      continue;
    std::set<size_t> vars;
    collectVars(*entry.node, vars);
    llvm::SmallVector<size_t, 8> all(vars.begin(), vars.end());
    uniteAll(all);
  }

  // Precompute each dimension's distinct values once.
  std::vector<Domain> allDomains(nDims);
  for (size_t d = 0; d < nDims; ++d) {
    const size_t card = space[d].cardinality();
    allDomains[d].values.reserve(card);
    for (size_t i = 0; i < card; ++i)
      allDomains[d].values.push_back(space[d].valueAt(i));
    llvm::sort(allDomains[d].values);
  }

  // Group dimensions by component root, keeping dims ascending so the slot
  // order and the tuple layout are both deterministic.
  std::map<size_t, std::vector<size_t>> byRoot;
  for (size_t d = 0; d < nDims; ++d)
    byRoot[sets.find(d)].push_back(d);

  for (auto &[root, dims] : byRoot) {
    if (dims.size() < 2)
      continue; // nothing linked it to anything

    // The relations wholly inside this component.
    std::vector<DivRel> myDivs;
    std::vector<size_t> myDivIdx;
    for (size_t i = 0; i < divs.size(); ++i)
      if (sets.find(divs[i].divisor) == root) {
        myDivs.push_back(divs[i]);
        myDivIdx.push_back(i);
      }
    std::vector<ProdEq> myProds;
    std::vector<size_t> myProdIdx;
    for (size_t i = 0; i < prods.size(); ++i) {
      const auto &p = prods[i];
      size_t any = p.lhsVars.empty() ? p.rhsVars.front() : p.lhsVars.front();
      if (sets.find(any) == root) {
        myProds.push_back(p);
        myProdIdx.push_back(i);
      }
    }

    // Inequalities and implications entirely inside this component. Neither
    // can determine a variable on its own, but both cut subtrees: an
    // inequality is monotone in the tile sizes, so a prefix whose smallest
    // completion already busts a capacity is dead, and an implication is
    // violated as soon as its guard is settled and its consequent impossible.
    // Both are checked again at full assignment, where every interval is a
    // point and the check is therefore exact -- so the component enforces them
    // outright and they need not stay as predicates.
    const std::set<size_t> dimSet(dims.begin(), dims.end());
    std::vector<const ConstraintNode *> myBounds;
    std::vector<const ConstraintNode *> myBoundNodes;
    for (const auto &entry : predicates_) {
      if (!entry.node || !ConstraintNode::isBoolKind(entry.node->kind))
        continue;
      if (llvm::is_contained(prodNodes, entry.node.get()))
        continue; // already handled as a product equality
      std::set<size_t> vars;
      collectVars(*entry.node, vars);
      if (vars.empty())
        continue;
      if (!llvm::all_of(vars, [&](size_t v) { return dimSet.count(v) != 0; }))
        continue;
      myBounds.push_back(entry.node.get());
      myBoundNodes.push_back(entry.node.get());
    }

    // The pre-matched consequents belonging to the implications just picked
    // up, so a settled guard can determine a variable rather than only reject
    // one.
    std::vector<GatedProdEq> myGated;
    for (const GatedProdEq &g : gated) {
      std::set<size_t> vars;
      collectVars(*g.guard, vars);
      for (llvm::ArrayRef<size_t> side : {llvm::ArrayRef<size_t>(g.eq.lhsVars),
                                          llvm::ArrayRef<size_t>(g.eq.rhsVars)})
        vars.insert(side.begin(), side.end());
      if (!vars.empty() &&
          llvm::all_of(vars, [&](size_t v) { return dimSet.count(v) != 0; }))
        myGated.push_back(g);
    }

    std::vector<Domain> myDomains;
    for (size_t d : dims)
      myDomains.push_back(allDomains[d]);

    std::vector<std::vector<ParmValue>> solutions;
    ComponentEnumerator enumerator(dims, myDomains, myDivs, myProds, myGated,
                                   myBounds, kSolutionCap);
    if (!enumerator.run(solutions)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-space]   component of " << dims.size()
                 << " dims exceeded the solution cap; left to predicates\n");
      report.components[report.componentOf(dims, space)].abandoned = true;
      continue;
    }

    LLVM_DEBUG({
      size_t product = 1;
      for (size_t d : dims)
        product *= space[d].cardinality();
      llvm::dbgs() << "[cinm-space]   component {";
      for (size_t i = 0; i < dims.size(); ++i)
        llvm::dbgs() << (i ? ", " : "") << space[dims[i]].name;
      llvm::dbgs() << "}: " << solutions.size() << " tuples (was " << product
                   << ", "
                   << (solutions.empty()
                           ? 0.0
                           : double(product) / double(solutions.size()))
                   << "x fewer)\n";
      if (!myGated.empty())
        llvm::dbgs() << "[cinm-space]     including " << myGated.size()
                     << " gated equalit" << (myGated.size() == 1 ? "y" : "ies")
                     << ", solvable once the guard is settled\n";
    });

    const int reportIdx = report.componentOf(dims, space);
    report.components[reportIdx].solutions = solutions.size();

    ConfigSpace::SolvedComponent comp;
    comp.dims = dims;
    comp.solutions = std::move(solutions);
    space.addSolvedComponent(std::move(comp));

    for (size_t i : myDivIdx) {
      absorbedMultiples.insert(divNames[i]);
      report.constraints.push_back(
          {divNames[i].first + " | " + divNames[i].second, "folded", reportIdx,
           true});
    }
    for (size_t i : myProdIdx)
      absorbedPredicates.insert(prodNodes[i]);
    for (const ConstraintNode *n : myBoundNodes)
      absorbedPredicates.insert(n);
    // Which predicate each folded node came from is resolved in buildInto,
    // where the descriptions live; here only the component is known.
    for (const ConstraintNode *n : myBoundNodes)
      report.foldedInto[n] = reportIdx;
    for (size_t i : myProdIdx)
      report.foldedInto[prodNodes[i]] = reportIdx;
  }

  // A divisibility relation whose component was abandoned is still worth
  // folding on its own: two variables and one relation enumerate in no time,
  // and leaving it to a predicate would put the whole cross product of the two
  // domains back into the space. So each is retried as a component of its own,
  // which is all a parent/child pair ever was.
  //
  // A dimension can only belong to one component, so the first relation to
  // claim a dimension wins and the rest fall through to predicates. Which
  // relation that is depends on declaration order, which is deterministic.
  std::set<size_t> claimed;
  for (const ConfigSpace::SolvedComponent &comp : space.components)
    claimed.insert(comp.dims.begin(), comp.dims.end());

  for (size_t i = 0; i < divs.size(); ++i) {
    if (absorbedMultiples.count(divNames[i]))
      continue;
    const std::vector<size_t> pair = {
        std::min(divs[i].divisor, divs[i].dividend),
        std::max(divs[i].divisor, divs[i].dividend)};
    if (claimed.count(pair[0]) || claimed.count(pair[1]))
      continue;

    std::vector<Domain> pairDomains{allDomains[pair[0]], allDomains[pair[1]]};
    std::vector<std::vector<ParmValue>> solutions;
    ComponentEnumerator enumerator(pair, pairDomains, divs[i], {}, {}, {},
                                   kSolutionCap);
    if (!enumerator.run(solutions))
      continue;

    const int reportIdx = report.componentOf(pair, space);
    report.components[reportIdx].solutions = solutions.size();
    report.constraints.push_back(
        {divNames[i].first + " | " + divNames[i].second, "folded", reportIdx,
         true});

    ConfigSpace::SolvedComponent comp;
    comp.dims = pair;
    comp.solutions = std::move(solutions);
    space.addSolvedComponent(std::move(comp));
    absorbedMultiples.insert(divNames[i]);
    claimed.insert(pair.begin(), pair.end());
  }
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
        return makeRange(entry.var.name_, entry.lo, entry.hi);
      case DimEntry::Pow2:
        return makePow2Range(entry.var.name_, entry.lo, entry.hi);
      case DimEntry::Permutation:
        return makePermutation(entry.var.name_, entry.permutationSize);
      }
      llvm_unreachable("unknown DimKind");
    }();
    // How the domain is stored and how its values are meant are independent;
    // the declaration carries the second on the handle.
    param.kind = entry.var.kind();

    std::sort(entry.divisorFilters.begin(), entry.divisorFilters.end());
    entry.divisorFilters.erase(
        std::unique(entry.divisorFilters.begin(), entry.divisorFilters.end()),
        entry.divisorFilters.end());
    for (ParmValue n : entry.divisorFilters) {
      param.keepDivisorsOf(n);
      report->constraints.push_back(
          {std::to_string(n) + " % " + entry.var.name_ + " == 0",
           "static-filter", -1, true});
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[cinm-space]   dim '" << entry.var.name_ << "': ";
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

    *entry.var.idx_ = space.addDim(std::move(param));
  }

  // Phase 2: analyze and commit structural multiples constraints.
  // Deduplicate first.
  std::sort(multiples_.begin(), multiples_.end());
  multiples_.erase(std::unique(multiples_.begin(), multiples_.end()),
                   multiples_.end());

  // Phase 2a: plan components. Anything a component absorbs is skipped by the
  // fallback below and by phase 3, because the encoding will never offer a
  // configuration that violates it.
  std::set<std::pair<std::string, std::string>> absorbedMultiples;
  std::set<const ConstraintNode *> absorbedPredicates;
  planComponents(space, *report, absorbedMultiples, absorbedPredicates);

  // Phase 2b: whatever planning could not fold stays a predicate. A relation
  // ends up here only when both of its variables were already claimed by other
  // components, or when the enumeration that would have absorbed it exceeded a
  // budget -- so this is a fallback, not a path the common case takes.
  for (const auto &m : multiples_) {
    if (absorbedMultiples.count({m.parent, m.child}))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   left to a predicate: '"
                            << m.parent << "' | '" << m.child << "'\n");
    report->constraints.push_back(
        {m.parent + " | " + m.child, "filter", -1, true});
    SpaceVar parent = findVarByName(m.parent), child = findVarByName(m.child);
    space.addConstraint(
        [parent, child](const ConfigurationVector &c, arma::urowvec &valid) {
          valid %= vecDivides(parent[c], child[c]);
        },
        m.parent + " | " + m.child);
  }

  // Analysis pass: variable indices are assigned by now, so DSL-registered
  // constraints can be matched against the forms the encoding knows how to
  // exploit. Reporting only: it does not change the space.
  LLVM_DEBUG(reportConstraintAnalysis(space));

  // Phase 3: dynamic predicates. Each entry holds exactly one form; the
  // scalar overload of addConstraint vectorizes it per lane.
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   dynamic predicates: "
                          << predicates_.size() << "\n");
  for (auto &entry : predicates_) {
    // A constraint a component absorbed is guaranteed by the encoding; keeping
    // it as a predicate would only re-test what can no longer be violated.
    if (entry.node && absorbedPredicates.count(entry.node.get())) {
      LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   absorbed into encoding: "
                              << entry.description << "\n");
      auto it = report->foldedInto.find(entry.node.get());
      report->constraints.push_back(
          {entry.description, "folded",
           it == report->foldedInto.end() ? -1 : it->second, true});
      continue;
    }
    report->constraints.push_back(
        {entry.description, "filter", -1, entry.node != nullptr});
    std::visit(
        [&](auto &pred) {
          space.addConstraint(std::move(pred), entry.description);
        },
        entry.pred);
  }

  space.metadata = std::move(report);
}

} // namespace mlir::cinm
