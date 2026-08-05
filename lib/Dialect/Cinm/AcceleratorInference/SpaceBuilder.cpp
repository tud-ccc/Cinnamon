#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <algorithm>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <set>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// SpaceBuilder — dimension declaration
// ===----------------------------------------------------------------------===//

SpaceVar SpaceBuilder::intRange(llvm::StringRef name, ParmValue lo,
                                ParmValue hi) {
  SpaceVar v(name, hi);
  dims_.push_back({v, DimEntry::IntRange, lo, hi, {}});
  return v;
}

SpaceVar SpaceBuilder::pow2Range(llvm::StringRef name, ParmValue expLo,
                                 ParmValue expHi) {
  SpaceVar v(name, ParmValue{1} << expHi);
  dims_.push_back({v, DimEntry::Pow2, expLo, expHi, {}});
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
  if (!isBoolKind(node->kind))
    return;
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
          matchProductEquality(*entry.node->operands[1]).has_value();
      llvm::dbgs() << "[cinm-analysis]   gated by "
                   << describeNode(*entry.node->operands[0]) << ": "
                   << describeNode(*entry.node->operands[1]) << " ("
                   << (solvable ? "Form A once the guard is settled"
                                : "not Form A")
                   << ")\n";
      continue;
    }
    auto eq = matchProductEquality(*entry.node);
    if (!eq) {
      llvm::dbgs() << "[cinm-analysis]   not Form A: " << entry.description
                   << "\n";
      continue;
    }

    const size_t lhsCard = jointCardinality(eq->lhs);
    const size_t rhsCard = jointCardinality(eq->rhs);
    llvm::dbgs() << "[cinm-analysis]   Form A: "
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

    // Enumerate the smaller side, solve for the larger (see the direction rule
    // in docs/ConstraintAnalysisDesign.md).
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

/// Whether `node` divides anywhere below it.
static bool containsDiv(const ConstraintNode &node) {
  if (node.kind == ConstraintNode::Kind::Div)
    return true;
  return llvm::any_of(node.operands, [](const ConstraintNodePtr &child) {
    return containsDiv(*child);
  });
}

void SpaceBuilder::extractDivConstraints(const ConstraintNodePtr &node) {
  if (!node)
    return;
  // Not under a guard. Everything this function reifies is unconditional, so a
  // `/` inside an implication would impose its divisibility on the very
  // configurations the guard exists to exclude.
  //
  // Leaving it unreified would be worse than refusing: integer division
  // truncates, so `implies(g, extent / block == 1)` would quietly accept a
  // block that does not divide the extent (1024/768 is 1). Reifying it
  // conditionally is possible and has no caller. So: write the multiplied-out
  // form, `implies(g, block == extent)`, which is what the analyser wants
  // anyway -- it is a product equality, and a division is not.
  if (node->kind == ConstraintNode::Kind::Implies) {
    assert(!containsDiv(*node) &&
           "an implication may not divide; state the multiplied-out form");
    return;
  }
  if (node->kind == ConstraintNode::Kind::Div)
    addDivConstraint(node->operands[0], node->operands[1]);
  for (const auto &child : node->operands)
    extractDivConstraints(child);
}

void SpaceBuilder::addDivConstraint(const ConstraintNodePtr &num,
                                    const ConstraintNodePtr &den) {
  using Kind = ConstraintNode::Kind;

  // const / var: the divisor can only ever take values dividing the constant,
  // so this is a static domain filter rather than a runtime check.
  if (num->kind == Kind::Const && den->kind == Kind::Var) {
    mustDivide(findVarByName(den->varName), num->value);
    return;
  }
  // var / var: structural, folded into the flat index encoding by
  // ConfigSpace::addMultiplesConstraint.
  if (num->kind == Kind::Var && den->kind == Kind::Var) {
    mustDivide(findVarByName(den->varName), findVarByName(num->varName));
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
                  valid %= vecDivides(dv, nv);
                  valid %= (dv <= nv);
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
// count. See docs/ConstraintAnalysisDesign.md.

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
    out.insert(*node.varIdx);
    return;
  }
  for (const auto &child : node.operands)
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

  /// Run `fn` over the product equalities in force under the current partial
  /// assignment -- the unconditional ones, plus every gated one whose guard is
  /// settled true -- stopping at the first that returns true.
  ///
  /// "Settled" is boolMustHold, not boolMayHold: acting on a guard that merely
  /// *might* hold would impose its consequent on the completions where it does
  /// not, which is the one way this could produce a wrong answer rather than a
  /// slow one.
  template <class Fn> bool anyActiveProdEq(Fn fn) const {
    for (const ProdEq &e : prods_)
      if (fn(e))
        return true;
    if (gated_.empty())
      return false;
    VarBounds vb = varBounds();
    for (const GatedProdEq &g : gated_)
      if (boolMustHold(*g.guard, vb) && fn(g.eq))
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
      bool feasible = checkComplete() && boundsMayHold();
      bool ok = feasible ? recurse() : true;
      assigned_[pos] = false;
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
    ConfigSpace &space,
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

  /// A product equality is only solvable when no variable occurs on both
  /// sides (it would determine nothing) and none repeats (not linear).
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
      if (auto eq = asSolvableProdEq(*entry.node->operands[1]))
        gated.push_back({entry.node->operands[0].get(), std::move(*eq)});
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
      if (!entry.node || !isBoolKind(entry.node->kind))
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

    ConfigSpace::SolvedComponent comp;
    comp.dims = dims;
    comp.solutions = std::move(solutions);
    space.addSolvedComponent(std::move(comp));

    for (size_t i : myDivIdx)
      absorbedMultiples.insert(divNames[i]);
    for (size_t i : myProdIdx)
      absorbedPredicates.insert(prodNodes[i]);
    for (const ConstraintNode *n : myBoundNodes)
      absorbedPredicates.insert(n);
  }
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder::buildInto
// ===----------------------------------------------------------------------===//

void SpaceBuilder::buildInto(ConfigSpace &space) {
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space] building config space:\n");

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
      }
      llvm_unreachable("unknown DimKind");
    }();

    std::sort(entry.divisorFilters.begin(), entry.divisorFilters.end());
    entry.divisorFilters.erase(
        std::unique(entry.divisorFilters.begin(), entry.divisorFilters.end()),
        entry.divisorFilters.end());
    for (ParmValue n : entry.divisorFilters)
      param.keepDivisorsOf(n);

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
  // pairwise handling below and by phase 3, because the encoding will never
  // offer a configuration that violates it.
  std::set<std::pair<std::string, std::string>> absorbedMultiples;
  std::set<const ConstraintNode *> absorbedPredicates;
  planComponents(space, absorbedMultiples, absorbedPredicates);

  // Build lookup for the full set.
  std::set<std::pair<std::string, std::string>> multsSet;
  for (auto &m : multiples_)
    multsSet.insert({m.parent, m.child});

  // childSet tracks dims already committed as structural children.
  std::set<std::string> childSet;

  std::vector<std::pair<SpaceVar, SpaceVar>> equalityFallbacks;
  std::vector<std::pair<SpaceVar, SpaceVar>> dynamicDivFallbacks;

  std::set<std::pair<std::string, std::string>> handled;

  for (auto &m : multiples_) {
    if (handled.count({m.parent, m.child}))
      continue;
    // Already guaranteed by a component's enumeration.
    if (absorbedMultiples.count({m.parent, m.child}))
      continue;

    LLVM_DEBUG({
      int pi = dimIndexByName(m.parent), ci = dimIndexByName(m.child);
      if (pi > ci)
        llvm::dbgs() << "[cinm-space]   note: '" << m.parent << "' (dim " << pi
                     << ") declared after child '" << m.child << "' (dim " << ci
                     << ") — OK for encoding\n";
    });

    // Detect mutual divisibility: A|B AND B|A → implies A == B.
    if (multsSet.count({m.child, m.parent})) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[cinm-space]   WARNING: mutual divisibility '" << m.parent
          << "' | '" << m.child << "' AND '" << m.child << "' | '" << m.parent
          << "'  (implies equality; replacing both with dynamic A==B)\n");
      handled.insert({m.parent, m.child});
      handled.insert({m.child, m.parent});
      equalityFallbacks.push_back(
          {findVarByName(m.parent), findVarByName(m.child)});
      continue;
    }

    // Detect chains: parent is already a structural child.
    if (childSet.count(m.parent)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-space]   WARNING: chained divisibility '" << m.parent
                 << "' | '" << m.child << "' where '" << m.parent
                 << "' is already a structural child"
                 << "  (converting to dynamic predicate)\n");
      handled.insert({m.parent, m.child});
      dynamicDivFallbacks.push_back(
          {findVarByName(m.parent), findVarByName(m.child)});
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   structural: '" << m.parent
                            << "' | '" << m.child << "'\n");
    space.addMultiplesConstraint(m.parent, m.child);
    childSet.insert(m.child);
  }

  // Add fallback dynamic predicates.
  for (auto [va, vb] : equalityFallbacks)
    space.addConstraint(
        [va, vb](const ConfigurationVector &c, arma::urowvec &valid) {
          valid %= va[c] == vb[c];
        },
        va.name().str() + " == " + vb.name().str());
  for (auto [parent, child] : dynamicDivFallbacks)
    space.addConstraint(
        [parent, child](const ConfigurationVector &c, arma::urowvec &valid) {
          valid %= vecDivides(parent[c], child[c]);
        },
        parent.name().str() + " | " + child.name().str());

  // Analysis pass: variable indices are assigned by now, so DSL-registered
  // constraints can be matched against the forms the encoding knows how to
  // exploit. Reporting only -- see docs/ConstraintAnalysisDesign.md, stage 4.
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
      continue;
    }
    std::visit(
        [&](auto &pred) {
          space.addConstraint(std::move(pred), entry.description);
        },
        entry.pred);
  }
}

} // namespace mlir::cinm
