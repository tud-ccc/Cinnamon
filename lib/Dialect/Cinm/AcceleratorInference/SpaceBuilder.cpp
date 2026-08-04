#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

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

SpaceVar SpaceBuilder::intRange(llvm::StringRef name, ParmValue lo, ParmValue hi) {
  SpaceVar v(name, hi);
  dims_.push_back({v, DimEntry::IntRange, lo, hi, {}});
  return v;
}

SpaceVar SpaceBuilder::pow2Range(llvm::StringRef name, ParmValue expLo, ParmValue expHi) {
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
  predicates_.push_back({description.str(), std::move(pred)});
}

void SpaceBuilder::require(Constraint pred, llvm::StringRef description) {
  predicates_.push_back({description.str(), std::move(pred)});
}

void SpaceBuilder::require(Expr expr, llvm::StringRef description) {
  const ConstraintNodePtr &node = expr.node();
  extractDivConstraints(node);
  // A bare arithmetic expression contributes only its divisibility conditions
  // (that is the `require(a / b)` spelling); only a comparison is a predicate.
  if (node->kind != ConstraintNode::Kind::Cmp)
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

void SpaceBuilder::extractDivConstraints(const ConstraintNodePtr &node) {
  if (!node)
    return;
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
    require(VecConstraint([num, den](const ConfigurationVector &c,
                                     arma::urowvec &valid) {
              const ParmVector nv = evalNodeVec(*num, c);
              const ParmVector dv = evalNodeVec(*den, c);
              valid %= vecDivides(dv, nv);
              valid %= (dv <= nv);
            }),
            desc);
    return;
  }
  require(VecConstraint([num, den](const ConfigurationVector &c,
                                   arma::urowvec &valid) {
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

/// The distinct values a dimension can take, sorted, for membership tests.
struct Domain {
  std::vector<ParmValue> values; ///< ascending
  bool contains(ParmValue v) const {
    return std::binary_search(values.begin(), values.end(), v);
  }
};

class ComponentEnumerator {
public:
  ComponentEnumerator(llvm::ArrayRef<size_t> dims,
                      llvm::ArrayRef<Domain> domains,
                      llvm::ArrayRef<DivRel> divs, llvm::ArrayRef<ProdEq> prods,
                      size_t cap)
      : dims_(dims), domains_(domains), divs_(divs), prods_(prods), cap_(cap),
        assigned_(dims.size(), false), values_(dims.size(), 0) {
    // Resolve the least-constrained-last ordering once: smaller domains first
    // both prunes earlier and, for a divisibility pair, naturally puts the
    // dividend (a divisor-filtered domain) ahead of the divisor (typically a
    // full range), which is what makes the divisor enumerable from it.
    order_.resize(dims.size());
    std::iota(order_.begin(), order_.end(), 0);
    llvm::stable_sort(order_, [&](size_t a, size_t b) {
      return domains_[a].values.size() < domains_[b].values.size();
    });
    for (size_t pos = 0; pos < dims.size(); ++pos)
      posOfDim_[dims[pos]] = pos;
  }

  /// Enumerate every satisfying tuple. False if the cap was exceeded, in which
  /// case `out` is meaningless and the caller must fall back.
  bool run(std::vector<std::vector<ParmValue>> &out) {
    out_ = &out;
    return recurse(0);
  }

private:
  size_t posOf(size_t dim) const { return posOfDim_.lookup(dim); }
  bool isAssigned(size_t dim) const { return assigned_[posOf(dim)]; }
  int64_t valueOf(size_t dim) const { return values_[posOf(dim)]; }

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
    for (const ProdEq &e : prods_) {
      if (auto v = solveFor(e, dim)) {
        if (*v >= std::numeric_limits<ParmValue>::min() &&
            *v <= std::numeric_limits<ParmValue>::max() &&
            dom.contains(static_cast<ParmValue>(*v)))
          out.push_back(static_cast<ParmValue>(*v));
        return out;
      }
    }
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

  bool recurse(size_t level) {
    if (level == order_.size()) {
      if (out_->size() >= cap_)
        return false;
      std::vector<ParmValue> tuple(dims_.size());
      for (size_t i = 0; i < dims_.size(); ++i)
        tuple[i] = values_[i];
      out_->push_back(std::move(tuple));
      return true;
    }
    const size_t pos = order_[level];
    for (ParmValue v : candidatesFor(pos)) {
      values_[pos] = v;
      assigned_[pos] = true;
      bool ok = checkComplete() ? recurse(level + 1) : true;
      assigned_[pos] = false;
      if (!ok)
        return false; // cap exceeded, abort the whole enumeration
    }
    return true;
  }

  llvm::ArrayRef<size_t> dims_;
  llvm::ArrayRef<Domain> domains_;
  llvm::ArrayRef<DivRel> divs_;
  llvm::ArrayRef<ProdEq> prods_;
  size_t cap_;
  std::vector<bool> assigned_;
  std::vector<ParmValue> values_;
  std::vector<size_t> order_;
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

  std::vector<ProdEq> prods;
  std::vector<const ConstraintNode *> prodNodes;
  for (const auto &entry : predicates_) {
    if (!entry.node)
      continue;
    auto eq = matchProductEquality(*entry.node);
    if (!eq)
      continue;
    // A variable on both sides determines nothing, and repeated variables are
    // not solvable linearly; leave those as ordinary predicates.
    std::set<size_t> lhsSet(eq->lhs.vars.begin(), eq->lhs.vars.end());
    if (llvm::any_of(eq->rhs.vars,
                     [&](size_t v) { return lhsSet.count(v) != 0; }))
      continue;
    ProdEq p;
    p.lhsCoeff = eq->lhs.coeff;
    p.rhsCoeff = eq->rhs.coeff;
    p.lhsVars.assign(eq->lhs.vars.begin(), eq->lhs.vars.end());
    p.rhsVars.assign(eq->rhs.vars.begin(), eq->rhs.vars.end());
    prods.push_back(std::move(p));
    prodNodes.push_back(entry.node.get());
  }

  if (divs.empty() && prods.empty())
    return;

  // Connected components over the variables the relations link.
  DisjointSets sets(nDims);
  for (const DivRel &d : divs)
    sets.unite(d.divisor, d.dividend);
  for (const ProdEq &p : prods) {
    llvm::SmallVector<size_t, 8> all(p.lhsVars.begin(), p.lhsVars.end());
    all.append(p.rhsVars.begin(), p.rhsVars.end());
    for (size_t i = 1; i < all.size(); ++i)
      sets.unite(all[0], all[i]);
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

    std::vector<Domain> myDomains;
    for (size_t d : dims)
      myDomains.push_back(allDomains[d]);

    std::vector<std::vector<ParmValue>> solutions;
    ComponentEnumerator enumerator(dims, myDomains, myDivs, myProds,
                                   kSolutionCap);
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
                   << ", " << (solutions.empty()
                                   ? 0.0
                                   : double(product) / double(solutions.size()))
                   << "x fewer)\n";
    });

    ConfigSpace::SolvedComponent comp;
    comp.dims = dims;
    comp.solutions = std::move(solutions);
    space.addSolvedComponent(std::move(comp));

    for (size_t i : myDivIdx)
      absorbedMultiples.insert(divNames[i]);
    for (size_t i : myProdIdx)
      absorbedPredicates.insert(prodNodes[i]);
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
