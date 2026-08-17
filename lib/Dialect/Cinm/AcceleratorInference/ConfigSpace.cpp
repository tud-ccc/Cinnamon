#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// SearchParam
// ===----------------------------------------------------------------------===//

double SearchParam::dlo() const {
  return std::visit(
      [](auto &&d) -> double {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return static_cast<double>(d.lo);
        else
          return d.values.empty() ? 0.0 : static_cast<double>(d.values.front());
      },
      domain);
}

double SearchParam::dhi() const {
  return std::visit(
      [](auto &&d) -> double {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return static_cast<double>(d.hi);
        else
          return d.values.empty() ? 0.0 : static_cast<double>(d.values.back());
      },
      domain);
}

std::string SearchParam::dimName(size_t k) const {
  assert(k < arity() && "dimension index out of range for this parameter");
  if (arity() == 1)
    return name;
  return name + "[" + std::to_string(k) + "]";
}

size_t SearchParam::cardinality() const {
  return std::visit(
      [](auto &&d) -> size_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return (d.hi - d.lo) / d.step + 1;
        else
          return d.values.size();
      },
      domain);
}

// ===----------------------------------------------------------------------===//
// ParmKind<ParmValue> — a quantity
// ===----------------------------------------------------------------------===//

void ParmKind<ParmValue>::appendFeatures(const SearchParam &param,
                                         llvm::ArrayRef<ParmValue> values,
                                         llvm::SmallVectorImpl<double> &out) {
  // A multiplicative quantity is taken into log space *before* the domain
  // scaling, so that equal ratios come out equally far apart: 2 and 4 land as
  // far apart as 512 and 1024. Both branches end in the same [0, 1], so the
  // choice changes how a parameter's values are distributed within its
  // feature, never how large that feature is next to another's.
  const size_t card = param.cardinality();
  const double lo = param.valueAt(0);
  const double hi = param.valueAt(card ? card - 1 : 0);
  double v = values[0], from = lo, to = hi;
  if (param.spacing == Spacing::Multiplicative) {
    v = std::log2(v);
    from = std::log2(lo);
    to = std::log2(hi);
  }
  out.push_back(to > from ? (v - from) / (to - from) : 0.0);
}

void ParmKind<ParmValue>::appendNeighbours(
    const SearchParam &param, llvm::ArrayRef<ParmValue> values,
    llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out) {
  const size_t card = param.cardinality();
  const size_t sub = param.subIndexOf(values[0]);
  llvm::SmallVector<size_t, 4> steps;
  if (sub > 0)
    steps.push_back(sub - 1);
  if (sub + 1 < card)
    steps.push_back(sub + 1);

  // A multiplicative quantity also steps by doubling and halving. The adjacent
  // values stay rather than being replaced: they are what keeps the
  // neighbourhood connected across a domain that is not a chain of ratios --
  // the divisors of 12 are not, and 4 would have no way to reach 6 -- and a
  // step that leaves the feasible set costs nothing, since the caller drops
  // it. What the ratio steps add is reach along a domain stored as a plain
  // range under a divides constraint, where v±1 is almost never a divisor and
  // v*2 usually is.
  if (param.spacing == Spacing::Multiplicative && values[0] > 0) {
    for (double target : {values[0] * 2.0, values[0] / 2.0}) {
      size_t near = param.nearestSubIndex(target);
      if (near != sub && !llvm::is_contained(steps, near))
        steps.push_back(near);
    }
  }

  for (size_t s : steps)
    out.push_back({param.valueAt(s)});
}

// ===----------------------------------------------------------------------===//
// ParmKind<Permutation> — an ordering
// ===----------------------------------------------------------------------===//
//
// These three are the only place that knows the encoding is one dimension per
// item holding a one-based place. Note the offset: every domain is strictly
// positive (SpaceBuilder.h §2) and Permutation::position is not.

Permutation ParmKind<Permutation>::decode(const SearchParam &param,
                                          llvm::ArrayRef<ParmValue> values) {
  assert(values.size() == param.permutationSize);
  Permutation perm;
  perm.position.reserve(values.size());
  for (ParmValue place : values)
    perm.position.push_back(static_cast<unsigned>(place) - 1);
  return perm;
}

void ParmKind<Permutation>::appendFeatures(const SearchParam &param,
                                           llvm::ArrayRef<ParmValue> values,
                                           llvm::SmallVectorImpl<double> &out) {
  const double scale =
      param.permutationSize > 1 ? param.permutationSize - 1 : 1;
  for (ParmValue place : values)
    out.push_back((place - 1) / scale);
}

void ParmKind<Permutation>::appendNeighbours(
    const SearchParam &param, llvm::ArrayRef<ParmValue> values,
    llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out) {
  // Adjacent transpositions: swap the items occupying two consecutive places.
  // Stepping one *dimension* would name no ordering at all -- two items would
  // share a place -- which is why a step is the model's business and spans the
  // whole parameter.
  llvm::SmallVector<ParmValue, 4> step(values.begin(), values.end());
  for (ParmValue place = 1; place < static_cast<ParmValue>(values.size());
       ++place) {
    for (ParmValue &v : step)
      v = v == place ? place + 1 : (v == place + 1 ? place : v);
    out.push_back(step);
    step.assign(values.begin(), values.end());
  }
}

SearchParam &SearchParam::keepDivisorsOf(ParmValue n) {
  if (auto *range = std::get_if<IntRange>(&domain)) {
    std::vector<ParmValue> kept;
    for (ParmValue v = range->lo; v <= range->hi; v += range->step)
      if (v > 0 && n % v == 0)
        kept.push_back(v);
    domain = ValueList{std::move(kept)};
  } else {
    auto &vals = std::get<ValueList>(domain).values;
    vals.erase(std::remove_if(vals.begin(), vals.end(),
                              [n](int64_t v) { return v <= 0 || n % v != 0; }),
               vals.end());
  }
  return *this;
}

ParmValue SearchParam::valueAt(size_t subIdx) const {
  return std::visit(
      [subIdx](auto &&d) -> int64_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return d.lo + static_cast<int64_t>(subIdx) * d.step;
        else
          return d.values[subIdx];
      },
      domain);
}

bool SearchParam::contains(ParmValue value) const {
  const size_t sub = subIndexOf(value);
  // subIndexOf is arithmetic for a range, so an out-of-domain value gives an
  // out-of-range (or wrapped) sub-index, and a value between two steps gives
  // one that decodes back to a different value. Both checks are needed.
  return sub < cardinality() && valueAt(sub) == value;
}

size_t SearchParam::subIndexOf(ParmValue value) const {
  return std::visit(
      [value](auto &&d) -> size_t {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, IntRange>)
          return static_cast<size_t>((value - d.lo) / d.step);
        else {
          auto it = std::find(d.values.begin(), d.values.end(), value);
          return static_cast<size_t>(std::distance(d.values.begin(), it));
        }
      },
      domain);
}

size_t SearchParam::nearestSubIndex(double target) const {
  const size_t card = cardinality();
  assert(card > 0 && "an empty domain has no nearest value");
  // Ratio distance, since the only caller is a multiplicative step: between 6
  // and 12, a target of 8 is nearer 6, and counting would have to be told so.
  // Domains are positive by construction (SpaceBuilder.h §2), so the log is
  // defined everywhere except a target of zero, which halving cannot reach.
  const double want = std::log2(std::max(target, 1e-9));
  auto dist = [&](size_t i) {
    return std::abs(std::log2(static_cast<double>(valueAt(i))) - want);
  };
  auto better = [&](size_t cand, size_t best) {
    return dist(cand) < dist(best) ? cand : best;
  };

  if (auto *range = std::get_if<IntRange>(&domain)) {
    // The values ascend, so the nearest is one of the two that bracket the
    // target. Landing on the bracket by arithmetic keeps this O(1): a domain
    // stored as a range is exactly the one large enough for a scan to show up
    // in the neighbourhood BFS.
    double pos = (target - static_cast<double>(range->lo)) /
                 static_cast<double>(range->step);
    auto lower = static_cast<int64_t>(std::floor(pos));
    size_t best = static_cast<size_t>(std::clamp<int64_t>(lower, 0, card - 1));
    if (lower + 1 >= 0 && static_cast<size_t>(lower + 1) < card)
      best = better(static_cast<size_t>(lower + 1), best);
    return best;
  }

  // A value list carries no ordering guarantee, and the lists that reach here
  // -- a divisor set, the powers of two -- are small enough that it does not
  // matter.
  size_t best = 0;
  for (size_t i = 1; i < card; ++i)
    best = better(i, best);
  return best;
}

// ===----------------------------------------------------------------------===//
// SearchParam factories
// ===----------------------------------------------------------------------===//

SearchParam makeRange(llvm::StringRef name, ParmValue lo, ParmValue hi,
                      ParmValue step, Spacing spacing) {
  SearchParam param(name, IntRange{lo, hi, step}, ParmVTable::of<ParmValue>());
  param.spacing = spacing;
  return param;
}

SearchParam makePow2Range(llvm::StringRef name, ParmValue loExp,
                          ParmValue hiExp) {
  std::vector<ParmValue> vals;
  for (int64_t e = loExp; e <= hiExp; ++e)
    vals.push_back(ParmValue(1) << e);
  SearchParam param(name, ValueList{std::move(vals)},
                    ParmVTable::of<ParmValue>());
  // Not overridable: a domain that *is* the powers of two is multiplicative by
  // construction, whatever it is used for.
  param.spacing = Spacing::Multiplicative;
  return param;
}

SearchParam makeValues(llvm::StringRef name, std::vector<ParmValue> values,
                       Spacing spacing) {
  SearchParam param(name, ValueList{std::move(values)},
                    ParmVTable::of<ParmValue>());
  param.spacing = spacing;
  return param;
}

SearchParam makePermutation(llvm::StringRef name, unsigned n) {
  assert(n > 0 && "an ordering of nothing is not a parameter");
  // One domain for all n dimensions: each holds a place in [1, n], and it is
  // the distinctness the solver posts that makes them an ordering rather than
  // n independent choices.
  SearchParam param(name, IntRange{1, static_cast<ParmValue>(n)},
                    ParmVTable::of<Permutation>());
  param.permutationSize = n;
  return param;
}

llvm::StringRef paramKindName(ParamKind kind) {
  switch (kind) {
  case ParamKind::Integer:
    return "integer";
  case ParamKind::Permutation:
    return "permutation";
  }
  llvm_unreachable("unknown ParamKind");
}

// ===----------------------------------------------------------------------===//
// ConfigSpace
// ===----------------------------------------------------------------------===//
size_t ConfigSpace::addParam(SearchParam &&param) {
  const size_t firstDim = paramOfDim_.size();
  const size_t arity = param.arity();
  assert(arity > 0 && "a parameter occupies at least one dimension");
  firstDimOfParam_.push_back(firstDim);
  paramOfDim_.insert(paramOfDim_.end(), arity, params.size());
  params.push_back(std::move(param));
  return firstDim;
}

int ConfigSpace::findParam(llvm::StringRef name) const {
  for (int i = 0; i < static_cast<int>(params.size()); ++i)
    if (params[i].name == name)
      return i;
  return -1;
}

ParmValue ConfigSpace::get(const Configuration &config,
                           llvm::StringRef name) const {
  int param = findParam(name);
  if (param < 0)
    return 0;
  assert(params[param].arity() == 1 &&
         "this parameter spans several dimensions; use getAs<T>()");
  size_t dim = firstDimOfParam_[param];
  return dim < config.size() ? config[dim] : 0;
}

size_t ConfigSpace::numFeatures() const {
  size_t n = 0;
  for (const SearchParam &param : params)
    n += param.numFeatures();
  return n;
}

void ConfigSpace::encode(const Configuration &conf,
                         llvm::SmallVectorImpl<double> &out) const {
  assert(conf.size() == numDims());
  for (size_t p = 0; p < params.size(); ++p)
    params[p].appendFeatures(paramValues(conf, p), out);
}

void ConfigSpace::setSolutions(std::vector<Configuration> &&solutions) {
  assert(llvm::is_sorted(solutions) &&
         "the flat index is a position in this list, so it has to be sorted");
  assert(llvm::all_of(solutions,
                      [this](const Configuration &c) {
                        return c.size() == numDims();
                      }) &&
         "every configuration must assign every dimension");
  solutions_ = std::move(solutions);
}

size_t ConfigSpace::totalSize() const { return solutions_.size(); }

void ConfigSpace::at(size_t idx, Configuration &conf) const {
  assert(idx < solutions_.size() && "flat index out of range");
  conf = solutions_[idx];
}

void ConfigSpace::forEach(
    std::function<bool(const Configuration &, size_t)> fn) const {
  forEachChunk(0, solutions_.size(), std::move(fn));
}

void ConfigSpace::forEachChunk(
    size_t lo, size_t hi,
    std::function<bool(const Configuration &, size_t)> fn) const {
  hi = std::min(hi, solutions_.size());
  for (size_t i = lo; i < hi; ++i)
    if (!fn(solutions_[i], i))
      return;
}

size_t ConfigSpace::indexOf(const Configuration &conf) const {
  auto it = std::lower_bound(solutions_.begin(), solutions_.end(), conf);
  assert(it != solutions_.end() && *it == conf &&
         "configuration is not in this space; check isEncodable() first");
  if (it == solutions_.end() || *it != conf)
    return solutions_.size();
  return static_cast<size_t>(std::distance(solutions_.begin(), it));
}

void ConfigSpace::neighborIndices(size_t idx,
                                  llvm::SmallVectorImpl<size_t> &result) const {
  // Step one parameter at a time and look the result up. Arithmetic on the
  // flat index would be meaningless: consecutive indices are lexicographic
  // neighbours in the *feasible* set, which says nothing about how far apart
  // two configurations are.
  const Configuration &conf = solutions_[idx];
  Configuration probe = conf;

  llvm::SmallVector<llvm::SmallVector<ParmValue, 4>, 4> steps;
  for (size_t p = 0; p < params.size(); ++p) {
    const SearchParam &param = params[p];
    const size_t d = firstDimOfParam_[p];
    const size_t arity = param.arity();
    steps.clear();
    // One *parameter*, which is not always one dimension: an ordering is a
    // single parameter however many dimensions its encoding takes, and moving
    // one of them alone need not name another ordering at all.
    param.appendNeighbours(paramValues(conf, p), steps);
    for (const auto &step : steps) {
      assert(step.size() == arity && "a step must assign the whole parameter");
      std::copy(step.begin(), step.end(), probe.begin() + d);
      // A step that leaves the feasible set is not a neighbour: the space
      // contains no such configuration, so there is nothing to move to.
      auto it = std::lower_bound(solutions_.begin(), solutions_.end(), probe);
      if (it != solutions_.end() && *it == probe)
        result.push_back(
            static_cast<size_t>(std::distance(solutions_.begin(), it)));
      std::copy(conf.begin() + d, conf.begin() + d + arity, probe.begin() + d);
    }
  }
}

bool ConfigSpace::isEncodable(const Configuration &conf) const {
  if (conf.size() != numDims())
    return false;
  return std::binary_search(solutions_.begin(), solutions_.end(), conf);
}

bool ConfigSpace::debugIsEncodable(const Configuration &conf,
                                   raw_ostream &os) const {
  if (conf.size() != numDims()) {
    os << "  - has " << conf.size() << " value(s) but this space has "
       << numDims() << " dimension(s)\n";
    return false;
  }

  bool ok = true;
  for (size_t d = 0; d < numDims(); ++d) {
    if (paramAtDim(d).contains(conf[d]))
      continue;
    os << "  - " << dimName(d) << "=" << conf[d]
       << " is not a value this parameter can take\n";
    ok = false;
  }
  if (!ok)
    return false;

  // Every value is one its parameter can take, so what rules the
  // configuration out is a constraint over several of them at once. Which one
  // is not recoverable here: the space holds the configurations the solver
  // found, not the constraints it found them from. What IS recoverable, and
  // localises the conflict nearly as well, is the nearest feasible
  // configuration: the dimensions it differs in are where the conflict lives.
  if (std::binary_search(solutions_.begin(), solutions_.end(), conf))
    return true;
  os << "  - no configuration in this space assigns these values together\n";

  size_t bestMismatches = std::numeric_limits<size_t>::max();
  const Configuration *best = nullptr;
  llvm::SmallVector<std::pair<size_t, ParmValue>> singleDimFixes;
  for (const Configuration &sol : solutions_) {
    size_t mismatches = 0;
    size_t lastDim = 0;
    for (size_t d = 0; d < numDims() && mismatches <= bestMismatches; ++d)
      if (sol[d] != conf[d]) {
        ++mismatches;
        lastDim = d;
      }
    if (mismatches == 1)
      singleDimFixes.push_back({lastDim, sol[lastDim]});
    if (mismatches < bestMismatches) {
      bestMismatches = mismatches;
      best = &sol;
    }
  }
  if (!singleDimFixes.empty()) {
    os << "  - it becomes feasible by changing ONE value; for example:\n";
    for (size_t i = 0; i < singleDimFixes.size() && i < 5; ++i)
      os << "      " << dimName(singleDimFixes[i].first) << "="
         << singleDimFixes[i].second << " (instead of "
         << conf[singleDimFixes[i].first] << ")\n";
    if (singleDimFixes.size() > 5)
      os << "      ... and " << (singleDimFixes.size() - 5) << " more\n";
  } else if (best) {
    os << "  - the nearest feasible configuration differs in " << bestMismatches
       << " value(s):\n";
    for (size_t d = 0; d < numDims(); ++d)
      if ((*best)[d] != conf[d])
        os << "      " << dimName(d) << "=" << (*best)[d] << " (instead of "
           << conf[d] << ")\n";
  }
  return false;
}

} // namespace mlir::cinm
