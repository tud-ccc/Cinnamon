#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"
#include "cinm-mlir/Utils/Permutation.h"

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
  // A value list is the divisors of an extent or a power-of-two range, so its
  // values span orders of magnitude and are spaced multiplicatively; a
  // contiguous range is not, and is scaled as it stands.
  const size_t card = param.cardinality();
  const double lo = param.valueAt(0);
  const double hi = param.valueAt(card ? card - 1 : 0);
  double v = values[0], from = lo, to = hi;
  if (std::holds_alternative<ValueList>(param.domain)) {
    v = std::log2(v);
    from = std::log2(lo);
    to = std::log2(hi);
  }
  out.push_back(to > from ? (v - from) / (to - from) : 0.0);
}

void ParmKind<ParmValue>::appendNeighbours(
    const SearchParam &param, llvm::ArrayRef<ParmValue> values,
    llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out) {
  const size_t sub = param.subIndexOf(values[0]);
  if (sub > 0)
    out.push_back({param.valueAt(sub - 1)});
  if (sub + 1 < param.cardinality())
    out.push_back({param.valueAt(sub + 1)});
}

// ===----------------------------------------------------------------------===//
// ParmKind<Permutation> — an ordering
// ===----------------------------------------------------------------------===//
//
// These four are the only place that knows the encoding is a one-based
// lexicographic rank in a single dimension. Note the offset: the parameter is
// one-based like every other, and unrankPermutation is not.

Permutation ParmKind<Permutation>::decode(const SearchParam &param,
                                          llvm::ArrayRef<ParmValue> values) {
  llvm::SmallVector<unsigned> order =
      unrankPermutation(values[0] - 1, param.permutationSize);
  Permutation perm;
  perm.position.resize(param.permutationSize);
  for (auto [place, item] : llvm::enumerate(order))
    perm.position[item] = place;
  return perm;
}

void ParmKind<Permutation>::appendFeatures(const SearchParam &param,
                                           llvm::ArrayRef<ParmValue> values,
                                           llvm::SmallVectorImpl<double> &out) {
  const Permutation perm = decode(param, values);
  const double scale =
      param.permutationSize > 1 ? param.permutationSize - 1 : 1;
  for (unsigned p : perm.position)
    out.push_back(p / scale);
}

void ParmKind<Permutation>::appendNeighbours(
    const SearchParam &param, llvm::ArrayRef<ParmValue> values,
    llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out) {
  // Adjacent transpositions. Stepping the *rank* would land on an unrelated
  // permutation, which is the whole reason a step is the model's business.
  llvm::SmallVector<unsigned> order =
      unrankPermutation(values[0] - 1, param.permutationSize);
  for (unsigned i = 0; i + 1 < param.permutationSize; ++i) {
    std::swap(order[i], order[i + 1]);
    out.push_back({static_cast<ParmValue>(rankPermutation(order) + 1)});
    std::swap(order[i], order[i + 1]);
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

// ===----------------------------------------------------------------------===//
// SearchParam factories
// ===----------------------------------------------------------------------===//

SearchParam makeRange(llvm::StringRef name, ParmValue lo, ParmValue hi,
                      ParmValue step) {
  return SearchParam(name, IntRange{lo, hi, step}, ParmVTable::of<ParmValue>());
}

SearchParam makePow2Range(llvm::StringRef name, ParmValue loExp,
                          ParmValue hiExp) {
  std::vector<ParmValue> vals;
  for (int64_t e = loExp; e <= hiExp; ++e)
    vals.push_back(ParmValue(1) << e);
  return SearchParam(name, ValueList{std::move(vals)},
                     ParmVTable::of<ParmValue>());
}

SearchParam makeValues(llvm::StringRef name, std::vector<ParmValue> values) {
  return SearchParam(name, ValueList{std::move(values)},
                     ParmVTable::of<ParmValue>());
}

SearchParam makePermutation(llvm::StringRef name, unsigned n) {
  std::optional<int64_t> count = factorial(n);
  assert(count && "too many dimensions to enumerate their permutations");
  SearchParam param(name, IntRange{1, static_cast<ParmValue>(*count)},
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
int ConfigSpace::findIndex(llvm::StringRef name) const {
  for (int i = 0; i < static_cast<int>(params.size()); ++i)
    if (params[i].name == name)
      return i;
  return -1;
}

ParmValue ConfigSpace::get(const Configuration &config,
                           llvm::StringRef name) const {
  int idx = findIndex(name);
  if (idx < 0 || idx >= static_cast<int>(config.size()))
    return 0;
  return config[idx];
}

void ConfigSpace::addConstraint(VecConstraint &&constraint,
                                std::string description) {
  constraints.emplace_back(std::move(description), std::move(constraint));
}

void ConfigSpace::addConstraint(Constraint &&constraint,
                                std::string description) {
  // Vectorize by evaluating the scalar predicate once per lane. Captures
  // `this` rather than copying anything about the space -- safe since
  // ConfigSpace can be neither copied nor moved (its copy constructor is
  // deleted), so the address stays valid for the space's lifetime.
  addConstraint(
      [this, scalar = std::move(constraint)](const ConfigurationVector &cv,
                                             arma::urowvec &valid) {
        Configuration conf(cv.numDims());
        for (size_t j = 0; j < cv.size(); ++j) {

          // short circuit - this means we don't necessarily
          // collect all failed constraints
          if (!valid[j])
            continue;

          for (size_t d = 0; d < cv.numDims(); ++d)
            conf[d] = cv[d][j];
          // Plain assignment, not `%=`: `%` is Armadillo's elementwise
          // multiply for arma *objects*, but valid[j] is a bare uword, where
          // `%=` would be integer modulo (`x % 1 == 0` clears a passing lane,
          // and `% 0` is UB). The lane is known live thanks to the check
          // above, so overwriting it is the same as AND-ing into it.
          valid[j] = scalar(ConfWrapper(*this, conf)) ? 1u : 0u;
        }
      },
      std::move(description));
}

arma::urowvec
ConfigSpace::evalVecConstraintsMask(const ConfigurationVector &cv) const {
  arma::urowvec mask(cv.size(), arma::fill::ones);
  for (auto &[desc, c] : constraints)
    c(cv, mask);
  return mask;
}

bool ConfigSpace::isValid(const Configuration &config) const {
  if (config.size() != params.size())
    return false;
  ConfigurationVector cv(params.size(), 1);
  cv.setColumn(0, config);
  return evalVecConstraintsMask(cv)[0] != 0;
}

bool ConfigSpace::debugIsValid(const Configuration &config,
                               raw_ostream &os) const {
  if (config.size() != params.size()) {
    os << "Configuration has " << config.size() << " value(s) but this space "
       << "has " << params.size() << " parameter(s): {";
    for (size_t i = 0; i < params.size(); ++i)
      os << params[i].name << (i + 1 < params.size() ? ", " : "");
    os << "}\n";
    return false;
  }
  ConfigurationVector cv(params.size(), 1);
  cv.setColumn(0, config);

  auto wrapper = ConfWrapper(*this, config);
  bool fullyValid = true;
  for (auto &[desc, c] : constraints) {
    // Use a fresh valid mask each time so that the constraint
    // doesn't short-circuit.
    arma::urowvec valid(cv.size(), arma::fill::ones);
    c(cv, valid);
    if (!valid[0]) {
      if (fullyValid) {
        os << "Configuration " << wrapper << " violates:\n";
        fullyValid = false;
      }
      os << "  - " << (desc.empty() ? "<unnamed constraint>" : desc) << "\n";
    }
  }
  return fullyValid;
}

size_t ConfigSpace::numFeatures() const {
  size_t n = 0;
  for (const SearchParam &param : params)
    n += param.numFeatures();
  return n;
}

void ConfigSpace::encode(const Configuration &conf,
                         llvm::SmallVectorImpl<double> &out) const {
  assert(conf.size() == params.size());
  for (auto [i, param] : llvm::enumerate(params))
    param.appendFeatures(llvm::ArrayRef(conf).slice(i, param.arity()), out);
}

void ConfigSpace::setSolutions(std::vector<Configuration> &&solutions) {
  assert(llvm::is_sorted(solutions) &&
         "the flat index is a position in this list, so it has to be sorted");
  assert(llvm::all_of(solutions,
                      [this](const Configuration &c) {
                        return c.size() == params.size();
                      }) &&
         "every configuration must assign every parameter");
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
  for (size_t d = 0; d < params.size();) {
    const SearchParam &param = params[d];
    const size_t arity = param.arity();
    steps.clear();
    // One *parameter*, which is not always one dimension: an ordering is a
    // single parameter however many dimensions its encoding takes, and moving
    // one of them alone need not name another ordering at all.
    param.appendNeighbours(llvm::ArrayRef(conf).slice(d, arity), steps);
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
    d += arity;
  }
}

bool ConfigSpace::isEncodable(const Configuration &conf) const {
  if (conf.size() != params.size())
    return false;
  return std::binary_search(solutions_.begin(), solutions_.end(), conf);
}

bool ConfigSpace::debugIsEncodable(const Configuration &conf,
                                   raw_ostream &os) const {
  if (conf.size() != params.size()) {
    os << "  - has " << conf.size() << " value(s) but this space has "
       << params.size() << " parameter(s)\n";
    return false;
  }

  bool ok = true;
  for (size_t d = 0; d < params.size(); ++d) {
    if (params[d].contains(conf[d]))
      continue;
    os << "  - " << params[d].name << "=" << conf[d]
       << " is not a value this parameter can take\n";
    ok = false;
  }
  if (!ok)
    return false;

  // Every value is one its parameter can take, so what rules the
  // configuration out is a constraint over several of them at once. Which one
  // is not recoverable here -- the space holds the configurations the solver
  // found, not the constraints it found them from -- but the registered
  // predicates are still evaluated, and those that reject it are worth
  // naming before falling back to the general statement.
  if (std::binary_search(solutions_.begin(), solutions_.end(), conf))
    return true;
  if (!debugIsValid(conf, os))
    return false;
  os << "  - no configuration in this space assigns these values together\n";
  return false;
}

} // namespace mlir::cinm
