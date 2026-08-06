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

void SearchParam::appendNeighbourValues(
    ParmValue value, llvm::SmallVectorImpl<ParmValue> &out) const {
  if (kind == ParamKind::Permutation) {
    llvm::SmallVector<unsigned> order =
        unrankPermutation(value - 1, permutationSize);
    for (unsigned i = 0; i + 1 < permutationSize; ++i) {
      std::swap(order[i], order[i + 1]);
      out.push_back(static_cast<ParmValue>(rankPermutation(order) + 1));
      std::swap(order[i], order[i + 1]);
    }
    return;
  }

  const size_t sub = subIndexOf(value);
  if (sub > 0)
    out.push_back(valueAt(sub - 1));
  if (sub + 1 < cardinality())
    out.push_back(valueAt(sub + 1));
}

size_t SearchParam::numFeatures() const {
  switch (kind) {
  case ParamKind::Integer:
    return 1;
  case ParamKind::Permutation:
    assert(permutationSize > 0 && "permutation parameter has no size");
    return permutationSize;
  }
  llvm_unreachable("unknown ParamKind");
}

void SearchParam::appendFeatures(ParmValue value,
                                 llvm::SmallVectorImpl<double> &out) const {
  if (kind == ParamKind::Permutation) {
    // The rank is one-based, the encoding zero-based.
    llvm::SmallVector<unsigned> order =
        unrankPermutation(value - 1, permutationSize);
    llvm::SmallVector<unsigned> position(permutationSize);
    for (auto [axis, dim] : llvm::enumerate(order))
      position[dim] = axis;
    const double scale = permutationSize > 1 ? permutationSize - 1 : 1;
    for (unsigned p : position)
      out.push_back(p / scale);
    return;
  }

  // A value list is the divisors of an extent or a power-of-two range, so its
  // values span orders of magnitude and are spaced multiplicatively; a
  // contiguous range is not, and is scaled as it stands.
  const size_t card = cardinality();
  const double lo = valueAt(0);
  const double hi = valueAt(card ? card - 1 : 0);
  double v = value, from = lo, to = hi;
  if (std::holds_alternative<ValueList>(domain)) {
    v = std::log2(v);
    from = std::log2(lo);
    to = std::log2(hi);
  }
  out.push_back(to > from ? (v - from) / (to - from) : 0.0);
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
  return SearchParam(name, IntRange{lo, hi, step});
}

SearchParam makePow2Range(llvm::StringRef name, ParmValue loExp,
                          ParmValue hiExp) {
  std::vector<ParmValue> vals;
  for (int64_t e = loExp; e <= hiExp; ++e)
    vals.push_back(ParmValue(1) << e);
  return SearchParam(name, ValueList{std::move(vals)});
}

SearchParam makeValues(llvm::StringRef name, std::vector<ParmValue> values) {
  return SearchParam(name, ValueList{std::move(values)});
}

SearchParam makePermutation(llvm::StringRef name, unsigned n) {
  std::optional<int64_t> count = factorial(n);
  assert(count && "too many dimensions to enumerate their permutations");
  SearchParam param(name, IntRange{1, static_cast<ParmValue>(*count)},
                    ParamKind::Permutation);
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
  for (auto [param, value] : llvm::zip_equal(params, conf))
    param.appendFeatures(value, out);
}

void ConfigSpace::addSolvedComponent(SolvedComponent &&component) {
  component.indexOfSolution.clear();
  for (size_t i = 0; i < component.solutions.size(); ++i)
    component.indexOfSolution.emplace(component.solutions[i], i);
  components.push_back(std::move(component));
  encodingValid_ = false;
}

void ConfigSpace::ensureEncoding() const {
  if (encodingValid_)
    return;

  slots_.clear();

  // A component owns every dimension it covers: those dimensions get no slot
  // of their own, and the component contributes a single slot sized by its
  // enumerated solution count. The first dimension of each component acts as
  // its anchor so slot order stays deterministic.
  std::vector<size_t> dimToComponent(params.size(), SIZE_MAX);
  for (size_t ci = 0; ci < components.size(); ++ci)
    for (size_t d : components[ci].dims)
      dimToComponent[d] = ci;

  for (size_t i = 0; i < params.size(); ++i) {
    size_t ci = dimToComponent[i];
    if (ci == SIZE_MAX) {
      slots_.push_back({i, SIZE_MAX, params[i].cardinality()});
      continue;
    }
    // Emit the component's slot once, at its first (lowest-index) dimension.
    if (!components[ci].dims.empty() && components[ci].dims[0] == i)
      slots_.push_back({i, ci, components[ci].size()});
  }

  const size_t S = slots_.size();
  suffixProd_.resize(S + 1);
  suffixProd_[S] = 1;
  for (size_t i = S; i-- > 0;)
    suffixProd_[i] = suffixProd_[i + 1] * slots_[i].slotSize;

  encodingValid_ = true;
}

size_t ConfigSpace::totalSize() const {
  ensureEncoding();
  return slots_.empty() ? 1 : suffixProd_[0];
}

void ConfigSpace::at(size_t idx, Configuration &conf) const {
  ensureEncoding();
  conf.resize(params.size());
  for (size_t si = slots_.size(); si-- > 0;) {
    const auto &slot = slots_[si];
    size_t subIdx = idx % slot.slotSize;
    idx /= slot.slotSize;
    if (slot.componentIdx == SIZE_MAX) {
      conf[slot.dimIdx] = params[slot.dimIdx].valueAt(subIdx);
      continue;
    }
    const auto &comp = components[slot.componentIdx];
    const auto &tuple = comp.solutions[subIdx];
    for (size_t k = 0; k < comp.dims.size(); ++k)
      conf[comp.dims[k]] = tuple[k];
  }
}

void ConfigSpace::forEach(
    std::function<bool(const Configuration &, size_t)> fn) const {
  ensureEncoding();
  // Use suffixProd_[0] directly — avoids a redundant ensureEncoding() call
  // inside totalSize() after we already ensured encoding above.
  const size_t total = slots_.empty() ? 1 : suffixProd_[0];
  forEachChunk(0, total, std::move(fn));
}

void ConfigSpace::forEachChunk(
    size_t lo, size_t hi,
    std::function<bool(const Configuration &, size_t)> fn) const {
  ensureEncoding();
  if (lo >= hi)
    return;
  const size_t S = slots_.size();

  // Per-slot combined sub-index in [0, slot.slotSize).
  std::vector<size_t> subIdx(S, 0);
  Configuration conf(params.size());

  // Decode sub-index k for slot si and write the corresponding parameter
  // values into conf, exactly as ConfigSpace::at() does for one slot.
  auto applySubIdx = [&](size_t si, size_t k) {
    const auto &slot = slots_[si];
    if (slot.componentIdx == SIZE_MAX) {
      conf[slot.dimIdx] = params[slot.dimIdx].valueAt(k);
      return;
    }
    const auto &comp = components[slot.componentIdx];
    const auto &tuple = comp.solutions[k];
    for (size_t j = 0; j < comp.dims.size(); ++j)
      conf[comp.dims[j]] = tuple[j];
  };

  // Decompose lo into per-slot sub-indices (same mixed-radix decoding as
  // at()) and initialise conf at flat index lo. This is the only O(S) step;
  // every subsequent step below is O(1) amortised.
  size_t rem = lo;
  for (size_t si = S; si-- > 0;) {
    subIdx[si] = rem % slots_[si].slotSize;
    rem /= slots_[si].slotSize;
    applySubIdx(si, subIdx[si]);
  }

  for (size_t flat = lo; flat < hi; ++flat) {
    if (!fn(conf, flat))
      return;

    if (flat + 1 == hi)
      break;

    // Mixed-radix increment from the least-significant slot.
    for (size_t si = S; si-- > 0;) {
      ++subIdx[si];
      bool carry = (subIdx[si] >= slots_[si].slotSize);
      if (carry)
        subIdx[si] = 0;
      applySubIdx(si, subIdx[si]);
      if (!carry)
        break;
    }
  }
}

size_t ConfigSpace::indexOf(const Configuration &conf) const {
  ensureEncoding();
  size_t idx = 0;
  for (size_t si = 0; si < slots_.size(); ++si) {
    const auto &slot = slots_[si];
    size_t subIdx;
    if (slot.componentIdx == SIZE_MAX) {
      subIdx = params[slot.dimIdx].subIndexOf(conf[slot.dimIdx]);
    } else {
      const auto &comp = components[slot.componentIdx];
      std::vector<ParmValue> tuple(comp.dims.size());
      for (size_t k = 0; k < comp.dims.size(); ++k)
        tuple[k] = conf[comp.dims[k]];
      auto it = comp.indexOfSolution.find(tuple);
      // A configuration the component does not offer has no flat index. This
      // is reachable only if a caller invents a configuration by hand; at()
      // can never produce one.
      assert(it != comp.indexOfSolution.end() &&
             "configuration violates a structural constraint");
      subIdx = it == comp.indexOfSolution.end() ? 0 : it->second;
    }
    idx = idx * slot.slotSize + subIdx;
  }
  return idx;
}

void ConfigSpace::neighborIndices(size_t idx,
                                  llvm::SmallVectorImpl<size_t> &result) const {
  ensureEncoding();

  // Decode, step one dimension, re-encode. The obvious alternative -- stride
  // arithmetic on the flat index -- only coincides with "one step in one
  // dimension" for slots holding a single dimension. Stepping a component's
  // slot moves to the next enumerated tuple, which can differ in several
  // dimensions at once, so it does not mean what this function claims.
  Configuration conf;
  at(idx, conf);
  Configuration probe = conf;

  llvm::SmallVector<ParmValue, 4> steps;
  for (size_t d = 0; d < params.size(); ++d) {
    steps.clear();
    params[d].appendNeighbourValues(conf[d], steps);
    for (ParmValue step : steps) {
      probe[d] = step;
      // Stepping a dimension inside a component can land on a tuple the
      // component does not offer -- that neighbour simply does not exist.
      if (isEncodable(probe))
        result.push_back(indexOf(probe));
      probe[d] = conf[d];
    }
  }
}

bool ConfigSpace::isEncodable(const Configuration &conf) const {
  ensureEncoding();
  if (conf.size() != params.size())
    return false;
  for (size_t d = 0; d < params.size(); ++d)
    if (!params[d].contains(conf[d]))
      return false;
  for (const auto &slot : slots_) {
    if (slot.componentIdx == SIZE_MAX)
      continue;
    const auto &comp = components[slot.componentIdx];
    std::vector<ParmValue> tuple(comp.dims.size());
    for (size_t k = 0; k < comp.dims.size(); ++k)
      tuple[k] = conf[comp.dims[k]];
    if (!comp.indexOfSolution.count(tuple))
      return false;
  }
  return true;
}

bool ConfigSpace::debugIsEncodable(const Configuration &conf,
                                   raw_ostream &os) const {
  ensureEncoding();
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

  for (const auto &slot : slots_) {
    if (slot.componentIdx == SIZE_MAX)
      continue;
    const auto &comp = components[slot.componentIdx];
    std::vector<ParmValue> tuple(comp.dims.size());
    for (size_t k = 0; k < comp.dims.size(); ++k)
      tuple[k] = conf[comp.dims[k]];
    if (comp.indexOfSolution.count(tuple))
      continue;
    // Which relation is broken is not recoverable here -- the component holds
    // the tuples it enumerated, not the constraints it enumerated them from --
    // so name the parameters and leave the reader to look at the constraints
    // over them.
    os << "  - no configuration in this space assigns {";
    for (size_t k = 0; k < comp.dims.size(); ++k)
      os << (k ? ", " : "") << params[comp.dims[k]].name << "=" << tuple[k];
    os << "} together\n";
    ok = false;
  }
  return ok;
}

} // namespace mlir::cinm
