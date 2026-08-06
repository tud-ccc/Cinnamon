#pragma once

#include <armadillo>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <memory>
#include <mlir/IR/Diagnostics.h>
#include <string>
#include <variant>
#include <vector>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// The configuration space
// ===----------------------------------------------------------------------===//
//
// The data structure a search searches: named integer parameters, an encoding
// that gives every configuration an index, and the predicates a configuration
// has to satisfy. Nothing here decides anything -- how a space is *built* is
// SpaceBuilder's business, and what is done with one is
// AcceleratorInference.h's. See the top of SpaceBuilder.h for how the pieces
// fit together.

using ParmValue = int32_t;
using ParmVector = arma::Row<ParmValue>;

// ===----------------------------------------------------------------------===//
// Configuration space types
// ===----------------------------------------------------------------------===//

/// Contiguous integer range [lo, hi] sampled at multiples of step.
struct IntRange {
  ParmValue lo, hi;
  ParmValue step = 1;
};

/// Explicit discrete value set.
struct ValueList {
  std::vector<ParmValue> values;
};

/// What a parameter's values *mean*, as opposed to how its domain is stored.
///
/// Every parameter is a positive integer either way; the kind says how those
/// integers are to be read, and therefore what may be done with them. A
/// magnitude can be added, multiplied, divided and ordered; the rank of a
/// permutation can only be compared for equality, since arithmetic on a rank
/// is arithmetic on an arbitrary numbering. The constraint DSL enforces this.
///
/// The kind also decides how a value is presented to the surrogate, which is
/// the same question one level down: two ranks one apart are not two similar
/// permutations, so a model fed a rank as a magnitude is fitting something that
/// does not exist. See appendFeatures.
enum class ParamKind {
  /// A quantity. Tile sizes, counts, capacities.
  Integer,
  /// The lexicographic rank of a permutation, one-based -- see
  /// cinm-mlir/Utils/Permutation.h for the encoding, and note the offset.
  Permutation,
};

llvm::StringRef paramKindName(ParamKind kind);

/// One dimension of the search space.
struct SearchParam {
  std::string name;
  std::variant<IntRange, ValueList> domain;
  ParamKind kind = ParamKind::Integer;
  /// For ParamKind::Permutation: how many items are permuted. The domain then
  /// holds the ranks 1..n!, which is not something to recover n from.
  unsigned permutationSize = 0;

  SearchParam(const SearchParam &) = delete;
  SearchParam(SearchParam &&) = default;
  SearchParam(StringRef name, IntRange &&range,
              ParamKind kind = ParamKind::Integer)
      : name(name.str()), domain(std::move(range)), kind(kind) {}
  SearchParam(StringRef name, ValueList &&list,
              ParamKind kind = ParamKind::Integer)
      : name(name.str()), domain(std::move(list)), kind(kind) {}
  SearchParam &operator=(SearchParam &&o) = default;

  /// Smallest and largest value this parameter can take. For reporting; the
  /// surrogate sees appendFeatures() instead, and nothing maps back from a
  /// feature to a value.
  double dlo() const;
  double dhi() const;
  /// Number of distinct values this parameter can take.
  size_t cardinality() const;
  /// How many surrogate features this parameter contributes. One for a
  /// quantity; a permutation of n items contributes n.
  size_t numFeatures() const;
  /// Append this parameter's features for `value` to `out`.
  ///
  /// Two things are going on, and they are both about what the surrogate can
  /// learn rather than about the parameter itself.
  ///
  /// **A permutation contributes its position vector**: feature `i` is the
  /// axis that iteration dimension `i` occupies. Euclidean distance between
  /// two such vectors is Spearman's rank distance, so two orders that agree
  /// about most dimensions land near each other and the network can generalise
  /// between them. Its rank cannot do that -- consecutive ranks are unrelated
  /// permutations -- and one feature per permutation could not either, since
  /// no single number carries the structure.
  ///
  /// **Every feature is scaled to [0, 1]** against the parameter's declared
  /// domain, so that features are comparable to each other. The scaling has to
  /// come from the domain rather than from the sample, because training and
  /// prediction encode different sets of configurations and a model fitted on
  /// one scale cannot be asked about another.
  void appendFeatures(ParmValue value,
                      llvm::SmallVectorImpl<double> &out) const;

  /// Return the i-th distinct value of this parameter (0-indexed).
  ParmValue valueAt(size_t subIdx) const;
  /// Return the sub-index of value within this parameter's domain (inverse of
  /// valueAt). Meaningful only for a value the domain contains -- see
  /// contains(), which is how a caller holding an arbitrary value checks.
  size_t subIndexOf(ParmValue value) const;
  /// Whether `value` is one this parameter can take. A value outside the
  /// domain has no sub-index, so subIndexOf() cannot report this itself.
  bool contains(ParmValue value) const;

  /// Append the values one step from `value`, for whatever "one step" means
  /// for this parameter: the adjacent values of a quantity, and the adjacent
  /// transpositions of a permutation. Stepping a permutation's *rank* would
  /// land on an unrelated permutation, so the kind decides this too.
  void appendNeighbourValues(ParmValue value,
                             llvm::SmallVectorImpl<ParmValue> &out) const;

  /// Retain only values that evenly divide n; converts a range to a ValueList.
  SearchParam &keepDivisorsOf(ParmValue n);
};

/// Factory functions — build a SearchParam without adding it to a space yet.
/// Use ConfigSpace::addDim to register the result.
SearchParam makeRange(StringRef name, ParmValue lo, ParmValue hi,
                      ParmValue step = 1);
SearchParam makePow2Range(StringRef name, ParmValue loExp, ParmValue hiExp);
SearchParam makeValues(StringRef name, std::vector<ParmValue> values);
/// A parameter ranging over the permutations of `[0, n)`, valued by
/// one-based lexicographic rank -- so 1 is the identity and n! the reverse.
SearchParam makePermutation(StringRef name, unsigned n);

/// A concrete assignment — one int64_t per SearchParam, in ConfigSpace order.
using Configuration = std::vector<ParmValue>;

struct ConfigSpace;
struct ConfWrapper;

/// Predicate over a configuration; returns true if the configuration is valid.
using Constraint = std::function<bool(const ConfWrapper)>;

/// A batch of configurations, laid out for vectorized constraint evaluation.
/// Conceptually a D (dims) x N (configs) matrix, but stored as D independent
/// arma::Row buffers — one per dimension — rather than as a single arma::Mat.
/// This matters because arma::Mat is column-major: a single owned matrix
/// would make each *dimension's* values (what a constraint actually slices
/// out via SpaceVar::operator[]) a strided row view, not contiguous memory.
/// Storing one contiguous arma::Row per dimension instead means every slice a
/// constraint operates on is a real contiguous SIMD-friendly buffer.
struct ConfigurationVector {
  std::vector<ParmVector> dims;

  ConfigurationVector(size_t numDims, size_t n) : dims(numDims) {
    for (auto &row : dims)
      row.set_size(n);
  }

  size_t size() const { return dims.empty() ? 0 : dims[0].n_elem; }
  size_t numDims() const { return dims.size(); }

  /// Write configuration `conf` into column `col` of every dimension's row.
  void setColumn(size_t col, const Configuration &conf) {
    for (size_t d = 0; d < dims.size(); ++d)
      dims[d][col] = conf[d];
  }

  const ParmVector &operator[](size_t dimIdx) const { return dims[dimIdx]; }

  ParmVector ones() const { return ParmVector(size(), arma::fill::ones); }
  ParmVector zeros() const { return ParmVector(size(), arma::fill::zeros); }
};

/// Vectorized predicate: evaluates a constraint over a whole batch of
/// configurations at once, AND-ing the second parameter with this
/// constraint's validity result. A constraint is allowed to short
/// circuit and avoid performing configuration for
/// entries of that are already set to zero.
using VecConstraint =
    std::function<void(const ConfigurationVector &c, arma::urowvec &valid)>;

/// Elementwise a / b, yielding 0 where b == 0. Matches the scalar OpDiv
/// (`b ? a / b : 0`) and, more importantly, avoids the UB that plain
/// elementwise division would hit — a vectorized constraint evaluates every
/// lane, so it cannot short-circuit past a zero divisor the way the
/// equivalent scalar predicate does.
inline ParmVector vecSafeDiv(const ParmVector &a, const ParmVector &b) {
  arma::uvec zeros = arma::find(b == 0);
  ParmVector safeB = b;
  safeB.elem(zeros).ones();
  ParmVector q = a / safeB;
  q.elem(zeros).zeros();
  return q;
}

/// Elementwise `b != 0 && a % b == 0` ("b divides a"), zero-safe as above.
inline arma::urowvec vecDivides(const ParmVector &b, const ParmVector &a) {
  arma::uvec zeros = arma::find(b == 0);
  ParmVector safeB = b;
  safeB.elem(zeros).ones();
  // `%` is Armadillo's elementwise multiply, so this is a - (a / b) * b.
  ParmVector rem = a - (a / safeB) % safeB;
  arma::urowvec ok = (rem == 0);
  ok.elem(zeros).zeros();
  return ok;
}

/// A record of how a space came to have the shape it has, carried by the space
/// but never interpreted by it.
///
/// The decisions are taken during planning and are invisible afterwards: the
/// encoding shows what a space *is*, not which constraint was folded into it,
/// which one was left to filter, or which grouping was tried and abandoned. So
/// whoever plans a space attaches its own account of it, and whoever reports on
/// a space asks that account to print itself. Neither has to know the other's
/// vocabulary, which is the point -- planning is the builder's business.
struct SpaceMetadata {
  virtual ~SpaceMetadata() = default;
  /// Print the record as the members of a JSON object -- no enclosing braces,
  /// and a trailing comma is the caller's problem, not this one's.
  virtual void printJSONMembers(std::ostream &os) const = 0;
};

/// Ordered collection of SearchParams that defines the search space.
struct ConfigSpace {
  std::vector<SearchParam> params;
  /// Each constraint paired with a human-readable description of what it
  /// checks (e.g. "wramRow | mramRow"); empty if the constraint was added
  /// without one. Used by debugIsValid() to report violations.
  /// Constraints, stored as their vectorized form only (see VecConstraint and
  /// addConstraint()) -- there is exactly one predicate per constraint, never
  /// a separate scalar/vector pair.
  std::vector<std::pair<std::string, VecConstraint>> constraints;

  /// A set of dimensions whose joint valuation is constrained, with every
  /// satisfying tuple enumerated ahead of time. The component occupies one
  /// slot of the flat index, sized by the number of solutions -- so the
  /// mixed-radix encoding keeps its fixed-size-slot invariant while the
  /// dimensions inside it are never enumerated independently.
  ///
  /// This is what turns a structural constraint from something the search
  /// filters into something it never offers.
  struct SolvedComponent {
    /// Dimensions covered, in the order the tuples store them.
    std::vector<size_t> dims;
    /// Every valid assignment. Order is deterministic (it is the enumeration
    /// order), which is what makes flat indices reproducible across runs.
    std::vector<std::vector<ParmValue>> solutions;
    /// Inverse of `solutions`, for indexOf(). std::map rather than a hash map
    /// so no tuple hash is needed; indexOf is not on a hot path.
    std::map<std::vector<ParmValue>, size_t> indexOfSolution;

    size_t size() const { return solutions.size(); }
  };
  std::vector<SolvedComponent> components;

  /// Register a pre-enumerated component. Dimensions it covers must not be
  /// claimed by any other component.
  void addSolvedComponent(SolvedComponent &&component);

  /// How this space was planned, for reporting. Null unless whoever built it
  /// attached one.
  std::unique_ptr<SpaceMetadata> metadata;

  ConfigSpace() = default;
  ConfigSpace(const ConfigSpace &) = delete;

  /// Add a fully-constructed SearchParam; returns its index in the space.
  size_t addDim(SearchParam &&param) {
    size_t idx = params.size();
    params.push_back(std::move(param));
    encodingValid_ = false;
    return idx;
  }

  std::pair<size_t, size_t> addDims(SmallVector<SearchParam> &&dims) {
    size_t start = params.size();
    for (auto &dim : dims)
      params.push_back(std::move(dim));
    encodingValid_ = false;
    return {start, (int64_t)params.size()};
  }

  /// Register a vectorized predicate; configurations whose lane any
  /// constraint clears to 0 are skipped and never passed to the plugin for
  /// evaluation. `description` is an optional human-readable label, reported
  /// by debugIsValid() when the constraint rejects a configuration.
  ///
  /// Constraints are stored and evaluated only in this vectorized form; the
  /// single-configuration check (isValid) is derived from it by evaluating a
  /// one-lane batch. Prefer this overload — it is the one the search actually
  /// runs, and the scalar view of it is free.
  void addConstraint(VecConstraint &&constraint, std::string description = "");
  /// Register a scalar predicate, vectorized automatically by evaluating it
  /// once per lane of the batch. Convenience for constraints not worth
  /// hand-vectorizing; there is still exactly one predicate registered, never
  /// a scalar/vector pair that could drift out of sync.
  void addConstraint(Constraint &&constraint, std::string description = "");
  /// AND every registered constraint's vectorized mask together (all-ones,
  /// i.e. everything passes, if none are registered).
  arma::urowvec evalVecConstraintsMask(const ConfigurationVector &cv) const;

  size_t size() const { return params.size(); }
  const SearchParam &operator[](size_t i) const { return params[i]; }
  SearchParam &operator[](size_t i) { return params[i]; }

  /// Width of the surrogate's input vector. Not size(): a parameter may
  /// contribute more than one feature (see SearchParam::appendFeatures).
  size_t numFeatures() const;
  /// The surrogate's input vector for `conf`, appended to `out`. Every path
  /// that hands a configuration to the model goes through here, so training
  /// and prediction cannot disagree about the encoding.
  void encode(const Configuration &conf,
              llvm::SmallVectorImpl<double> &out) const;

  /// Index of param with the given name, or -1.
  int findIndex(llvm::StringRef name) const;
  /// Value of the named param in a configuration, or 0 if not found.
  ParmValue get(const Configuration &config, llvm::StringRef name) const;
  /// Return true iff all registered constraints accept this configuration.
  bool isValid(const Configuration &config) const;
  /// Like isValid(), but also prints the configuration and the description
  /// of every violated constraint to `os` (nothing is printed if the
  /// configuration is valid). Returns the same result as isValid().
  bool debugIsValid(const Configuration &config, raw_ostream &os) const;

  /// Total number of configurations reachable by at() -- excludes everything a
  /// component's enumeration ruled out, includes the configurations that are
  /// still to be filtered by isValid().
  size_t totalSize() const;
  /// Fill conf with the configuration at flat index idx.
  /// idx must be in [0, totalSize()). Constraints are NOT checked.
  void at(size_t idx, Configuration &conf) const;
  /// Convert a configuration to its flat index (inverse of at()).
  size_t indexOf(const Configuration &conf) const;
  /// Iterate all configurations in flat-index order, calling fn(conf, flatIdx)
  /// for each. Return false from fn to stop early. Successive calls update only
  /// the suffix of conf that changed (O(1) amortised per step vs O(S) for
  /// at()).
  void forEach(std::function<bool(const Configuration &, size_t)> fn) const;
  /// Like forEach(), but restricted to the flat-index range [lo, hi). Used to
  /// parallelise a scan over the full space while keeping each chunk's
  /// per-step cost O(1) amortised (only the initial config at `lo` costs
  /// O(S), same as at()).
  void
  forEachChunk(size_t lo, size_t hi,
               std::function<bool(const Configuration &, size_t)> fn) const;
  /// Append to result all flat indices one discrete step away in any
  /// dimension. Steps that land on a configuration the encoding does not
  /// offer (because a structural constraint rules it out) are skipped, so a
  /// configuration near the edge of a component has fewer neighbours.
  void neighborIndices(size_t idx, llvm::SmallVectorImpl<size_t> &result) const;
  /// True if `conf` satisfies every structurally-encoded constraint, i.e. if
  /// at()/indexOf() can round-trip it. Configurations produced by at() always
  /// satisfy this; hand-built ones need not.
  ///
  /// This is a different question from isValid(). A constraint folded into the
  /// encoding is *deregistered* as a predicate -- there is no configuration
  /// left for it to reject -- so isValid() says nothing about it, and a
  /// hand-built configuration violating one passes every check while naming a
  /// point the space does not contain.
  bool isEncodable(const Configuration &conf) const;
  /// Like isEncodable(), but reports every reason `conf` cannot be encoded to
  /// `os` (nothing is printed if it can). Returns the same result.
  bool debugIsEncodable(const Configuration &conf, raw_ostream &os) const;

  template <class Out> void dump(Out &out, const Configuration &config) const {
    out << " {";
    for (auto [i, dim, value] : llvm::enumerate(params, config)) {
      out << dim.name << "=" << value << (i + 1 < size() ? ", " : "");
    }
    out << "}";
  }

private:
  /// One slot of the flat index: either a single dimension, or a whole
  /// SolvedComponent whose dimensions are enumerated jointly and therefore
  /// share one slot.
  struct EncodingSlot {
    size_t dimIdx; ///< index into params[]; the component's first dim if any
    size_t componentIdx; ///< index into components[], or SIZE_MAX
    size_t slotSize;     ///< number of distinct sub-indices this slot has
  };

  mutable bool encodingValid_ = false;
  mutable std::vector<EncodingSlot> slots_;
  /// suffixProd_[i] = product of slotSizes[i..end]; suffixProd_[slots_.size()]
  /// = 1.
  mutable std::vector<size_t> suffixProd_;

  void ensureEncoding() const;
};

/// Wrap a space and config for nicer interface.
struct ConfWrapper {
  const ConfigSpace &space;
  const Configuration &conf;
  ConfWrapper(const ConfigSpace &space, const Configuration &conf)
      : space(space), conf(conf) {}

  /// Get the value of a variable
  ParmValue operator[](StringRef name) const { return space.get(conf, name); }
  ParmValue operator[](size_t ix) const { return conf[ix]; }
};

inline raw_ostream &operator<<(raw_ostream &os, const ConfWrapper &wrapper) {
  wrapper.space.dump(os, wrapper.conf);
  return os;
}
inline Diagnostic &operator<<(Diagnostic &os, const ConfWrapper &wrapper) {
  wrapper.space.dump(os, wrapper.conf);
  return os;
}

} // namespace mlir::cinm
