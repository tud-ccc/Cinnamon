#pragma once

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
/// Every parameter is stored as positive integers either way; the kind says
/// how those integers are to be read, and therefore what may be done with
/// them. A magnitude can be added, multiplied, divided and ordered; an
/// ordering can only be compared for equality, since arithmetic on the
/// encoding of an ordering is arithmetic on an arbitrary numbering.
///
/// This enum is the *erased* form of that distinction -- what a SearchParam
/// carries so it can be stored next to parameters of other kinds. The unerased
/// form is the type parameter of SpaceVar, which is where the DSL gets to
/// enforce it at compile time rather than report it at run time.
enum class ParamKind {
  /// A quantity. Tile sizes, counts, capacities.
  Integer,
  /// An ordering of n items. See Permutation, and note that how it is *stored*
  /// is ParmKind<Permutation>'s business and nothing else's.
  Permutation,
};

llvm::StringRef paramKindName(ParamKind kind);

/// An ordering of `[0, n)`: `position[i]` is the place item `i` takes,
/// zero-based, so `position` is the inverse of "which item is at place p".
///
/// This is what a permutation parameter *is*, as opposed to how it is encoded
/// in a Configuration -- see ParmKind<Permutation>. Callers that read one back
/// out of a configuration get this, and never the encoding, which is the point
/// of routing the two through a model.
struct Permutation {
  llvm::SmallVector<int64_t, 4> position;

  bool operator==(const Permutation &o) const = default;
  size_t size() const { return position.size(); }
  unsigned operator[](size_t item) const { return position[item]; }
};

struct SearchParam;

// ===----------------------------------------------------------------------===//
// Parameter models
// ===----------------------------------------------------------------------===//
//
// How a value of some type `T` is stored in a Configuration, shown to the
// surrogate, and stepped to reach a neighbour. Specialise ParmKind<T> to add a
// parameter type; there is no other place to touch, which is the reason the
// trait exists at all -- the alternative is a `switch (kind)` in each of those
// three operations, drifting apart one operation at a time.
//
// Every operation takes the values as a *span*, not a single ParmValue, so
// that a type occupying several dimensions needs no different signature from
// one occupying a single dimension.

/// The model for `T`. Undefined for a type that has none, so declaring a
/// parameter of one is an error at the declaration rather than a link failure.
template <class T> struct ParmKind;

/// What a parameter model has to provide. `Model` is a separate parameter from
/// `T` so that the diagnostic names the missing operation instead of
/// unravelling inside SpaceVar.
template <class Model, class T>
concept ParmModel =
    requires(const SearchParam &param, llvm::ArrayRef<ParmValue> values,
             llvm::SmallVectorImpl<double> &features,
             llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &steps) {
      /// The erased kind, for a SearchParam to carry and reporting to print.
      { Model::kind() } -> std::same_as<ParamKind>;
      /// How many Configuration entries one value of `T` occupies.
      { Model::arity(param) } -> std::convertible_to<size_t>;
      /// The value those entries encode.
      { Model::decode(param, values) } -> std::convertible_to<T>;
      /// Width and content of this parameter's slice of the surrogate's input.
      { Model::numFeatures(param) } -> std::convertible_to<size_t>;
      Model::appendFeatures(param, values, features);
      /// The encodings one discrete step away, each a full span.
      Model::appendNeighbours(param, values, steps);
    };

/// The model, erased. A SearchParam holds one of these rather than a `T`,
/// because the space stores parameters of different types side by side; it is
/// filled in from ParmKind<T> so the two paths cannot disagree.
struct ParmVTable {
  ParamKind kind;
  size_t (*arity)(const SearchParam &);
  size_t (*numFeatures)(const SearchParam &);
  void (*appendFeatures)(const SearchParam &, llvm::ArrayRef<ParmValue>,
                         llvm::SmallVectorImpl<double> &);
  void (*appendNeighbours)(
      const SearchParam &, llvm::ArrayRef<ParmValue>,
      llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &);

  template <class T, class Model = ParmKind<T>>
    requires ParmModel<Model, T>
  static const ParmVTable *of() {
    static const ParmVTable table = {
        Model::kind(), &Model::arity, &Model::numFeatures,
        &Model::appendFeatures, &Model::appendNeighbours};
    return &table;
  }
};

/// One parameter of the search space, with its type erased model trait
/// instance. The generic type SpaceVar<T> has the higher-level API used to
/// build constraints.
struct SearchParam {
  std::string name;
  std::variant<IntRange, ValueList> domain;
  /// Never null: every constructor takes one, so there is no valid state in
  /// which a parameter does not know what it is.
  const ParmVTable *model;
  /// For ParmKind::Permutation: how many items are permuted. What the domain
  /// holds is the model's business and not something to recover n from.
  unsigned permutationSize = 0;

  SearchParam(const SearchParam &) = delete;
  SearchParam(SearchParam &&) = default;
  // The model is not defaulted: the factories below are the way a parameter is
  // built, and each of them knows its own type. A default would also have to
  // name ParmKind<ParmValue> before it is defined.
  SearchParam(StringRef name, IntRange &&range, const ParmVTable *model)
      : name(name.str()), domain(std::move(range)), model(model) {}
  SearchParam(StringRef name, ValueList &&list, const ParmVTable *model)
      : name(name.str()), domain(std::move(list)), model(model) {}
  SearchParam &operator=(SearchParam &&o) = default;

  ParamKind kind() const { return model->kind; }
  /// How many Configuration entries this parameter occupies.
  size_t arity() const { return model->arity(*this); }
  /// Name of the `k`-th of them, which is the parameter's own name when there
  /// is only one. Every per-dimension listing -- a CSV header, a diagnostic
  /// about an out-of-domain value -- goes through here, so a parameter
  /// spanning several dimensions names them consistently.
  std::string dimName(size_t k) const;

  /// Smallest and largest value one dimension of this parameter can take. For
  /// reporting; the surrogate sees appendFeatures() instead, and nothing maps
  /// back from a feature to a value.
  double dlo() const;
  double dhi() const;
  /// Number of distinct values *one dimension* of this parameter can take,
  /// which is the size of `domain`. Not the number of values the parameter
  /// has: an ordering of n items has n! of those and a domain of n, and the
  /// difference is exactly what the solver's distinctness constraint removes.
  size_t cardinality() const;

  /// How many surrogate features this parameter contributes. One for a
  /// quantity; a permutation of n items contributes n.
  size_t numFeatures() const { return model->numFeatures(*this); }
  /// Append this parameter's features for `values` to `out`.
  ///
  /// Two things are going on, and they are both about what the surrogate can
  /// learn rather than about the parameter itself.
  ///
  /// **A permutation contributes its position vector**: feature `i` is the
  /// place item `i` takes. Euclidean distance between two such vectors is
  /// Spearman's rank distance, so two orders that agree about most items land
  /// near each other and the network can generalise between them. A rank
  /// cannot do that -- consecutive ranks are unrelated permutations -- and one
  /// feature per permutation could not either, since no single number carries
  /// the structure.
  ///
  /// **Every feature is scaled to [0, 1]** against the parameter's declared
  /// domain, so that features are comparable to each other. The scaling has to
  /// come from the domain rather than from the sample, because training and
  /// prediction encode different sets of configurations and a model fitted on
  /// one scale cannot be asked about another.
  void appendFeatures(llvm::ArrayRef<ParmValue> values,
                      llvm::SmallVectorImpl<double> &out) const {
    model->appendFeatures(*this, values, out);
  }

  /// Append the encodings one step from `values`, for whatever "one step"
  /// means for this parameter: the adjacent values of a quantity, and the
  /// adjacent transpositions of an ordering. Each entry is a full span, so a
  /// step is allowed to move several dimensions at once -- which it must, for
  /// a parameter whose encoding spans more than one.
  void appendNeighbours(
      llvm::ArrayRef<ParmValue> values,
      llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out) const {
    model->appendNeighbours(*this, values, out);
  }

  /// Return the i-th distinct value of this parameter (0-indexed).
  ParmValue valueAt(size_t subIdx) const;
  /// Return the sub-index of value within this parameter's domain (inverse of
  /// valueAt). Meaningful only for a value the domain contains -- see
  /// contains(), which is how a caller holding an arbitrary value checks.
  size_t subIndexOf(ParmValue value) const;
  /// Whether `value` is one this parameter can take. A value outside the
  /// domain has no sub-index, so subIndexOf() cannot report this itself.
  bool contains(ParmValue value) const;

  /// Retain only values that evenly divide n; converts a range to a ValueList.
  SearchParam &keepDivisorsOf(ParmValue n);
};

// ===----------------------------------------------------------------------===//
// The models
// ===----------------------------------------------------------------------===//

/// A quantity: one dimension, one feature, stepped to its adjacent values.
template <> struct ParmKind<ParmValue> {
  static ParamKind kind() { return ParamKind::Integer; }
  static size_t arity(const SearchParam &) { return 1; }
  static ParmValue decode(const SearchParam &, llvm::ArrayRef<ParmValue> v) {
    return v[0];
  }
  static size_t numFeatures(const SearchParam &) { return 1; }
  static void appendFeatures(const SearchParam &param,
                             llvm::ArrayRef<ParmValue> values,
                             llvm::SmallVectorImpl<double> &out);
  static void
  appendNeighbours(const SearchParam &param, llvm::ArrayRef<ParmValue> values,
                   llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out);
};

/// An ordering of n items, encoded positionally: n dimensions, dimension `i`
/// holding the one-based place item `i` takes.
///
/// One-based because §2 of SpaceBuilder.h has every domain strictly positive.
/// What makes the n dimensions an ordering rather than n independent numbers
/// is a distinctness constraint, which the solver posts for every parameter of
/// this kind (see ConstraintGecode.cpp) -- so it holds by construction and no
/// caller writes it.
///
/// The alternative encoding, a lexicographic rank in a single dimension, is
/// the one this replaced. A rank is compact but opaque: every constraint about
/// where an item sits has to be stated about the whole ordering at once, which
/// in practice means an opaque predicate that decodes it. Positions cost n-1
/// extra dimensions and make "these two items share an axis" a comparison
/// between two variables.
///
/// That the encoding is positions is stated here and nowhere else.
template <> struct ParmKind<Permutation> {
  static ParamKind kind() { return ParamKind::Permutation; }
  static size_t arity(const SearchParam &param) {
    return param.permutationSize;
  }
  static Permutation decode(const SearchParam &param,
                            llvm::ArrayRef<ParmValue> values);
  static size_t numFeatures(const SearchParam &param) {
    return param.permutationSize;
  }
  static void appendFeatures(const SearchParam &param,
                             llvm::ArrayRef<ParmValue> values,
                             llvm::SmallVectorImpl<double> &out);
  static void
  appendNeighbours(const SearchParam &param, llvm::ArrayRef<ParmValue> values,
                   llvm::SmallVectorImpl<llvm::SmallVector<ParmValue, 4>> &out);
};

/// Factory functions — build a SearchParam without adding it to a space yet.
/// Use ConfigSpace::addParam to register the result.
SearchParam makeRange(StringRef name, ParmValue lo, ParmValue hi,
                      ParmValue step = 1);
SearchParam makePow2Range(StringRef name, ParmValue loExp, ParmValue hiExp);
SearchParam makeValues(StringRef name, std::vector<ParmValue> values);
/// A parameter ranging over the orderings of `[0, n)`, occupying n dimensions
/// of `[1, n]` -- see ParmKind<Permutation>. On its own this describes n
/// independent numbers; what makes it an ordering is the distinctness the
/// solver posts for it.
SearchParam makePermutation(StringRef name, unsigned n);

/// A concrete assignment — one entry per *dimension*, in ConfigSpace order,
/// which is one entry per parameter only when every parameter has arity one.
using Configuration = std::vector<ParmValue>;

struct ConfigSpace;
struct ConfWrapper;

/// Predicate over a configuration; returns true if the configuration is valid.
using Constraint = std::function<bool(const ConfWrapper)>;

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
///
/// **Parameters and dimensions are not the same count.** A parameter occupies
/// `arity()` consecutive entries of a Configuration, which is one for a
/// quantity and n for an ordering of n items. Everything indexed by parameter
/// says `param`; everything indexed by Configuration entry says `dim`; and
/// `dimOffset()` / `paramAtDim()` are the two ways between them. The two used
/// to be the same number, and this is the distinction that was implicit then.
struct ConfigSpace {
  std::vector<SearchParam> params;
  /// Each constraint paired with a human-readable description of what it
  /// checks (e.g. "wramRow | mramRow"); empty if the constraint was added
  /// without one. Used by debugIsValid() to report violations.
  std::vector<std::pair<std::string, Constraint>> constraints;

  /// Install the configurations this space contains.
  ///
  /// `solutions` must be sorted lexicographically and hold no duplicates --
  /// which is what constraints::solveSpace returns -- because the flat index
  /// *is* a position in it and indexOf() binary-searches it.
  void setSolutions(std::vector<Configuration> &&solutions);

  /// How this space was planned, for reporting. Null unless whoever built it
  /// attached one.
  std::unique_ptr<SpaceMetadata> metadata;

  ConfigSpace() = default;
  ConfigSpace(const ConfigSpace &) = delete;

  /// Add a fully-constructed SearchParam; returns the index of its *first
  /// dimension*, which is what a handle onto it has to hold -- that is the
  /// index a constraint reads.
  size_t addParam(SearchParam &&param);

  /// Register a predicate; configurations it rejects are never passed to the
  /// plugin for evaluation. `description` is an optional human-readable label,
  /// reported by debugIsValid() when the constraint rejects a configuration.
  void addConstraint(Constraint &&constraint, std::string description = "");

  /// How many parameters were declared.
  size_t numParams() const { return params.size(); }
  /// How many entries a Configuration of this space has. Not numParams():
  /// see the note above the class.
  size_t numDims() const { return paramOfDim_.size(); }
  /// First Configuration entry of parameter `param`.
  size_t dimOffset(size_t param) const { return firstDimOfParam_[param]; }
  /// The parameter that Configuration entry `dim` belongs to.
  const SearchParam &paramAtDim(size_t dim) const {
    return params[paramOfDim_[dim]];
  }
  /// Name of Configuration entry `dim`, which is its parameter's name when
  /// that parameter has only this one.
  std::string dimName(size_t dim) const {
    return paramAtDim(dim).dimName(dim - firstDimOfParam_[paramOfDim_[dim]]);
  }
  /// The entries of `conf` belonging to parameter `param`.
  llvm::ArrayRef<ParmValue> paramValues(const Configuration &conf,
                                        size_t param) const {
    return llvm::ArrayRef(conf).slice(firstDimOfParam_[param],
                                      params[param].arity());
  }

  const SearchParam &operator[](size_t param) const { return params[param]; }
  SearchParam &operator[](size_t param) { return params[param]; }

  /// Width of the surrogate's input vector. Neither numParams() nor numDims():
  /// a parameter may contribute a different number of features from either
  /// (see SearchParam::appendFeatures).
  size_t numFeatures() const;
  /// The surrogate's input vector for `conf`, appended to `out`. Every path
  /// that hands a configuration to the model goes through here, so training
  /// and prediction cannot disagree about the encoding.
  void encode(const Configuration &conf,
              llvm::SmallVectorImpl<double> &out) const;

  /// Index of the parameter with the given name, or -1.
  int findParam(llvm::StringRef name) const;
  /// Value of the named parameter in a configuration, or 0 if it has no
  /// parameter of that name. Only for a parameter occupying one dimension --
  /// a value spanning several is not a ParmValue, and asking for one is a
  /// mistake rather than something to truncate; use getAs<T>() for those.
  ParmValue get(const Configuration &config, llvm::StringRef name) const;
  /// The value of the named parameter, decoded as a `T`.
  ///
  /// This is how a caller that has only the parameter's *name* -- because it
  /// crossed an interface carrying strings, as a stamped attribute does --
  /// reads a value whose encoding it must not know.
  template <class T, class Model = ParmKind<T>>
    requires ParmModel<Model, T>
  T getAs(const Configuration &config, llvm::StringRef name) const {
    int param = findParam(name);
    assert(param >= 0 && "no parameter of that name");
    assert(params[param].kind() == Model::kind() && "wrong parameter kind");
    return Model::decode(params[param], paramValues(config, param));
  }
  /// Return true iff all registered constraints accept this configuration.
  bool isValid(const Configuration &config) const;
  /// Like isValid(), but also prints the configuration and the description
  /// of every violated constraint to `os` (nothing is printed if the
  /// configuration is valid). Returns the same result as isValid().
  bool debugIsValid(const Configuration &config, raw_ostream &os) const;

  /// Number of configurations the space contains -- every one the solver
  /// found, which is every one satisfying the constraints it could be given.
  /// What is left to filter is whatever was registered as an opaque predicate;
  /// see isValid().
  size_t totalSize() const;
  /// Fill conf with the configuration at flat index idx.
  /// idx must be in [0, totalSize()). Constraints are NOT checked.
  void at(size_t idx, Configuration &conf) const;
  /// Convert a configuration to its flat index (inverse of at()).
  ///
  /// The configurations are sorted, so this is a binary search rather than a
  /// side table. A configuration the space does not contain has no index; ask
  /// isEncodable() first if that is in doubt.
  size_t indexOf(const Configuration &conf) const;
  /// Iterate all configurations in flat-index order, calling fn(conf, flatIdx)
  /// for each. Return false from fn to stop early.
  void forEach(std::function<bool(const Configuration &, size_t)> fn) const;
  /// Like forEach(), but restricted to the flat-index range [lo, hi). Used to
  /// parallelise a scan over the whole space.
  void
  forEachChunk(size_t lo, size_t hi,
               std::function<bool(const Configuration &, size_t)> fn) const;
  /// Append to result all flat indices one discrete step away in any
  /// dimension. A step that lands on a configuration the space does not
  /// contain is skipped, so a configuration near the edge of the feasible set
  /// has fewer neighbours.
  void neighborIndices(size_t idx, llvm::SmallVectorImpl<size_t> &result) const;
  /// True if `conf` is one of the configurations this space contains, i.e. if
  /// at()/indexOf() can round-trip it. Configurations produced by at() always
  /// satisfy this; hand-built ones need not.
  ///
  /// This is a different question from isValid(). A constraint the solver
  /// enforced has no configuration left to reject, so isValid() says nothing
  /// about it, and a hand-built configuration violating one passes every check
  /// while naming a point the space does not contain.
  bool isEncodable(const Configuration &conf) const;
  /// Like isEncodable(), but reports every reason `conf` is not in the space to
  /// `os` (nothing is printed if it is). Returns the same result.
  bool debugIsEncodable(const Configuration &conf, raw_ostream &os) const;

  template <class Out> void dump(Out &out, const Configuration &config) const {
    out << " {";
    for (auto [dim, value] : llvm::enumerate(config))
      out << dimName(dim) << "=" << value
          << (dim + 1 < config.size() ? ", " : "");
    out << "}";
  }

private:
  /// Every configuration in the space, sorted lexicographically. The flat
  /// index is a position in here, which is why this is the one thing the space
  /// requires to be sorted.
  std::vector<Configuration> solutions_;

  /// The two directions between a parameter index and a Configuration entry.
  /// Both are maintained by addParam() and neither is derivable from `params`
  /// without a scan, which is why they are stored rather than computed.
  std::vector<size_t> firstDimOfParam_;
  std::vector<size_t> paramOfDim_;
};

/// Wrap a space and config for nicer interface.
struct ConfWrapper {
  const ConfigSpace &space;
  const Configuration &conf;
  ConfWrapper(const ConfigSpace &space, const Configuration &conf)
      : space(space), conf(conf) {}

  /// Get the value of a single-dimension parameter by name.
  ParmValue operator[](StringRef name) const { return space.get(conf, name); }
  /// Get one Configuration entry, by *dimension* index.
  ParmValue operator[](size_t dim) const { return conf[dim]; }
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
