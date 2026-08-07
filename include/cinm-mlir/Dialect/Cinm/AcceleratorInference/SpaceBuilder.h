#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"
#include <armadillo>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// The design space
// ===----------------------------------------------------------------------===//
//
// This is the reference for how a design space is described and solved. It
// documents the state of the implementation, not a target to build towards.
//
// The space itself is three files: `ConfigSpace.h` is what a space *is* --
// parameters, contents, predicates -- this one is how one is *built*, and
// `AcceleratorInference.h` is what is done with one.
//
// # 1. What a design space is
//
// A *search parameter* is a named variable with a finite domain. A
// *configuration* is one value per parameter, in declaration order. The design
// space is the set of configurations that satisfy every constraint -- and it is
// that set literally: `buildInto()` enumerates it with a finite-domain solver
// and the space holds the result. There is no superset to filter down from,
// except for whatever was registered as an opaque predicate (§5).
//
// Parameters and constraints are not written by hand per kernel: an
// InferencePlugin derives them from the IR of one compute block, so the space
// describes the lowerings that block admits. `SpaceBuilder` is the interface it
// derives them through.
//
// Two objects, with different jobs:
//
//   SpaceBuilder   declaration. Collects parameters and constraints, and
//                  commits them exactly once, in buildInto().
//   ConfigSpace    contents. Holds the parameters, the configurations, and the
//                  predicates that could not be given to the solver.
//
// Nothing is committed until buildInto(): declaration order does not have to
// match dependency order, and a constraint may mention a parameter declared
// after it.
//
// # 2. Parameters
//
// **Every domain is a set of strictly positive integers**, which is checked at
// declaration. Division is meaningful on them, products are monotone in every
// factor, and no propagator has to reason about a sign.
//
// | declaration                  | domain                                    |
// |------------------------------|-------------------------------------------|
// | intRange(n, lo, hi)          | lo, lo+1, ..., hi                         |
// | pow2Range(n, a, b)           | 2^a, ..., 2^b                             |
// | divisorsOf(n, k)             | the divisors of the constant k            |
// | divisorsOf(n, v)             | [1, v.maxVal()], plus `v % result == 0`   |
// | permutation(n, k)            | the orderings of [0, k)                   |
//
// A domain is either a contiguous range or an explicit value list; a range
// becomes a list as soon as a static filter narrows it (SearchParam::
// keepDivisorsOf). Note the asymmetry in the `divisorsOf` rows: of a
// *constant* it narrows the domain at declaration time, while of another
// *parameter* it cannot -- the divisibility depends on a value not known until
// the solve -- so it declares the full range and records a constraint instead.
//
// **A parameter is typed.** `intRange` returns an `IntVar` and `permutation` a
// `PermVar`, and the type is what the value reads back as: an ordering comes
// back as a `Permutation`, never as whatever integers encode it. The encoding
// is `ParmKind<T>`'s business and nothing else's -- one specialisation states
// how a `T` is stored, how it is shown to the surrogate, and what one step from
// it is. Adding a parameter type is adding a specialisation, and changing how
// an existing one is stored is changing four functions with no caller affected.
//
// The surrogate consequence is worth naming: a permutation contributes its
// position vector rather than an index into an enumeration of permutations, so
// that distance between feature vectors is Spearman's rank distance and two
// orderings that agree about most items land near each other. See
// SearchParam::appendFeatures.
//
// # 3. Constraints
//
// Constraints are written in an embedded DSL (see Expr below) that builds a
// constraint-IR tree: constants, parameters, n-ary sums and products, exact
// division, the six comparisons, `divides`, and `implies`. There is no
// subtraction and no disjunction. A predicate that cannot be expressed in it
// can still be registered as an opaque C++ lambda, which is then enforced but
// never given to the solver.
//
// `a / b` means *exact* division: it denotes the quotient and asserts that `b`
// divides `a`. A comparison containing an inexact division is false rather than
// a comparison of a truncated quotient. `divides(b, a)` tests the same property
// without asserting it, which is the spelling to use under a guard.
//
// **The DSL only does arithmetic on quantities, and the type system is what
// says so.** Only `IntVar` converts to an expression, so `order * 2` and
// `order < 3` do not compile -- there is no viable operator, reported at the
// line that wrote them. This used to be a run-time abort during declaration,
// which was the best a single untyped handle could manage.
//
// # 4. Solving
//
// buildInto() runs in three steps:
//
//   1. Materialise each parameter, apply its static filters, add it to the
//      space. Only now does a parameter have an index.
//   2. Translate every DSL constraint into a finite-domain model and enumerate
//      every solution (`ConstraintGecode.h`). The configurations come back
//      sorted, and the space holds them.
//   3. Register the predicates on the space.
//
// There is no partition to choose, no component to enumerate, no classification
// of a constraint as static, structural or dynamic, and no budget deciding
// between them. A constraint is either expressible in the IR, in which case the
// solver enforces it, or it is an opaque lambda.
//
// The space comes away with a record of what happened (SpaceMetadata): what the
// solver was given, what it cost, and how the result compares to the Cartesian
// product of the domains.
//
// A configuration has an integer index, which is its position in the sorted
// list. The index is an identity -- used to cache costs, communicate points
// between threads, sample uniformly, and walk the space deterministically --
// and not a representation: code wanting structure should decode, work on the
// Configuration, and re-encode. Because the list is sorted rather than in
// discovery order, the index is a property of the space and not of the search
// that produced it, so changing the branching heuristic or the thread count
// renumbers nothing.
//
// # 5. What is left over
//
// Only the opaque lambdas, which `CandidatePool::computeValidMask` evaluates
// over the space in vectorized batches. The DSL-derived predicates stay
// registered too, but not because anything is left for them to reject: they are
// what `ConfigSpace::debugIsValid` uses to say *why* a hand-built configuration
// is not in the space, and their agreeing with the solver is the standing check
// on the translation. A space whose valid count is below its size means the two
// disagree, and that is a bug in ConstraintGecode.cpp.
//
// ===----------------------------------------------------------------------===//

// ===----------------------------------------------------------------------===//
// SpaceVar — lazy typed handle to a named search-space parameter
// ===----------------------------------------------------------------------===//

/// A handle to a named parameter created by SpaceBuilder, carrying the type of
/// the value it stands for.
///
/// The index into ConfigSpace is written lazily when buildInto() is called; all
/// handles remain valid and return the correct index after that point.
///
/// **`T` is the type read back, not the encoding.** `operator[]` returns a `T`
/// -- an ordering comes back as a Permutation, whatever the dimensions behind
/// it hold -- because `Model` decodes it. Nothing outside ParmKind<T> needs to
/// know how a value is stored, which is what lets the encoding change without
/// a caller moving.
///
/// **`T` is also what the DSL type-checks against.** Only a handle whose values
/// are a quantity converts to IntExpr, so `order * 2` and `order < 3` do not
/// compile. They used to abort at run time, at the line that wrote them, which
/// was the best a single erased handle type could do.
///
/// The typed read is for cold paths -- stamping, lowering, reporting. It
/// returns by value and a Permutation allocates, so a per-lane loop wants the
/// ConfigurationVector overload, which stays untyped and per-dimension because
/// that is the shape vectorized evaluation needs.
template <class T, class Model = ParmKind<T>>
  requires ParmModel<Model, T>
class SpaceVar {
public:
  using ValueType = T;

  /// The cell must be allocated up front, even though the index is not known
  /// until buildInto(): every copy of this handle shares it (that is how they
  /// all see the index once it is written), and findEntry() uses the cell's
  /// address as the handle's identity.
  SpaceVar() : idx_(std::make_shared<size_t>(kUnassigned)), maxVal_(0) {}

  llvm::StringRef name() const { return name_; }
  /// The erased kind, for callers that have lost the type -- reporting, mostly.
  static ParamKind kind() { return Model::kind(); }
  /// Index of this parameter's first dimension in the ConfigSpace — valid only
  /// after SpaceBuilder::buildInto().
  size_t idx() const { return *idx_; }

  /// The value this parameter takes in `c`.
  T get(const ConfWrapper &c) const {
    const SearchParam &param = c.space[*idx_];
    return Model::decode(
        param, llvm::ArrayRef(c.conf).slice(*idx_, Model::arity(param)));
  }
  T operator[](const ConfWrapper &c) const { return get(c); }

  /// Upper bound of this variable's domain. Used by divisorsOf(name, SpaceVar).
  ParmValue maxVal() const { return maxVal_; }

  /// This parameter as a constraint-IR node. Only meaningful for a parameter
  /// occupying one dimension and read as a number, which is what the DSL's
  /// conversion to IntExpr already restricts it to.
  constraints::ConstraintNodePtr node() const {
    return std::make_shared<constraints::ConstraintNode>(name_, idx_);
  }

  /// Vectorized form: the contiguous row of this parameter's values across
  /// every configuration in the batch. Untyped by design -- see the note above
  /// -- and available only for a quantity, where the encoding *is* the value.
  const ParmVector &operator[](const ConfigurationVector &c) const
    requires(std::is_same_v<T, ParmValue>)
  {
    return c[*idx_];
  }

  /// The raw first entry of this parameter's encoding.
  ///
  /// These two read the *encoding*, not the value, and are therefore only
  /// correct for a model of arity one -- which is not something the type says,
  /// so a caller is on its own.
  ///
  /// They exist for a predicate that has to reinterpret the encoding rather
  /// than read the value it denotes. The order parameters are the case:
  /// their rank ranks only the dimensions a configuration actually
  /// distributes, so what it means depends on the tile sizes, and decoding it
  /// against the parameter's declared size -- which is what get() does -- would
  /// be decoding a different permutation. Needing this is the signal that the
  /// parameter wants modelling properly instead.
  const ParmVector &encodedRow(const ConfigurationVector &c) const {
    assert(c.numDims() > *idx_ && "parameter index out of range");
    return c[*idx_];
  }
  ParmValue encodedValue(const ConfWrapper &c) const { return c[*idx_]; }

private:
  friend class SpaceBuilder;
  explicit SpaceVar(llvm::StringRef name, ParmValue maxVal)
      : name_(name.str()), idx_(std::make_shared<size_t>(kUnassigned)),
        maxVal_(maxVal) {}
  /// Rebuild a handle onto an already-declared parameter, sharing its cell.
  SpaceVar(llvm::StringRef name, ParmValue maxVal, std::shared_ptr<size_t> idx)
      : name_(name.str()), idx_(std::move(idx)), maxVal_(maxVal) {}

  /// Sentinel held by idx_ until buildInto() writes the real index.
  static constexpr size_t kUnassigned = SIZE_MAX;

  std::string name_;
  std::shared_ptr<size_t> idx_;
  ParmValue maxVal_;
};

/// A quantity — a tile size, a count, a capacity. The overwhelmingly common
/// case, and the only one the DSL does arithmetic on.
using IntVar = SpaceVar<ParmValue>;
/// An ordering of n items.
using PermVar = SpaceVar<Permutation>;

// ===----------------------------------------------------------------------===//
// Expr — DSL handle wrapping a constraint-IR node
// ===----------------------------------------------------------------------===//

/// Thin wrapper around a ConstraintNodePtr. It exists so the DSL operators can
/// be defined without colliding with the ones std::shared_ptr already has
/// (notably operator==, which would otherwise mean pointer comparison).
///
/// It also adds compile-time type-checking to the expression system.
template <constraints::Type Ty> class Expr {
public:
  Expr(constraints::ConstraintNodePtr node) : node_(std::move(node)) {}
  /// Only a handle whose values are a quantity becomes an expression. This is
  /// the constraint that used to be a run-time abort: `order * 2` now fails to
  /// convert rather than failing to run.
  Expr(const IntVar &v) : node_(v.node()) {}
  Expr(ParmValue v) : node_(std::make_shared<constraints::ConstraintNode>(v)) {}

  const constraints::ConstraintNodePtr &node() const { return node_; }
  std::string describe() const { return describeNode(*node_); }

private:
  constraints::ConstraintNodePtr node_;
};

using IntExpr = Expr<constraints::Type::INT>;
using BoolExpr = Expr<constraints::Type::BOOL>;

namespace detail {
/// Enables the operators below only when at least one side is a search-space
/// expression, so they never hijack plain integer arithmetic.
/// Enables the operators below only when at least one side is a search-space
/// expression, so they never hijack plain integer arithmetic.
///
/// IntVar and not SpaceVar<T>: a handle to something that is not a quantity is
/// deliberately not expression-like, so an operator over it is not found at
/// all. That is the whole of the kind checking the DSL used to do at run time
/// -- `order < 3` reports no viable operator, at the line that wrote it,
/// without a configuration ever existing.
template <class T, constraints::Type Ty>
inline constexpr bool isExprLike =
    std::is_same_v<std::decay_t<T>, Expr<Ty>> ||
    std::is_same_v<std::decay_t<T>, IntVar> ||
    std::is_same_v<std::decay_t<T>, constraints::ConstraintNodePtr>;

template <class A, class B, constraints::Type Ty>
inline constexpr bool eitherIsExpr = isExprLike<A, Ty> || isExprLike<B, Ty>;
} // namespace detail

// ===----------------------------------------------------------------------===//
// Arithmetic and comparison operators
// ===----------------------------------------------------------------------===//

#define CINM_DEFINE_BIN_OP(SYM, KIND, TY)                                      \
  template <class A, class B,                                                  \
            std::enable_if_t<detail::eitherIsExpr<A, B, TY>, int> = 0>         \
  Expr<TY> operator SYM(const A &a, const B &b) {                              \
    auto lhs = Expr<TY>(a).node();                                             \
    auto rhs = Expr<TY>(b).node();                                             \
    return Expr<TY>(std::make_shared<constraints::ConstraintNode>(             \
        constraints::ConstraintNode::Kind::KIND, lhs, rhs));                   \
  }

/// Division also *asserts* that the divisor divides the dividend exactly:
/// SpaceBuilder::require() extracts every `/` as a static, structural, or
/// dynamic divisibility constraint.
CINM_DEFINE_BIN_OP(/, Div, constraints::Type::INT)
#undef CINM_DEFINE_BIN_OP

/// Add and Mul are n-ary in the IR; the binary operators build a two-operand
/// node, and prod()/sum() build a flat one.
template <class A, class B,
          std::enable_if_t<detail::eitherIsExpr<A, B, constraints::Type::INT>,
                           int> = 0>
IntExpr operator*(const A &a, const B &b) {
  auto lhs = IntExpr(a).node();
  auto rhs = IntExpr(b).node();
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Mul, lhs, rhs));
}
template <class A, class B,
          std::enable_if_t<detail::eitherIsExpr<A, B, constraints::Type::INT>,
                           int> = 0>
IntExpr operator+(const A &a, const B &b) {
  auto lhs = IntExpr(a).node();
  auto rhs = IntExpr(b).node();
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Add, lhs, rhs));
}

/// Every comparison takes quantities, because only IntVar is expression-like.
///
/// `==` and `!=` used to be exempt from the kind check, so that two orderings
/// could be compared even though `<` on them meant nothing. Nothing writes
/// that today -- an agreement between two orderings is stated per item, not
/// between their encodings -- so equality on a non-quantity stays out until a
/// caller for it exists, and ParmKind<T> is where it would go when one does.
#define CINM_DEFINE_CMP_OP(SYM, KIND)                                          \
  template <class A, class B,                                                  \
            std::enable_if_t<                                                  \
                detail::eitherIsExpr<A, B, constraints::Type::INT>, int> = 0>  \
  BoolExpr operator SYM(const A &a, const B &b) {                              \
    auto lhs = IntExpr(a).node();                                              \
    auto rhs = IntExpr(b).node();                                              \
    return BoolExpr(std::make_shared<constraints::ConstraintNode>(             \
        constraints::ConstraintNode::Kind::KIND, lhs, rhs));                   \
  }

CINM_DEFINE_CMP_OP(<=, Le)
CINM_DEFINE_CMP_OP(>=, Ge)
CINM_DEFINE_CMP_OP(<, Lt)
CINM_DEFINE_CMP_OP(>, Gt)
CINM_DEFINE_CMP_OP(==, Eq)
CINM_DEFINE_CMP_OP(!=, Ne)
#undef CINM_DEFINE_CMP_OP

/// Flat n-ary product. This is the shape the analyser wants: one node with a
/// child per factor, rather than a left-leaning tree it would have to
/// re-flatten. Arity is a runtime value (e.g. the number of iteration
/// dimensions), which is exactly what a type-level encoding could not express.
inline IntExpr prod(llvm::ArrayRef<IntExpr> factors) {
  if (factors.empty())
    return IntExpr(std::make_shared<constraints::ConstraintNode>(1));
  llvm::SmallVector<constraints::ConstraintNodePtr, 2> ops;
  for (const IntExpr &f : factors) {
    ops.push_back(f.node());
  }
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Mul, std::move(ops)));
}

/// Flat n-ary sum; see prod().
inline IntExpr sum(llvm::ArrayRef<IntExpr> terms) {
  if (terms.empty())
    return IntExpr(std::make_shared<constraints::ConstraintNode>(0));
  llvm::SmallVector<constraints::ConstraintNodePtr, 2> ops;
  for (const IntExpr &t : terms) {
    ops.push_back(t.node());
  }
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Add, std::move(ops)));
}

/// Evaluates to the truth value of "`divisor` divides `dividend`".
/// This can be used as the guard of an implies() node.
///
/// This operator is here because `b.require(a / b == 1, X)` would
/// otherwise be ambiguous. Is it that b structurally divides a, and
/// a == b implies X, or is the antecedent equivalent to the rewritten
/// `a == b`, with no div constraint?
/// With this operator, both variants are expressible:
/// ```cpp
/// // Without structural constraint:
/// b.require(implies(divides(b, a), X));
///
/// // With structural constraint:
/// b.require(implies(a / b == 1, X));
/// // which is equivalent to
/// b.require(divides(b, a));
/// b.require(implies(a == b, X));
/// ```
inline BoolExpr divides(IntExpr divisor, IntExpr dividend) {
  return BoolExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Divides, divisor.node(),
      dividend.node()));
}

/// `antecedent => consequent`: the consequent is required only of the
/// configurations the antecedent selects.
///
/// This is used to implement conditional constraints on the design space.
/// Note that the `/` operator implies divisibility structurally in the
/// antecedent, but not in the consequent. If you want to write a condition
/// "when a / b == 2, X" without requiring that b divide a always, you need to
/// nest implications:
/// ```
/// implies(divides(a, b), implies(a / b == 2, X))
/// ```
inline BoolExpr implies(BoolExpr antecedent, BoolExpr consequent) {
  return BoolExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Implies, antecedent.node(),
      consequent.node()));
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder
// ===----------------------------------------------------------------------===//

/// Declarative builder for ConfigSpace.
///
/// Declare dimensions and constraints, then commit everything in dependency
/// order via buildInto() (static filters → addDim → multiples constraints →
/// dynamic predicates). All SpaceVar handles remain valid after buildInto().
/// Duplicate static and structural constraints are silently deduplicated.
///
/// Naming convention for mustDivide(parent, child):
///   "parent divides child" = child % parent == 0.
class SpaceBuilder {
public:
  /// Declare a parameter with integer range [lo, hi] (inclusive, step 1).
  IntVar intRange(llvm::StringRef name, ParmValue lo, ParmValue hi);
  /// Declare a parameter with values 2^expLo, ..., 2^expHi.
  IntVar pow2Range(llvm::StringRef name, ParmValue expLo, ParmValue expHi);
  /// Declare a parameter ranging over the orderings of `[0, n)`.
  ///
  /// The result reads back as a Permutation, so a caller never sees the
  /// encoding; the DSL will not do arithmetic on it, and there is nothing to
  /// decode by hand. How it is stored is ParmKind<Permutation>'s business.
  PermVar permutation(llvm::StringRef name, unsigned n);
  /// Declare a parameter whose values are exactly the divisors of n.
  IntVar divisorsOf(llvm::StringRef name, ParmValue n);
  /// Declare a parameter in [1, v.maxVal()] with the constraint that its
  /// values must divide the runtime value of v (v % result == 0).
  IntVar divisorsOf(llvm::StringRef name, IntVar v);

  /// Static filter: retain only values of v that are divisors of n.
  void mustDivide(IntVar v, ParmValue n);
  /// Structural constraint: child must be a multiple of parent (parent divides
  /// child; child % parent == 0). Applied after all dims are added.
  void mustDivide(IntVar parent, IntVar child);
  /// Arbitrary vectorized predicate; configurations whose lane it clears to 0
  /// are skipped by the framework. `description` is optional; it is reported
  /// by ConfigSpace::debugIsValid() when the predicate rejects a
  /// configuration. Prefer the Expr overload where the constraint can be
  /// written in the DSL — only that form is analysable.
  void require(VecConstraint pred, llvm::StringRef description = "");
  /// Same, for a scalar predicate — vectorized automatically by evaluating it
  /// once per configuration in the batch. Convenience for predicates not
  /// worth hand-vectorizing.
  void require(Constraint pred, llvm::StringRef description = "");

  /// Require that the given boolean expression evaluate to true.
  /// Note that all division expressions are recursively found and
  /// structurally assert that the division is exact. The only
  /// exception is on the right-hand-side of an [implies] node.
  ///
  /// `description` defaults to the rendered expression.
  void require(BoolExpr expr, llvm::StringRef description = "") {
    require(expr.node(), description);
  }

  /// Commit all declarations and constraints into space in the correct order.
  /// The space also comes away with a SpaceMetadata recording what planning
  /// decided, since none of that is recoverable from the result.
  void buildInto(ConfigSpace &space);

private:
  /// What planning did, accumulated as it happens. Defined in the .cpp: it is
  /// a report about this class's decisions and nothing else needs the type.
  struct PlanMetadata;
  void require(const constraints::ConstraintNodePtr &expr,
               llvm::StringRef description = "");

  struct DimEntry {
    /// The name and the index cell, not a handle: a declaration is the same
    /// record whatever type the handle it was returned through has.
    std::string name;
    std::shared_ptr<size_t> idx;
    enum Kind { IntRange, Pow2, DivisorsOfConst, Permutation } kind;
    ParmValue lo, hi;
    std::vector<ParmValue> divisorFilters; ///< keepDivisorsOf(n) for each n
    unsigned permutationSize = 0;          ///< for Kind::Permutation
  };

  struct MultiplesEntry {
    std::string parent, child;
    bool operator==(const MultiplesEntry &o) const {
      return parent == o.parent && child == o.child;
    }
    bool operator<(const MultiplesEntry &o) const {
      return std::tie(parent, child) < std::tie(o.parent, o.child);
    }
  };

  struct PredicateEntry {
    std::string description;
    /// Exactly one form — whichever require() overload was called. A scalar
    /// predicate is vectorized by ConfigSpace::addConstraint at buildInto
    /// time rather than here, because wrapping it needs a ConfWrapper and so
    /// the ConfigSpace, which does not exist yet when require() runs.
    std::variant<Constraint, VecConstraint> pred;
    /// The source expression, for constraints registered through the DSL;
    /// null for opaque predicates. Only these can be analysed.
    constraints::ConstraintNodePtr node;
  };

  std::vector<DimEntry> dims_;
  std::vector<MultiplesEntry> multiples_;
  std::vector<PredicateEntry> predicates_;

  DimEntry &findEntry(const IntVar &v);
  IntVar findVarByName(llvm::StringRef name) const;
  int dimIndexByName(llvm::StringRef name) const;

  /// Walk `node` and reify every Div as a divisibility constraint.
  void extractDivConstraints(const constraints::ConstraintNodePtr &node);
  /// Reify a single num/den divisibility constraint found on a Div node:
  ///  - const / var   → static filter on den's values
  ///  - var / var     → structural mustDivide
  ///  - everything else → dynamic predicate
  void addDivConstraint(const constraints::ConstraintNodePtr &num,
                        const constraints::ConstraintNodePtr &den);
};

} // namespace mlir::cinm
