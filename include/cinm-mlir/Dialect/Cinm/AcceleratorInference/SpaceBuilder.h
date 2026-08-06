#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
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
// This file is the reference for how a design space is described, planned and
// encoded. It documents the state of the implementation, not a target to build
// towards.
//
// # 1. What a design space is
//
// A *search parameter* is a named integer variable with a finite domain. A
// *configuration* is one value per parameter, in declaration order. The design
// space is the set of configurations that satisfy every registered
// *constraint*.
//
// Parameters and constraints are not written by hand per kernel: an
// InferencePlugin derives them from the IR of one compute block, so the space
// describes the lowerings that block admits. `SpaceBuilder` is the interface it
// derives them through.
//
// Two objects, with different jobs:
//
//   SpaceBuilder   declaration and planning. Collects parameters and
//                  constraints, decides how each constraint is enforced, and
//                  commits the result exactly once, in buildInto().
//   ConfigSpace    encoding and filtering. Holds the parameters, the flat
//                  index, and the predicates that survived planning.
//
// Nothing is committed until buildInto(): declaration order does not have to
// match dependency order, and a constraint may mention a parameter declared
// after it.
//
// # 2. Parameters
//
// **Every domain is a set of strictly positive integers**, which is checked at
// declaration. Several things depend on it: the interval reasoning treats
// products as monotone in every factor, division is meaningful, and assigning
// a variable always tightens what the unassigned ones can contribute to a
// capacity bound. A parameter that is naturally an index — the rank of a
// permutation, a mode selector — is declared one-based, and converted where it
// is consumed.
//
// | declaration                  | domain                                    |
// |------------------------------|-------------------------------------------|
// | intRange(n, lo, hi)          | lo, lo+1, ..., hi                         |
// | pow2Range(n, a, b)           | 2^a, ..., 2^b                             |
// | divisorsOf(n, k)             | the divisors of the constant k            |
// | divisorsOf(n, v)             | [1, v.maxVal()], plus `v % result == 0`   |
// | permutation(n, k)            | 1, ..., k! — a rank, see below            |
//
// A domain is either a contiguous range or an explicit value list; a range
// becomes a list as soon as a static filter narrows it (SearchParam::
// keepDivisorsOf). Note the asymmetry in the `divisorsOf` rows: of a
// *constant* it narrows the domain at declaration time, while of another
// *parameter* it cannot — the divisibility depends on a value not known until
// enumeration — so it declares the full range and records a constraint
// instead. A parameter's declared cardinality is therefore an upper bound on
// how many values it can actually take.
//
// Independently of how its domain is stored, a parameter has a *kind* saying
// how its values are to be read (`ParamKind`). `Integer` is a quantity.
// `Permutation` is the lexicographic rank of a permutation, which is a
// numbering: arithmetic and ordering on it are arithmetic and ordering on an
// arbitrary enumeration, so the DSL rejects them (§3) and only `==` and `!=`
// are available. The encoding itself is in cinm-mlir/Utils/Permutation.h,
// shared with the lowering that reads what the search stamped — note that the
// parameter is one-based and the encoding is zero-based, so whoever consumes a
// rank converts.
//
// The kind also decides how the surrogate sees a value: a parameter
// contributes `numFeatures()` features, scaled to [0, 1], and a permutation
// contributes its position vector rather than its rank, so that distance
// between feature vectors is Spearman's rank distance. See
// SearchParam::appendFeatures.
//
// # 3. Constraints
//
// Constraints are written in an embedded DSL (see Expr below) that builds a
// constraint-IR tree: constants, parameters, n-ary sums and products, exact
// division, the six comparisons, `divides`, and `implies`. There is no
// subtraction and no disjunction. Everything the DSL can build is analysable;
// a predicate that cannot be expressed in it can still be registered as an
// opaque C++ lambda, which is then enforced but never reasoned about.
//
// `a / b` means *exact* division: it denotes the quotient and asserts that `b`
// divides `a`. A comparison containing an inexact division evaluates to false
// rather than comparing a truncated quotient. `divides(b, a)` tests the same
// property without asserting it, which is the spelling to use under a guard.
//
// Every operator except `==` and `!=` requires operands of kind `Integer`, and
// aborts otherwise. The check runs where the expression is built — while the
// space is being declared, at the line that wrote it — because a rank compared
// with `<` is a mistake about what the parameter means, not a configuration
// that fails.
//
// require() classifies each tree by shape and enforces it in the strongest
// form available:
//
//   const / var       static domain filter (drop the values that cannot work)
//   var / var         structural relation (recorded, resolved by planning)
//   product equality  structural relation
//   anything else     dynamic predicate over the whole space
//
// Every `/` in a tree also contributes its divisibility assertion, except
// under an `implies` — what is reified there is unconditional, so it would
// constrain the configurations the guard exists to exclude.
//
// # 4. The encoding
//
// A configuration has an integer index. The index is an identity, used to
// cache costs, communicate points between threads, sample uniformly, and walk
// the space deterministically; it is not a representation, and code wanting
// structure should decode, work on the Configuration, and re-encode.
//
// The index is mixed-radix over *slots*, each of a fixed size. A slot is
// either one parameter, sized by its cardinality, or a whole SolvedComponent
// -- a set of parameters with every satisfying tuple enumerated ahead of time
// -- sized by the tuple count. There is no third form: a divisibility pair is
// a component of two parameters, not a case of its own.
//
// So `ConfigSpace::totalSize()` is the number of *addressable* configurations,
// which is the product of the slot sizes — not the Cartesian product of the
// parameter domains. The two differ by whatever the structural constraints
// removed, which is several orders of magnitude in practice. A configuration
// that no slot offers has no index at all (isEncodable() is the test).
//
// Enumeration order inside a component is deterministic, which is what makes
// indices reproducible across runs, and therefore seeded sampling
// reproducible.
//
// # 5. Planning
//
// buildInto() runs in phases:
//
//   1. Materialise each parameter, apply its static filters, add it to the
//      space. Only now does a parameter have an index, so analysis cannot run
//      before this point.
//   2. Partition the parameters and enumerate the components (below). Anything
//      a component absorbs is dropped from the later phases: the encoding can
//      no longer offer a configuration violating it.
//   3. Register the surviving predicates on the space -- the constraints no
//      component could absorb, plus every opaque lambda.
//
// The space also comes away with a record of what was decided (SpaceMetadata),
// since the encoding shows what a space is and not why.
//
// The partition is connected components over the parameters, with an edge for
// every structural relation, plus an edge between an implication's guard and
// its consequent so that a guard does not keep a slot of its own. Each
// component is then enumerated by backtracking:
//
//   - The variable order is chosen against the current partial assignment, not
//     fixed: a variable some equality already determines, then a variable some
//     implication is guarded on, then a variable participating in an equality
//     (narrowest domain first), then the rest. A static order by domain size
//     defers exactly the variables that prune.
//   - Candidates are narrowed before they are tried: a determined variable
//     offers one value, a variable dividing an assigned dividend offers that
//     dividend's divisors, everything else offers its domain.
//   - Comparisons and implications over the component's variables prune
//     subtrees by interval arithmetic, and are exact at a full assignment
//     (every interval is a point), so the component enforces them outright
//     rather than leaving them as filters. A gated equality determines a
//     variable only once its guard *must* hold; acting on one that merely
//     *may* hold would impose the consequent on completions the constraint
//     says nothing about.
//
// Two budgets bound the work: a cap on the number of tuples and a cap on the
// number of search nodes. Exceeding either abandons the component entirely —
// it absorbs nothing, and every relation in it falls back to the paths above.
// There is no intermediate outcome, and no cost model deciding whether merging
// two parameters into a component is worth it.
//
// # 6. What is left over
//
// The encoding offers a superset of the feasible set. What remains is
// filtered, once per configuration, by the predicates registered in phase 4:
// constraints spanning two components, shapes planning could not use, and
// every opaque lambda. `CandidatePool::computeValidMask` evaluates them over
// the whole space in vectorized batches; the result is the feasible set the
// search actually explores.
//
// ===----------------------------------------------------------------------===//

// ===----------------------------------------------------------------------===//
// SpaceVar — lazy handle to a named search-space dimension
// ===----------------------------------------------------------------------===//

/// A handle to a named search-space dimension created by SpaceBuilder.
/// The index into ConfigSpace is written lazily when buildInto() is called;
/// all handles remain valid and return the correct index after that point.
/// A SpaceVar converts to an Expr, so it can be used directly in arithmetic
/// and comparison expressions.
class SpaceVar {
public:
  /// The cell must be allocated up front, even though the index is not known
  /// until buildInto(): every copy of this handle shares it (that is how they
  /// all see the index once it is written), and findEntry() uses the cell's
  /// address as the handle's identity.
  SpaceVar() : idx_(std::make_shared<size_t>(kUnassigned)), maxVal_(0) {}

  llvm::StringRef name() const { return name_; }
  /// How this dimension's values are to be read -- which decides what the DSL
  /// lets them be written into.
  ParamKind kind() const { return kind_; }
  /// Index in the ConfigSpace — valid only after SpaceBuilder::buildInto().
  size_t idx() const { return *idx_; }
  ParmValue get(const ConfWrapper &c) const { return c[*idx_]; }
  ParmValue operator[](const ConfWrapper &c) const { return get(c); }
  /// Vectorized form: the contiguous row of this dimension's values across
  /// every configuration in the batch.
  const ParmVector &operator[](const ConfigurationVector &c) const {
    return c[*idx_];
  }
  /// Upper bound of this variable's domain. Used by divisorsOf(name, SpaceVar).
  ParmValue maxVal() const { return maxVal_; }

  /// This dimension as a constraint-IR node.
  constraints::ConstraintNodePtr node() const {
    return std::make_shared<constraints::ConstraintNode>(name_, idx_, kind_);
  }

private:
  friend class SpaceBuilder;
  explicit SpaceVar(llvm::StringRef name, ParmValue maxVal,
                    ParamKind kind = ParamKind::Integer)
      : name_(name.str()), idx_(std::make_shared<size_t>(kUnassigned)),
        maxVal_(maxVal), kind_(kind) {}

  /// Sentinel held by idx_ until buildInto() writes the real index.
  static constexpr size_t kUnassigned = SIZE_MAX;

  std::string name_;
  std::shared_ptr<size_t> idx_;
  ParmValue maxVal_;
  ParamKind kind_ = ParamKind::Integer;
};

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
  Expr(const SpaceVar &v) : node_(v.node()) {}
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
template <class T, constraints::Type Ty>
inline constexpr bool isExprLike =
    std::is_same_v<std::decay_t<T>, Expr<Ty>> ||
    std::is_same_v<std::decay_t<T>, SpaceVar> ||
    std::is_same_v<std::decay_t<T>, constraints::ConstraintNodePtr>;

template <class A, class B, constraints::Type Ty>
inline constexpr bool eitherIsExpr = isExprLike<A, Ty> || isExprLike<B, Ty>;

/// Abort if either operand mentions a parameter whose values are a numbering
/// rather than a quantity. `op` names the operation for the message.
///
/// A permutation rank is the clear case: `order * 2` and `order < 3` are both
/// arithmetic on an arbitrary enumeration order, and mean nothing about the
/// permutations they name. Only `==` and `!=` survive, which is why they are
/// the only two operators that do not call this.
///
/// The check is at expression-construction time, so it fires while the space
/// is being declared -- at the line that wrote the expression, before any
/// configuration exists.
void assertArithmeticOperands(const constraints::ConstraintNodePtr &lhs,
                              const constraints::ConstraintNodePtr &rhs,
                              llvm::StringRef op);
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
    detail::assertArithmeticOperands(lhs, rhs, #SYM);                          \
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
  detail::assertArithmeticOperands(lhs, rhs, "*");
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Mul, lhs, rhs));
}
template <class A, class B,
          std::enable_if_t<detail::eitherIsExpr<A, B, constraints::Type::INT>,
                           int> = 0>
IntExpr operator+(const A &a, const B &b) {
  auto lhs = IntExpr(a).node();
  auto rhs = IntExpr(b).node();
  detail::assertArithmeticOperands(lhs, rhs, "+");
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::Add, lhs, rhs));
}

/// `CHECKED` selects whether the comparison is one a numbering supports.
/// Equality and inequality are; ordering is not, since the order of the ranks
/// is not an order on what they name.
#define CINM_DEFINE_CMP_OP(SYM, KIND, CHECKED)                                 \
  template <class A, class B,                                                  \
            std::enable_if_t<                                                  \
                detail::eitherIsExpr<A, B, constraints::Type::INT>, int> = 0>  \
  BoolExpr operator SYM(const A &a, const B &b) {                              \
    auto lhs = IntExpr(a).node();                                              \
    auto rhs = IntExpr(b).node();                                              \
    if (CHECKED)                                                               \
      detail::assertArithmeticOperands(lhs, rhs, #SYM);                        \
    return BoolExpr(std::make_shared<constraints::ConstraintNode>(             \
        constraints::ConstraintNode::Kind::KIND, lhs, rhs));                   \
  }

CINM_DEFINE_CMP_OP(<=, Le, true)
CINM_DEFINE_CMP_OP(>=, Ge, true)
CINM_DEFINE_CMP_OP(<, Lt, true)
CINM_DEFINE_CMP_OP(>, Gt, true)
CINM_DEFINE_CMP_OP(==, Eq, false)
CINM_DEFINE_CMP_OP(!=, Ne, false)
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
    detail::assertArithmeticOperands(f.node(), f.node(), "prod");
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
    detail::assertArithmeticOperands(t.node(), t.node(), "sum");
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
  detail::assertArithmeticOperands(divisor.node(), dividend.node(), "divides");
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
  /// Declare a dimension with integer range [lo, hi] (inclusive, step 1).
  SpaceVar intRange(llvm::StringRef name, ParmValue lo, ParmValue hi);
  /// Declare a dimension with values 2^expLo, ..., 2^expHi.
  SpaceVar pow2Range(llvm::StringRef name, ParmValue expLo, ParmValue expHi);
  /// Declare a dimension ranging over the permutations of `[0, n)`, valued by
  /// one-based lexicographic rank (so 1 is the identity). The DSL rejects
  /// arithmetic and ordering on the result; decode it with
  /// cinm-mlir/Utils/Permutation.h, remembering the offset.
  SpaceVar permutation(llvm::StringRef name, unsigned n);
  /// Declare a dimension whose values are exactly the divisors of n.
  SpaceVar divisorsOf(llvm::StringRef name, ParmValue n);
  /// Declare a dimension in [1, v.maxVal()] with the constraint that its
  /// values must divide the runtime value of v (v % result == 0).
  SpaceVar divisorsOf(llvm::StringRef name, SpaceVar v);

  /// Static filter: retain only values of v that are divisors of n.
  void mustDivide(SpaceVar v, ParmValue n);
  /// Structural constraint: child must be a multiple of parent (parent divides
  /// child; child % parent == 0). Applied after all dims are added.
  void mustDivide(SpaceVar parent, SpaceVar child);
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
    SpaceVar var;
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

  DimEntry &findEntry(const SpaceVar &v);
  SpaceVar findVarByName(llvm::StringRef name) const;
  int dimIndexByName(llvm::StringRef name) const;

  /// Fold structural constraints into the index encoding: group variables
  /// linked by divisibility or product relations into connected components,
  /// enumerate each component's satisfying tuples, and register them with
  /// `space` so those configurations are never offered in the first place.
  ///
  /// Relations a component absorbs are reported back through the two output
  /// sets, so the dynamic-predicate phase skips them. A component whose
  /// enumeration exceeds a budget absorbs nothing; its divisibility relations
  /// are then retried one at a time, as components of two parameters, and only
  /// what fails that too is left to a predicate.
  void planComponents(
      ConfigSpace &space, PlanMetadata &report,
      std::set<std::pair<std::string, std::string>> &absorbedMultiples,
      std::set<const constraints::ConstraintNode *> &absorbedPredicates);

  /// Report what the identity recogniser makes of each DSL-registered
  /// constraint. Analysis only — it does not change the space. Runs after phase
  /// 1 of buildInto, since variable indices are unassigned before that.
  void reportConstraintAnalysis(const ConfigSpace &space) const;

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
