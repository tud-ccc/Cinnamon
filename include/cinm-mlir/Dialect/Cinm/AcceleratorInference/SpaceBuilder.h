#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Declaring a design space
// ===----------------------------------------------------------------------===//
//
// How a space is *built*: parameters are declared, constraints are written in
// the DSL below, and buildInto() commits both at once. See the top of
// ConfigSpace.h for what a space is and how the pieces fit together.
//
// ===----------------------------------------------------------------------===//

// ===----------------------------------------------------------------------===//
// SpaceVar — lazy typed handle to a named search-space parameter
// ===----------------------------------------------------------------------===//

template <constraints::Type Ty> class Expr;

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
/// A read returns by value and a Permutation allocates, so a loop over many
/// configurations wants `node()` and the constraint evaluator rather than a
/// typed read per configuration.
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
    const SearchParam &param = c.space.paramAtDim(*idx_);
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

  /// The place item `item` takes, as a constraint-IR expression: a
  /// **one-based** workgroup axis, with 1 the outermost.
  ///
  /// This is the one thing an ordering exposes to the DSL, and it is what
  /// makes a constraint about orderings writable at all. "These two ops put
  /// the same value dimension on the same axis" is `a.axis(i) == b.axis(j)`
  /// -- a comparison between two variables the solver propagates, where under
  /// a rank encoding it was an opaque predicate decoding both sides per
  /// configuration.
  ///
  /// The result is a quantity, deliberately: the places are ordinals, and
  /// `<` between two of them is exactly what "outer than" means. It is an
  /// Expr and not a bare node because two bare nodes are two shared_ptrs, and
  /// `==` between those is pointer comparison -- which compiles, and is never
  /// what the caller meant.
  Expr<constraints::Type::INT> axis(unsigned item) const
    requires(std::is_same_v<T, Permutation>);

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

template <class T, class Model>
  requires ParmModel<Model, T>
IntExpr SpaceVar<T, Model>::axis(unsigned item) const
  requires(std::is_same_v<T, Permutation>)
{
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      name_ + "[" + std::to_string(item) + "]", idx_, item));
}

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

/// A truth value as the number 0 or 1, so that conditions can be counted:
/// `sum([asInt(c_0), asInt(c_1), ...])` is how many of them hold.
///
/// This is the only way back from a truth value to a number, and it is what a
/// constraint over "how many dimensions satisfy X" is written with. A division
/// inside `b` is discharged inside `b` -- it makes this expression 0, not the
/// comparison containing it false.
inline IntExpr asInt(BoolExpr b) {
  return IntExpr(std::make_shared<constraints::ConstraintNode>(
      constraints::ConstraintNode::Kind::BoolAsInt, b.node()));
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

  /// Declare a parameter ranging over the orderings of the *active* items --
  /// those whose expression holds in a given configuration -- with the
  /// inactive ones taking no place at all.
  ///
  /// Which items are active is a property of the configuration, not of the
  /// declaration: `active[i]` is an expression over other parameters. So the
  /// number of distinct orderings varies from configuration to configuration,
  /// and the constraints that make the parameter mean one thing per
  /// configuration are posted here rather than written by the caller. They are
  /// stated in terms of the encoding, which is exactly what a caller must not
  /// have to know: the active items take the low places, the inactive ones the
  /// high places in index order, and the distinctness is the solver's. Between
  /// them these leave exactly (number active)! assignments per configuration,
  /// which is the point -- a configuration that orders k items has k! ways to
  /// do it and no duplicates of any of them.
  ///
  /// `active.size()` is the number of items.
  PermVar permutation(llvm::StringRef name, llvm::ArrayRef<BoolExpr> active);
  /// Declare a parameter whose values are exactly the divisors of n.
  IntVar divisorsOf(llvm::StringRef name, ParmValue n);
  /// Declare a parameter in [1, v.maxVal()] with the constraint that its
  /// values must divide the runtime value of v (v % result == 0).
  IntVar divisorsOf(llvm::StringRef name, IntVar v);

  /// Arbitrary predicate; configurations it rejects are dropped from the space
  /// once the solver has enumerated it. `description` is optional; it names the
  /// predicate in the space's report. Prefer the Expr overload where the
  /// constraint can be written in the DSL — only that form is analysable, and
  /// only it prunes the search rather than filtering its output.
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

  /// Pin the declared integer parameter `name` to exactly `value`. This is
  /// how a caller *outside* the plugin conditions the space on a decision
  /// taken at a higher level -- the graph level pinning the device size for
  /// a Stage-A profiling run or a Stage-C budgeted search
  /// (docs/GraphOptimizationDesign.md) -- without the plugin having to know.
  /// Returns false if no integer parameter of that name has been declared;
  /// the caller must treat that as an error, since a search that silently
  /// ignores a pin measures something other than what was asked.
  bool pin(llvm::StringRef name, ParmValue value);

  /// Commit all declarations and constraints into space in the correct order.
  /// The space also comes away with a SpaceMetadata recording what planning
  /// decided, since none of that is recoverable from the result.
  void buildInto(ConfigSpace &space, unsigned nWorkers = 1);

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
  /// Exactly one of the two is set: a constraint written in the DSL is a tree
  /// the solver is given, and one registered as a lambda is a filter run over
  /// its output.
  struct PredicateEntry {
    std::string description;
    Constraint pred;
    constraints::ConstraintNodePtr node;
  };

  std::vector<DimEntry> dims_;
  std::vector<PredicateEntry> predicates_;

  DimEntry &findEntry(const IntVar &v);
  IntVar findVarByName(llvm::StringRef name) const;
  int dimIndexByName(llvm::StringRef name) const;

  /// Walk `node` and reify every Div as a divisibility constraint.
  void extractDivConstraints(const constraints::ConstraintNodePtr &node);
};

} // namespace mlir::cinm
