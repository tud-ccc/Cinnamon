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
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace mlir::cinm {

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
  ConstraintNodePtr node() const { return makeVarNode(idx_, name_); }

private:
  friend class SpaceBuilder;
  explicit SpaceVar(llvm::StringRef name, ParmValue maxVal)
      : name_(name.str()), idx_(std::make_shared<size_t>(kUnassigned)),
        maxVal_(maxVal) {}

  /// Sentinel held by idx_ until buildInto() writes the real index.
  static constexpr size_t kUnassigned = SIZE_MAX;

  std::string name_;
  std::shared_ptr<size_t> idx_;
  ParmValue maxVal_;
};

// ===----------------------------------------------------------------------===//
// Expr — DSL handle wrapping a constraint-IR node
// ===----------------------------------------------------------------------===//

/// Thin wrapper around a ConstraintNodePtr. It exists so the DSL operators can
/// be defined without colliding with the ones std::shared_ptr already has
/// (notably operator==, which would otherwise mean pointer comparison).
class Expr {
public:
  Expr(ConstraintNodePtr node) : node_(std::move(node)) {}
  Expr(const SpaceVar &v) : node_(v.node()) {}
  Expr(ParmValue v) : node_(makeConstNode(v)) {}

  const ConstraintNodePtr &node() const { return node_; }
  std::string describe() const { return describeNode(*node_); }

private:
  ConstraintNodePtr node_;
};

namespace detail {
/// Enables the operators below only when at least one side is a search-space
/// expression, so they never hijack plain integer arithmetic.
template <class T>
inline constexpr bool isExprLike =
    std::is_same_v<std::decay_t<T>, Expr> ||
    std::is_same_v<std::decay_t<T>, SpaceVar> ||
    std::is_same_v<std::decay_t<T>, ConstraintNodePtr>;

template <class A, class B>
inline constexpr bool eitherIsExpr = isExprLike<A> || isExprLike<B>;
} // namespace detail

// ===----------------------------------------------------------------------===//
// Arithmetic and comparison operators
// ===----------------------------------------------------------------------===//

#define CINM_DEFINE_BIN_OP(SYM, KIND)                                          \
  template <class A, class B,                                                  \
            std::enable_if_t<detail::eitherIsExpr<A, B>, int> = 0>             \
  Expr operator SYM(const A &a, const B &b) {                                  \
    return Expr(makeBinNode(ConstraintNode::Kind::KIND, Expr(a).node(),        \
                            Expr(b).node()));                                  \
  }

/// Division also *asserts* that the divisor divides the dividend exactly:
/// SpaceBuilder::require() extracts every `/` as a static, structural, or
/// dynamic divisibility constraint.
CINM_DEFINE_BIN_OP(/, Div)
CINM_DEFINE_BIN_OP(-, Sub)
#undef CINM_DEFINE_BIN_OP

/// Add and Mul are n-ary in the IR; the binary operators build a two-operand
/// node, and prod()/sum() build a flat one.
template <class A, class B,
          std::enable_if_t<detail::eitherIsExpr<A, B>, int> = 0>
Expr operator*(const A &a, const B &b) {
  return Expr(makeNaryNode(ConstraintNode::Kind::Mul,
                           {Expr(a).node(), Expr(b).node()}));
}
template <class A, class B,
          std::enable_if_t<detail::eitherIsExpr<A, B>, int> = 0>
Expr operator+(const A &a, const B &b) {
  return Expr(makeNaryNode(ConstraintNode::Kind::Add,
                           {Expr(a).node(), Expr(b).node()}));
}

#define CINM_DEFINE_CMP_OP(SYM, KIND)                                          \
  template <class A, class B,                                                  \
            std::enable_if_t<detail::eitherIsExpr<A, B>, int> = 0>             \
  Expr operator SYM(const A &a, const B &b) {                                  \
    return Expr(makeCmpNode(CmpKind::KIND, Expr(a).node(), Expr(b).node()));   \
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
inline Expr prod(llvm::ArrayRef<Expr> factors) {
  if (factors.empty())
    return Expr(makeConstNode(1));
  llvm::SmallVector<ConstraintNodePtr, 2> ops;
  for (const Expr &f : factors)
    ops.push_back(f.node());
  return Expr(makeNaryNode(ConstraintNode::Kind::Mul, std::move(ops)));
}

/// Flat n-ary sum; see prod().
inline Expr sum(llvm::ArrayRef<Expr> terms) {
  if (terms.empty())
    return Expr(makeConstNode(0));
  llvm::SmallVector<ConstraintNodePtr, 2> ops;
  for (const Expr &t : terms)
    ops.push_back(t.node());
  return Expr(makeNaryNode(ConstraintNode::Kind::Add, std::move(ops)));
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

  /// Register a constraint written in the DSL.
  ///
  /// Every `/` in the expression is extracted as a divisibility constraint —
  /// static filter, structural multiples constraint, or dynamic predicate,
  /// whichever the operand shapes allow (see addDivConstraint). If the
  /// expression is a comparison it is additionally registered as a predicate,
  /// evaluated by the constraint-IR interpreter; a bare arithmetic expression
  /// contributes only its divisibility conditions.
  ///
  /// `description` defaults to the rendered expression.
  void require(Expr expr, llvm::StringRef description = "");

  /// Commit all declarations and constraints into space in the correct order.
  void buildInto(ConfigSpace &space);

private:
  struct DimEntry {
    SpaceVar var;
    enum Kind { IntRange, Pow2, DivisorsOfConst } kind;
    ParmValue lo, hi;
    std::vector<ParmValue> divisorFilters; ///< keepDivisorsOf(n) for each n
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
    ConstraintNodePtr node;
  };

  std::vector<DimEntry> dims_;
  std::vector<MultiplesEntry> multiples_;
  std::vector<PredicateEntry> predicates_;

  DimEntry &findEntry(const SpaceVar &v);
  SpaceVar findVarByName(llvm::StringRef name) const;
  int dimIndexByName(llvm::StringRef name) const;

  /// Report what the Form A recogniser makes of each DSL-registered constraint.
  /// Analysis only — it does not change the space. Runs after phase 1 of
  /// buildInto, since variable indices are unassigned before that.
  void reportConstraintAnalysis(const ConfigSpace &space) const;

  /// Walk `node` and reify every Div as a divisibility constraint.
  void extractDivConstraints(const ConstraintNodePtr &node);
  /// Reify a single num/den divisibility constraint found on a Div node:
  ///  - const / var   → static filter on den's values
  ///  - var / var     → structural mustDivide
  ///  - everything else → dynamic predicate
  void addDivConstraint(const ConstraintNodePtr &num,
                        const ConstraintNodePtr &den);
};

} // namespace mlir::cinm
