#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include <armadillo>
#include <cstdint>
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
// SpaceExprBase — CRTP base for all space expressions
// ===----------------------------------------------------------------------===//

template <typename Derived> struct SpaceExprBase {
  ParmValue eval(const ConfWrapper &c) const {
    return static_cast<const Derived &>(*this).evalImpl(c);
  }
  /// Vectorized eval: one value per configuration in the batch. `decltype`d
  /// so leaf nodes can hand back a reference to the batch's existing row
  /// instead of copying it; interior nodes return a fresh row by value.
  decltype(auto) evalVec(const ConfigurationVector &c) const {
    return static_cast<const Derived &>(*this).evalVecImpl(c);
  }
  /// Human-readable rendering of this expression, e.g. "(wramRow * tasklets)".
  /// Used to give debugIsValid() a description of violated constraints.
  std::string describe() const {
    return static_cast<const Derived &>(*this).describeImpl();
  }
};

// ===----------------------------------------------------------------------===//
// SpaceVar — lazy handle to a named search-space dimension
// ===----------------------------------------------------------------------===//

/// A handle to a named search-space dimension created by SpaceBuilder.
/// The index into ConfigSpace is written lazily when buildInto() is called;
/// all handles remain valid and return the correct index after that point.
/// SpaceVar is also a SpaceExprBase<SpaceVar>, so it can be used directly
/// in arithmetic and comparison expressions.
class SpaceVar : public SpaceExprBase<SpaceVar> {
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

  ParmValue evalImpl(const ConfWrapper &c) const { return get(c); }
  const ParmVector &evalVecImpl(const ConfigurationVector &c) const {
    return c[*idx_];
  }
  std::string describeImpl() const { return name_; }

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
// ConstExpr — compile-time integer constant
// ===----------------------------------------------------------------------===//

struct ConstExpr : SpaceExprBase<ConstExpr> {
  ParmValue value;
  explicit ConstExpr(ParmValue v) : value(v) {}
  ParmValue evalImpl(const ConfWrapper &) const { return value; }
  ParmVector evalVecImpl(const ConfigurationVector &c) const {
    ParmVector r(c.size());
    r.fill(value);
    return r;
  }
  std::string describeImpl() const { return std::to_string(value); }
};

// ===----------------------------------------------------------------------===//
// BinExpr — binary arithmetic node
// ===----------------------------------------------------------------------===//

namespace detail {
struct OpAdd {
  static ParmValue apply(ParmValue a, ParmValue b) { return a + b; }
  static ParmVector applyVec(const ParmVector &a,
                                     const ParmVector &b) {
    return a + b;
  }
  static const char *symbol() { return " + "; }
};
struct OpSub {
  static ParmValue apply(ParmValue a, ParmValue b) { return a - b; }
  static ParmVector applyVec(const ParmVector &a,
                                     const ParmVector &b) {
    return a - b;
  }
  static const char *symbol() { return " - "; }
};
struct OpMul {
  static ParmValue apply(ParmValue a, ParmValue b) { return a * b; }
  static ParmVector applyVec(const ParmVector &a,
                                     const ParmVector &b) {
    return a % b; // `%` is Armadillo's elementwise multiply.
  }
  static const char *symbol() { return " * "; }
};
struct OpDiv {
  static ParmValue apply(ParmValue a, ParmValue b) { return b ? a / b : 0; }
  static ParmVector applyVec(const ParmVector &a,
                                     const ParmVector &b) {
    return vecSafeDiv(a, b);
  }
  static const char *symbol() { return " / "; }
};
} // namespace detail

template <typename L, typename R, typename Op>
struct BinExpr : SpaceExprBase<BinExpr<L, R, Op>> {
  L lhs;
  R rhs;
  BinExpr(L l, R r) : lhs(std::move(l)), rhs(std::move(r)) {}
  ParmValue evalImpl(const ConfWrapper &c) const {
    return Op::apply(lhs.eval(c), rhs.eval(c));
  }
  ParmVector evalVecImpl(const ConfigurationVector &c) const {
    return Op::applyVec(lhs.evalVec(c), rhs.evalVec(c));
  }
  std::string describeImpl() const {
    return "(" + lhs.describe() + Op::symbol() + rhs.describe() + ")";
  }
};

template <typename L, typename R> using AddExpr = BinExpr<L, R, detail::OpAdd>;
template <typename L, typename R> using SubExpr = BinExpr<L, R, detail::OpSub>;
template <typename L, typename R> using MulExpr = BinExpr<L, R, detail::OpMul>;
template <typename L, typename R> using DivExpr = BinExpr<L, R, detail::OpDiv>;

namespace detail {
template <typename T> struct is_div_expr : std::false_type {};
template <typename L, typename R>
struct is_div_expr<DivExpr<L, R>> : std::true_type {};
template <typename T> constexpr bool is_div_expr_v = is_div_expr<T>::value;

template <typename T> struct is_bin_expr : std::false_type {};
template <typename L, typename R, typename Op>
struct is_bin_expr<BinExpr<L, R, Op>> : std::true_type {};
template <typename T> constexpr bool is_bin_expr_v = is_bin_expr<T>::value;

template <typename T> struct is_mul_expr : std::false_type {};
template <typename L, typename R>
struct is_mul_expr<MulExpr<L, R>> : std::true_type {};
template <typename T> constexpr bool is_mul_expr_v = is_mul_expr<T>::value;
} // namespace detail

// ===----------------------------------------------------------------------===//
// Arithmetic operators
// ===----------------------------------------------------------------------===//

// Expr op Expr
template <typename L, typename R>
auto operator+(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return AddExpr<L, R>(static_cast<const L &>(l), static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator-(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return SubExpr<L, R>(static_cast<const L &>(l), static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator*(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return MulExpr<L, R>(static_cast<const L &>(l), static_cast<const R &>(r));
}
/// Division: signals that lhs must be exactly divisible by rhs.
/// SpaceBuilder::require() extracts this as a constraint automatically.
template <typename L, typename R>
auto operator/(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return DivExpr<L, R>(static_cast<const L &>(l), static_cast<const R &>(r));
}

// ParmValue op Expr
template <typename R> auto operator+(ParmValue l, const SpaceExprBase<R> &r) {
  return AddExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}
template <typename R> auto operator-(ParmValue l, const SpaceExprBase<R> &r) {
  return SubExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}
template <typename R> auto operator*(ParmValue l, const SpaceExprBase<R> &r) {
  return MulExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}
template <typename R> auto operator/(ParmValue l, const SpaceExprBase<R> &r) {
  return DivExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}

// Expr op ParmValue
template <typename L> auto operator+(const SpaceExprBase<L> &l, ParmValue r) {
  return AddExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}
template <typename L> auto operator-(const SpaceExprBase<L> &l, ParmValue r) {
  return SubExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}
template <typename L> auto operator*(const SpaceExprBase<L> &l, ParmValue r) {
  return MulExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}
template <typename L> auto operator/(const SpaceExprBase<L> &l, ParmValue r) {
  return DivExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}

// ===----------------------------------------------------------------------===//
// ConstraintExpr — typed comparison between two expressions
// ===----------------------------------------------------------------------===//

enum class CmpKind { Le, Ge, Lt, Gt, Eq, Ne };

inline const char *cmpSymbol(CmpKind k) {
  switch (k) {
  case CmpKind::Le:
    return " <= ";
  case CmpKind::Ge:
    return " >= ";
  case CmpKind::Lt:
    return " < ";
  case CmpKind::Gt:
    return " > ";
  case CmpKind::Eq:
    return " == ";
  case CmpKind::Ne:
    return " != ";
  }
  llvm_unreachable("unknown CmpKind");
}

template <typename L, typename R, CmpKind K> struct ConstraintExpr {
  L lhs;
  R rhs;
  ConstraintExpr(L l, R r) : lhs(std::move(l)), rhs(std::move(r)) {}
  std::string describe() const {
    return lhs.describe() + cmpSymbol(K) + rhs.describe();
  }
};

// Expr cmp Expr
template <typename L, typename R>
auto operator<=(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<L, R, CmpKind::Le>(static_cast<const L &>(l),
                                           static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator>=(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<L, R, CmpKind::Ge>(static_cast<const L &>(l),
                                           static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator<(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<L, R, CmpKind::Lt>(static_cast<const L &>(l),
                                           static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator>(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<L, R, CmpKind::Gt>(static_cast<const L &>(l),
                                           static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator==(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<L, R, CmpKind::Eq>(static_cast<const L &>(l),
                                           static_cast<const R &>(r));
}
template <typename L, typename R>
auto operator!=(const SpaceExprBase<L> &l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<L, R, CmpKind::Ne>(static_cast<const L &>(l),
                                           static_cast<const R &>(r));
}

// Expr cmp ParmValue
template <typename L> auto operator<=(const SpaceExprBase<L> &l, ParmValue r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Le>(static_cast<const L &>(l),
                                                   ConstExpr{r});
}
template <typename L> auto operator>=(const SpaceExprBase<L> &l, ParmValue r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Ge>(static_cast<const L &>(l),
                                                   ConstExpr{r});
}
template <typename L> auto operator<(const SpaceExprBase<L> &l, ParmValue r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Lt>(static_cast<const L &>(l),
                                                   ConstExpr{r});
}
template <typename L> auto operator>(const SpaceExprBase<L> &l, ParmValue r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Gt>(static_cast<const L &>(l),
                                                   ConstExpr{r});
}
template <typename L> auto operator==(const SpaceExprBase<L> &l, ParmValue r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Eq>(static_cast<const L &>(l),
                                                   ConstExpr{r});
}
template <typename L> auto operator!=(const SpaceExprBase<L> &l, ParmValue r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Ne>(static_cast<const L &>(l),
                                                   ConstExpr{r});
}

// ParmValue cmp Expr
template <typename R> auto operator<=(ParmValue l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Le>(ConstExpr{l},
                                                   static_cast<const R &>(r));
}
template <typename R> auto operator>=(ParmValue l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Ge>(ConstExpr{l},
                                                   static_cast<const R &>(r));
}
template <typename R> auto operator<(ParmValue l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Lt>(ConstExpr{l},
                                                   static_cast<const R &>(r));
}
template <typename R> auto operator>(ParmValue l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Gt>(ConstExpr{l},
                                                   static_cast<const R &>(r));
}
template <typename R> auto operator==(ParmValue l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Eq>(ConstExpr{l},
                                                   static_cast<const R &>(r));
}
template <typename R> auto operator!=(ParmValue l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Ne>(ConstExpr{l},
                                                   static_cast<const R &>(r));
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
  /// configuration. Prefer this overload: constraints are evaluated in
  /// vectorized form, and the single-configuration check is derived from it
  /// for free (see ConfigSpace::addConstraint).
  void require(VecConstraint pred, llvm::StringRef description = "");
  /// Same, for a scalar predicate — vectorized automatically by evaluating it
  /// once per configuration in the batch. Convenience for predicates not
  /// worth hand-vectorizing; pass a VecConstraint instead when it is.
  void require(Constraint pred, llvm::StringRef description = "");

  /// Walk expr for / nodes; each one is extracted as a static, structural, or
  /// dynamic divisibility constraint (see addDivConstraint).
  template <typename E> void require(const SpaceExprBase<E> &expr) {
    extractDivConstraints(static_cast<const E &>(expr));
  }

  /// Same as require(SpaceExpr) on both sub-expressions, plus register the
  /// comparison as a dynamic predicate (with a specialized, inlined eval).
  template <typename L, typename R, CmpKind K>
  void require(ConstraintExpr<L, R, K> expr) {
    extractDivConstraints(expr.lhs);
    extractDivConstraints(expr.rhs);
    std::string desc = expr.describe();
    require(
        VecConstraint([lhs = std::move(expr.lhs), rhs = std::move(expr.rhs)](
                          const ConfigurationVector &c) -> arma::urowvec {
          const auto &lv = lhs.evalVec(c);
          const auto &rv = rhs.evalVec(c);
          if constexpr (K == CmpKind::Le)
            return lv <= rv;
          else if constexpr (K == CmpKind::Ge)
            return lv >= rv;
          else if constexpr (K == CmpKind::Lt)
            return lv < rv;
          else if constexpr (K == CmpKind::Gt)
            return lv > rv;
          else if constexpr (K == CmpKind::Eq)
            return lv == rv;
          else
            return lv != rv;
        }),
        desc);
  }

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
  };

  std::vector<DimEntry> dims_;
  std::vector<MultiplesEntry> multiples_;
  std::vector<PredicateEntry> predicates_;

  DimEntry &findEntry(const SpaceVar &v);
  SpaceVar findVarByName(llvm::StringRef name) const;
  int dimIndexByName(llvm::StringRef name) const;

  /// Reify a single A/B divisibility constraint detected from a DivExpr node.
  /// - ConstExpr/SpaceVar  → static filter on den's values
  /// - SpaceVar/SpaceVar   → structural mustDivide
  /// - everything else     → dynamic predicate
  template <typename Num, typename Den>
  void addDivConstraint(const Num &num, const Den &den) {
    if constexpr (std::is_same_v<Num, ConstExpr> &&
                  std::is_same_v<Den, SpaceVar>) {
      mustDivide(den, num.value);
    } else if constexpr (std::is_same_v<Num, SpaceVar> &&
                         std::is_same_v<Den, SpaceVar>) {
      mustDivide(den, num);
    } else if constexpr (detail::is_mul_expr_v<Den>) {
      // (B * C) | A  ⟺  B | A  ∧  C | A  ∧  B * C ≤ A
      // addDivConstraint(num, den.lhs);
      // addDivConstraint(num, den.rhs);
      std::string desc = den.describe() + " | " + num.describe();
      require(
          [num, den](const ConfigurationVector &c, arma::urowvec &valid) {
            const auto &nv = num.evalVec(c);
            const auto &dv = den.evalVec(c);
            // `%` is elementwise multiply = AND.
            valid %= vecDivides(dv, nv);
            valid %= (dv <= nv);
          },
          desc);
    } else {
      std::string desc = den.describe() + " | " + num.describe();
      require(
          [num, den](const ConfigurationVector &c, arma::urowvec &valid) {
            valid %= vecDivides(den.evalVec(c), num.evalVec(c));
          },
          desc);
    }
  }

  /// Recursively walk expr at compile time; for each DivExpr node, call
  /// addDivConstraint on its children.
  template <typename E> void extractDivConstraints(const E &expr) {
    if constexpr (detail::is_div_expr_v<E>) {
      addDivConstraint(expr.lhs, expr.rhs);
      extractDivConstraints(expr.lhs);
      extractDivConstraints(expr.rhs);
    } else if constexpr (detail::is_bin_expr_v<E>) {
      extractDivConstraints(expr.lhs);
      extractDivConstraints(expr.rhs);
    }
    // SpaceVar and ConstExpr are leaf nodes — nothing to recurse into.
  }
};

} // namespace mlir::cinm
