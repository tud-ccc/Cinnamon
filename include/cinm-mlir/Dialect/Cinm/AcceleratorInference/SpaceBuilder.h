#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// SpaceExprBase — CRTP base for all space expressions
// ===----------------------------------------------------------------------===//

template <typename Derived>
struct SpaceExprBase {
  int64_t eval(const ConfWrapper &c) const {
    return static_cast<const Derived &>(*this).evalImpl(c);
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
  SpaceVar() : idx_(std::make_shared<int64_t>(-1)), maxVal_(0) {}

  llvm::StringRef name() const { return name_; }
  /// Index in the ConfigSpace — valid only after SpaceBuilder::buildInto().
  int64_t idx() const { return *idx_; }
  int64_t get(const ConfWrapper &c) const { return c[*idx_]; }
  int64_t operator[](const ConfWrapper &c) const { return get(c); }
  /// Upper bound of this variable's domain. Used by divisorsOf(name, SpaceVar).
  int64_t maxVal() const { return maxVal_; }

  int64_t evalImpl(const ConfWrapper &c) const { return get(c); }

private:
  friend class SpaceBuilder;
  explicit SpaceVar(llvm::StringRef name, int64_t maxVal)
      : name_(name.str()), idx_(std::make_shared<int64_t>(-1)),
        maxVal_(maxVal) {}

  std::string name_;
  std::shared_ptr<int64_t> idx_;
  int64_t maxVal_;
};

// ===----------------------------------------------------------------------===//
// ConstExpr — compile-time integer constant
// ===----------------------------------------------------------------------===//

struct ConstExpr : SpaceExprBase<ConstExpr> {
  int64_t value;
  explicit ConstExpr(int64_t v) : value(v) {}
  int64_t evalImpl(const ConfWrapper &) const { return value; }
};

// ===----------------------------------------------------------------------===//
// BinExpr — binary arithmetic node
// ===----------------------------------------------------------------------===//

namespace detail {
struct OpAdd { static int64_t apply(int64_t a, int64_t b) { return a + b; } };
struct OpSub { static int64_t apply(int64_t a, int64_t b) { return a - b; } };
struct OpMul { static int64_t apply(int64_t a, int64_t b) { return a * b; } };
struct OpDiv { static int64_t apply(int64_t a, int64_t b) { return b ? a / b : 0; } };
} // namespace detail

template <typename L, typename R, typename Op>
struct BinExpr : SpaceExprBase<BinExpr<L, R, Op>> {
  L lhs;
  R rhs;
  BinExpr(L l, R r) : lhs(std::move(l)), rhs(std::move(r)) {}
  int64_t evalImpl(const ConfWrapper &c) const {
    return Op::apply(lhs.eval(c), rhs.eval(c));
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

// int64_t op Expr
template <typename R>
auto operator+(int64_t l, const SpaceExprBase<R> &r) {
  return AddExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}
template <typename R>
auto operator-(int64_t l, const SpaceExprBase<R> &r) {
  return SubExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}
template <typename R>
auto operator*(int64_t l, const SpaceExprBase<R> &r) {
  return MulExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}
template <typename R>
auto operator/(int64_t l, const SpaceExprBase<R> &r) {
  return DivExpr<ConstExpr, R>(ConstExpr{l}, static_cast<const R &>(r));
}

// Expr op int64_t
template <typename L>
auto operator+(const SpaceExprBase<L> &l, int64_t r) {
  return AddExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}
template <typename L>
auto operator-(const SpaceExprBase<L> &l, int64_t r) {
  return SubExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}
template <typename L>
auto operator*(const SpaceExprBase<L> &l, int64_t r) {
  return MulExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}
template <typename L>
auto operator/(const SpaceExprBase<L> &l, int64_t r) {
  return DivExpr<L, ConstExpr>(static_cast<const L &>(l), ConstExpr{r});
}

// ===----------------------------------------------------------------------===//
// ConstraintExpr — typed comparison between two expressions
// ===----------------------------------------------------------------------===//

enum class CmpKind { Le, Ge, Lt, Gt, Eq, Ne };

template <typename L, typename R, CmpKind K>
struct ConstraintExpr {
  L lhs;
  R rhs;
  ConstraintExpr(L l, R r) : lhs(std::move(l)), rhs(std::move(r)) {}
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

// Expr cmp int64_t
template <typename L>
auto operator<=(const SpaceExprBase<L> &l, int64_t r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Le>(static_cast<const L &>(l),
                                                    ConstExpr{r});
}
template <typename L>
auto operator>=(const SpaceExprBase<L> &l, int64_t r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Ge>(static_cast<const L &>(l),
                                                    ConstExpr{r});
}
template <typename L>
auto operator<(const SpaceExprBase<L> &l, int64_t r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Lt>(static_cast<const L &>(l),
                                                    ConstExpr{r});
}
template <typename L>
auto operator>(const SpaceExprBase<L> &l, int64_t r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Gt>(static_cast<const L &>(l),
                                                    ConstExpr{r});
}
template <typename L>
auto operator==(const SpaceExprBase<L> &l, int64_t r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Eq>(static_cast<const L &>(l),
                                                    ConstExpr{r});
}
template <typename L>
auto operator!=(const SpaceExprBase<L> &l, int64_t r) {
  return ConstraintExpr<L, ConstExpr, CmpKind::Ne>(static_cast<const L &>(l),
                                                    ConstExpr{r});
}

// int64_t cmp Expr
template <typename R>
auto operator<=(int64_t l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Le>(ConstExpr{l},
                                                    static_cast<const R &>(r));
}
template <typename R>
auto operator>=(int64_t l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Ge>(ConstExpr{l},
                                                    static_cast<const R &>(r));
}
template <typename R>
auto operator<(int64_t l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Lt>(ConstExpr{l},
                                                    static_cast<const R &>(r));
}
template <typename R>
auto operator>(int64_t l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Gt>(ConstExpr{l},
                                                    static_cast<const R &>(r));
}
template <typename R>
auto operator==(int64_t l, const SpaceExprBase<R> &r) {
  return ConstraintExpr<ConstExpr, R, CmpKind::Eq>(ConstExpr{l},
                                                    static_cast<const R &>(r));
}
template <typename R>
auto operator!=(int64_t l, const SpaceExprBase<R> &r) {
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
  SpaceVar intRange(llvm::StringRef name, int64_t lo, int64_t hi);
  /// Declare a dimension with values 2^expLo, ..., 2^expHi.
  SpaceVar pow2Range(llvm::StringRef name, int expLo, int expHi);
  /// Declare a dimension whose values are exactly the divisors of n.
  SpaceVar divisorsOf(llvm::StringRef name, int64_t n);
  /// Declare a dimension in [1, v.maxVal()] with the constraint that its
  /// values must divide the runtime value of v (v % result == 0).
  SpaceVar divisorsOf(llvm::StringRef name, SpaceVar v);

  /// Static filter: retain only values of v that are divisors of n.
  void mustDivide(SpaceVar v, int64_t n);
  /// Structural constraint: child must be a multiple of parent (parent divides
  /// child; child % parent == 0). Applied after all dims are added.
  void mustDivide(SpaceVar parent, SpaceVar child);
  /// Arbitrary predicate; invalid configurations are skipped by the framework.
  void require(Constraint pred);

  /// Walk expr for / nodes; each one is extracted as a static, structural, or
  /// dynamic divisibility constraint (see addDivConstraint).
  template <typename E>
  void require(const SpaceExprBase<E> &expr) {
    extractDivConstraints(static_cast<const E &>(expr));
  }

  /// Same as require(SpaceExpr) on both sub-expressions, plus register the
  /// comparison as a dynamic predicate (with a specialized, inlined eval).
  template <typename L, typename R, CmpKind K>
  void require(ConstraintExpr<L, R, K> expr) {
    extractDivConstraints(expr.lhs);
    extractDivConstraints(expr.rhs);
    predicates_.push_back(
        [lhs = std::move(expr.lhs),
         rhs = std::move(expr.rhs)](const ConfWrapper &c) -> bool {
          const auto lv = lhs.eval(c), rv = rhs.eval(c);
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
        });
  }

  /// Commit all declarations and constraints into space in the correct order.
  void buildInto(ConfigSpace &space);

private:
  struct DimEntry {
    SpaceVar var;
    enum Kind { IntRange, Pow2, DivisorsOfConst } kind;
    int64_t lo, hi;
    std::vector<int64_t> divisorFilters; ///< keepDivisorsOf(n) for each n
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

  std::vector<DimEntry> dims_;
  std::vector<MultiplesEntry> multiples_;
  std::vector<Constraint> predicates_;

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
      if constexpr (std::is_same_v<Num, SpaceVar>) {
        // When num is a SpaceVar, recursing into addDivConstraint(num, den.lhs) and
        // addDivConstraint(num, den.rhs) would call mustDivide() twice with the same
        // child (num), registering it in two separate parent groups. Both groups then
        // write conf[num] from different slots, making at() and forEach() decode the
        // same flat index to different values — breaking the isValid assertion.
        predicates_.push_back([num, den](const ConfWrapper &c) -> bool {
          const auto nv = num.eval(c);
          const auto dv = den.eval(c);
          return dv != 0 && nv % dv == 0 && nv <= dv;
        });
      } else {
        // (B * C) | A  ⟺  B | A  ∧  C | A  ∧  B * C ≤ A
        addDivConstraint(num, den.lhs);
        addDivConstraint(num, den.rhs);
        predicates_.push_back([num, den](const ConfWrapper &c) -> bool {
          return den.eval(c) <= num.eval(c);
        });
      }
    } else {
      predicates_.push_back([num, den](const ConfWrapper &c) -> bool {
        const auto dv = den.eval(c);
        return dv != 0 && num.eval(c) % dv == 0;
      });
    }
  }

  /// Recursively walk expr at compile time; for each DivExpr node, call
  /// addDivConstraint on its children.
  template <typename E>
  void extractDivConstraints(const E &expr) {
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
