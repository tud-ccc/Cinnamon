#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"
#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <string>
#include <vector>

namespace mlir::cinm {

class SpaceVar;
class SpaceExpr;
class ConstraintExpr;

// ===----------------------------------------------------------------------===//
// SpaceVar
// ===----------------------------------------------------------------------===//

/// A handle to a named search-space dimension created by SpaceBuilder.
/// The index into ConfigSpace is written lazily when buildInto() is called;
/// all handles remain valid and return the correct index after that point.
class SpaceVar {
public:
  SpaceVar() : idx_(std::make_shared<int64_t>(-1)), maxVal_(0) {}

  llvm::StringRef name() const { return name_; }
  /// Index in the ConfigSpace — valid only after SpaceBuilder::buildInto().
  int64_t idx() const { return *idx_; }
  int64_t get(const ConfWrapper &c) const { return c[*idx_]; }
  int64_t operator[](const ConfWrapper &c) const { return get(c); }
  /// Upper bound of this variable's domain. Used by divisorsOf(name, SpaceVar).
  int64_t maxVal() const { return maxVal_; }

private:
  friend class SpaceBuilder;
  explicit SpaceVar(llvm::StringRef name, int64_t maxVal)
      : name_(name.str()), idx_(std::make_shared<int64_t>(-1)), maxVal_(maxVal) {}

  std::string name_;
  std::shared_ptr<int64_t> idx_;
  int64_t maxVal_;
};

// ===----------------------------------------------------------------------===//
// Expression AST
// ===----------------------------------------------------------------------===//

namespace detail {
/// A node in the SpaceExpr expression tree.
struct ExprNode {
  enum Kind { Const, Var, Add, Sub, Mul, Div };
  Kind kind;
  int64_t constVal = 0; // Const
  SpaceVar var;         // Var
  std::shared_ptr<const ExprNode> lhs, rhs; // binary ops
};
using NodePtr = std::shared_ptr<const ExprNode>;
} // namespace detail

// ===----------------------------------------------------------------------===//
// SpaceExpr
// ===----------------------------------------------------------------------===//

/// An arithmetic expression over SpaceVars and integer constants.
///
/// SpaceExprs form an AST. SpaceBuilder::require() traverses the tree to find
/// division nodes and reifies them as static, structural, or dynamic
/// divisibility constraints. No constraint state is stored inside SpaceExpr.
///
/// Arithmetic operators combine expressions; the /  operator signals that the
/// numerator must be exactly divisible by the denominator.
class SpaceExpr {
public:
  SpaceExpr(int64_t constant);          // implicit
  SpaceExpr(const SpaceVar &var);       // implicit

  /// Evaluate the expression at a given configuration point.
  /// Valid only after SpaceBuilder::buildInto() (SpaceVar indices are needed).
  int64_t eval(const ConfWrapper &c) const;

private:
  friend class SpaceBuilder;
  friend SpaceExpr operator+(SpaceExpr, SpaceExpr);
  friend SpaceExpr operator-(SpaceExpr, SpaceExpr);
  friend SpaceExpr operator*(SpaceExpr, SpaceExpr);
  friend SpaceExpr operator/(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator<=(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator>=(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator<(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator>(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator==(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator!=(SpaceExpr, SpaceExpr);

  explicit SpaceExpr(std::shared_ptr<const detail::ExprNode> root)
      : root_(std::move(root)) {}

  std::shared_ptr<const detail::ExprNode> root_;
};

SpaceExpr operator+(SpaceExpr lhs, SpaceExpr rhs);
SpaceExpr operator-(SpaceExpr lhs, SpaceExpr rhs);
SpaceExpr operator*(SpaceExpr lhs, SpaceExpr rhs);
/// Division: signals that lhs must be divisible by rhs.
/// SpaceBuilder::require() extracts this as a constraint automatically.
SpaceExpr operator/(SpaceExpr lhs, SpaceExpr rhs);

// ===----------------------------------------------------------------------===//
// ConstraintExpr
// ===----------------------------------------------------------------------===//

/// A boolean comparison between two SpaceExprs, produced by ==, !=, <, <=, >, >=.
/// Pass to SpaceBuilder::require() — it extracts any divisibility constraints
/// from both sub-expressions and registers the comparison as a dynamic predicate.
class ConstraintExpr {
public:
  enum Kind { Le, Ge, Lt, Gt, Eq, Ne };

private:
  friend class SpaceBuilder;
  friend ConstraintExpr operator<=(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator>=(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator<(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator>(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator==(SpaceExpr, SpaceExpr);
  friend ConstraintExpr operator!=(SpaceExpr, SpaceExpr);

  ConstraintExpr(Kind k, SpaceExpr l, SpaceExpr r)
      : kind_(k), lhs_(std::move(l)), rhs_(std::move(r)) {}

  Kind kind_;
  SpaceExpr lhs_, rhs_;
};

ConstraintExpr operator<=(SpaceExpr lhs, SpaceExpr rhs);
ConstraintExpr operator>=(SpaceExpr lhs, SpaceExpr rhs);
ConstraintExpr operator<(SpaceExpr lhs, SpaceExpr rhs);
ConstraintExpr operator>(SpaceExpr lhs, SpaceExpr rhs);
ConstraintExpr operator==(SpaceExpr lhs, SpaceExpr rhs);
ConstraintExpr operator!=(SpaceExpr lhs, SpaceExpr rhs);

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
  /// Extract divisibility constraints from all / nodes in expr, then register
  /// them as static, structural, or dynamic constraints as appropriate.
  void require(SpaceExpr expr);
  /// Same as require(SpaceExpr) on both sub-expressions, plus register the
  /// comparison itself as a dynamic predicate.
  void require(ConstraintExpr expr);

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

  /// Recursively walk node, reifying each / as a constraint.
  void extractDivConstraints(const detail::NodePtr &node);
  /// Reify a single A/B divisibility constraint.
  void addDivConstraint(const detail::NodePtr &num, const detail::NodePtr &den);
};

} // namespace mlir::cinm
