#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir::cinm::constraints {

// ===----------------------------------------------------------------------===//
// Constraint IR
// ===----------------------------------------------------------------------===//
//
// A small runtime expression tree. Constraints used to be encoded in the C++
// type system (CRTP expression templates), which bought inlined per-config
// evaluation -- irrelevant now that predicates run once per chunk over a
// ConfigurationVector of thousands of lanes -- at the cost of making the
// framework unable to reason about them. Analysis wants to pattern-match
// arbitrary shapes, partition variables, and rewrite; that is ordinary code
// over a runtime tree and template metaprogramming over a type.
//
// The node set is deliberately closed and small: everything here has to stay
// analysable. See docs/ConstraintAnalysisDesign.md.

struct ConstraintNode;
using ConstraintNodePtr = std::shared_ptr<const ConstraintNode>;

struct ConstraintNode {
  enum class Kind {
    Const, ///< literal; `value`
    Var,   ///< search parameter; `varIdx` / `varName`
    Add,   ///< n-ary sum
    Mul,   ///< n-ary product
    Div,   ///< binary quotient; also *asserts* exact divisibility (see below)

    // Boolean predicates

    Le,
    FirstBooleanKind = Le,
    Ge,
    Lt,
    Gt,
    Eq,
    Ne,

    /// binary divisibility *test*; `operands[0]` divides `operands[1]`. Unlike
    /// `Div` it produces a truth value and claims nothing: see the DSL's
    /// `divides()`.
    Divides,
    /// binary implication; both operands are boolean. The only connective:
    /// conjunction needs no node (two `require` calls), and disjunction has no
    /// caller and no story for the analyser, so it stays out until one exists.
    Implies,
  };

  Kind kind;

private:
  using VarState = std::pair<std::shared_ptr<size_t>, std::string>;
  using OpndState = SmallVector<ConstraintNodePtr, 2>;

public:
  // const
  ConstraintNode(ParmValue v) : kind(Kind::Const), state(v) {}
  // var
  ConstraintNode(llvm::StringRef v, std::shared_ptr<size_t> idx)
      : kind(Kind::Var), state(VarState(std::move(idx), v)) {}
  // binary
  ConstraintNode(ConstraintNode::Kind kind, ConstraintNodePtr lhs,
                 ConstraintNodePtr rhs)
      : kind(kind), state(OpndState{lhs, rhs}) {
    switch (kind) {
    case Kind::Const:
    case Kind::Var:
      assert(false && "Not a binary kind");
      break;
    case Kind::Implies:
      assert(isBoolKind(lhs->kind) && isBoolKind(rhs->kind));
      break;
    default:
      break;
    }
  }
  // n-ary
  ConstraintNode(ConstraintNode::Kind kind, ArrayRef<ConstraintNodePtr> nodes)
      : kind(kind), state{OpndState(nodes)} {
    assert(nodes.size() == 2 || kind == Kind::Mul || kind == Kind::Add);
  }

  /// Whether a node of this kind evaluates to a truth value rather than a
  /// number. The DSL's `Expr<Type::BOOL>` guarantees this statically; the
  /// analyser, which works on bare nodes, has to ask.
  inline static bool isBoolKind(ConstraintNode::Kind kind) {
    return kind >= ConstraintNode::Kind::FirstBooleanKind;
  }

  ParmValue constValue() const {
    assert(kind == Kind::Const);
    return std::get<ParmValue>(state);
  }
  llvm::StringRef varName() const {
    assert(kind == Kind::Var);
    return std::get<VarState>(state).second;
  }
  size_t varIdx() const {
    assert(kind == Kind::Var);
    return *std::get<VarState>(state).first;
  }
  std::shared_ptr<size_t> varIdxPtr() const {
    assert(kind == Kind::Var);
    return std::get<VarState>(state).first;
  }
  ArrayRef<ConstraintNodePtr> operands() const {
    if (std::holds_alternative<OpndState>(state))
      return std::get<OpndState>(state);
    return {};
  }

private:
  std::variant<
      /// Kind::Const
      ParmValue,
      /// Kind::Var — the shared ptr is the same cell SpaceVar holds, so the
      /// index resolves once SpaceBuilder::buildInto() has run and every handle
      /// sees it.
      VarState,
      /// Add/Mul are n-ary; others are binary.
      OpndState>
      state;
};

enum class Type { BOOL, INT };

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

/// Whether `node` divides anywhere below it.
bool containsDivision(const ConstraintNode &node);

/// Evaluate an arithmetic (non-boolean) node over a whole batch.
///
/// `a / b` in this IR means *exact* division, so when `exact` is given, the
/// lanes where some division below `node` did not come out exact are cleared
/// in it. The quotient returned for those lanes is the truncated one and is
/// meaningless; a boolean context must consult `exact` rather than trust it
/// (evalBoolNodeVec does). Passing null asks only for the quotient, which is
/// what a caller wanting a plain number out of a space expression wants.
ParmVector evalNodeVec(const ConstraintNode &node, const ConfigurationVector &c,
                       arma::urowvec *exact = nullptr);

/// Evaluate an arithmetic (non-Cmp) node for a single configuration. Used by
/// callers that need a plain number out of a space expression rather than a
/// predicate (see SpaceValue in the UPMEM plugin).
ParmValue evalNodeScalar(const ConstraintNode &node, const ConfWrapper &c);

/// Evaluate a boolean node over a batch, as a 0/1 mask per lane.
arma::urowvec evalBoolNodeVec(const ConstraintNode &node,
                              const ConfigurationVector &c);

/// Evaluate a boolean node over a batch, AND-ing the result into `valid`.
void evalBoolNodeInto(const ConstraintNode &node, const ConfigurationVector &c,
                      arma::urowvec &valid);

/// Wrap a boolean node as a VecConstraint, so a tree can be registered with
/// ConfigSpace::addConstraint like any other predicate.
VecConstraint toVecConstraint(ConstraintNodePtr node);

/// Human-readable rendering, e.g. "((8192 / gemv.M0) * (16384 / gemv.K0)) ==
/// (dpus * tasklets)". Used for the constraint descriptions debugIsValid()
/// reports.
std::string describeNode(const ConstraintNode &node);

// ===----------------------------------------------------------------------===//
// Analysis — identities (product equalities)
// ===----------------------------------------------------------------------===//

/// A product of search parameters with a constant coefficient:
/// `coeff * prod(vars)`. `vars` holds ConfigSpace parameter indices and may
/// repeat (a variable squared is the same index twice); it is kept sorted so
/// that two spellings of the same monomial compare equal.
struct Monomial {
  ParmValue coeff = 1;
  llvm::SmallVector<size_t, 4> vars;

  bool operator==(const Monomial &o) const {
    return coeff == o.coeff && vars == o.vars;
  }
};

/// `lhs == rhs`, with all division cleared by cross-multiplication. This is the
/// canonical form of an identity constraint; see
/// docs/ConstraintAnalysisDesign.md.
struct ProductEquality {
  Monomial lhs, rhs;
};

/// Reduce an arithmetic node to `(numer) / (denom)` as monomials. Fails
/// (returns nullopt) on anything that is not a rational monomial — in
/// particular on any Add or Sub, which is why capacity bounds (inequalities)
/// are not matched here.
///
/// Variable indices are read from the shared cells, so this is only meaningful
/// after SpaceBuilder::buildInto() has assigned them.
std::optional<std::pair<Monomial, Monomial>>
normalizeRationalMonomial(const ConstraintNode &node);

/// Match a Cmp node as a product equality, clearing denominators. Returns
/// nullopt unless the comparison is `==` and both sides are rational monomials.
std::optional<ProductEquality> matchProductEquality(const ConstraintNode &node);

std::string describeMonomial(const Monomial &m,
                             llvm::ArrayRef<std::string> paramNames);

// ===----------------------------------------------------------------------===//
// Analysis — interval bounds (inequalities)
// ===----------------------------------------------------------------------===//

/// The range a subexpression can span. `valid` is false when no useful bound
/// could be derived, in which case the interval must be ignored rather than
/// trusted.
struct Interval {
  int64_t lo = 0, hi = 0;
  bool valid = true;
};

/// Range a variable can still take: a fixed value once assigned, otherwise its
/// domain's extent. Returning an invalid Interval disables pruning for any
/// expression mentioning that variable.
using VarBounds = std::function<Interval(size_t varIdx)>;

/// Bound an arithmetic node given partial knowledge of its variables. Products
/// are bounded assuming non-negative operands -- true of every search
/// parameter here, and checked rather than assumed.
Interval evalNodeBounds(const ConstraintNode &node, const VarBounds &bounds);

/// Whether a boolean node can still be satisfied by some completion of the
/// current partial assignment. False means every completion violates it, so
/// the caller may prune. True is the safe answer: it never prunes a subtree
/// that might contain a solution.
///
/// This is what lets a capacity bound cut the search rather than filter it
/// afterwards: `sum of tile products <= MRAM` is monotone, so once the
/// smallest possible completion exceeds the limit the subtree is dead.
bool boolMayHold(const ConstraintNode &node, const VarBounds &bounds);

/// Whether a boolean node holds under *every* completion of the current
/// partial assignment. The dual of boolMayHold, and false is its safe answer.
///
/// An implication's consequent may only be acted on once its antecedent is
/// settled this way -- believing an antecedent that merely *might* hold would
/// impose the consequent on completions the constraint says nothing about.
bool boolMustHold(const ConstraintNode &node, const VarBounds &bounds);

} // namespace mlir::cinm::constraints
