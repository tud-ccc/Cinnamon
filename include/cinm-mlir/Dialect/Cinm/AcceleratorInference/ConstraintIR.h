#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"

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
// framework unable to reason about them.
//
// What reasons about them now is a finite-domain solver: every node here has
// an image in Gecode, and ConstraintGecode.cpp is the translation. That is
// what keeps the node set closed and small -- a node with no propagator is a
// node that would send its whole constraint back to being filtered.
//
// The evaluator below is no longer how a constraint is enforced. It survives
// because a space still has to be able to say *why* a configuration is not in
// it (ConfigSpace::debugIsValid), and because agreeing with it is the check on
// the translation.

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
  struct VarState {
    std::shared_ptr<size_t> idx;
    std::string name;
    ParamKind kind;
  };
  using OpndState = SmallVector<ConstraintNodePtr, 2>;

public:
  // const
  ConstraintNode(ParmValue v) : kind(Kind::Const), state(v) {}
  // var
  ConstraintNode(llvm::StringRef v, std::shared_ptr<size_t> idx,
                 ParamKind paramKind = ParamKind::Integer)
      : kind(Kind::Var), state(VarState{std::move(idx), v.str(), paramKind}) {}
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
    return std::get<VarState>(state).name;
  }
  size_t varIdx() const {
    assert(kind == Kind::Var);
    return *std::get<VarState>(state).idx;
  }
  std::shared_ptr<size_t> varIdxPtr() const {
    assert(kind == Kind::Var);
    return std::get<VarState>(state).idx;
  }
  ParamKind varKind() const {
    assert(kind == Kind::Var);
    return std::get<VarState>(state).kind;
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

/// The kind of the first parameter below `node` that is not `ParamKind::
/// Integer`, with its name, or nullopt if there is none.
///
/// Values of such a parameter are a numbering, not a quantity: only equality
/// and inequality mean anything on them, and even that only against another
/// value of the same parameter. The DSL uses this to reject the rest at the
/// point the expression is built, which is where the mistake is.
std::optional<std::pair<ParamKind, llvm::StringRef>>
findNonArithmeticVar(const ConstraintNode &node);

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

} // namespace mlir::cinm::constraints
