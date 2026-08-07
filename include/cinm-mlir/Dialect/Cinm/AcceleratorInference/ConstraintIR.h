#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConfigSpace.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <string>
#include <utility>
#include <variant>

namespace mlir::cinm::constraints {

// ===----------------------------------------------------------------------===//
// Constraint IR
// ===----------------------------------------------------------------------===//
//
// A small runtime expression tree.
//
// What reasons about them is a finite-domain solver: every node here has
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
    /// unary; the truth value of its (boolean) operand as 0 or 1. The one node
    /// that crosses from a truth value back to a number, which is what makes a
    /// *count* of conditions expressible -- and a count is what a constraint
    /// over "how many dimensions satisfy X" needs.
    ///
    /// A division below it is discharged inside it, by the boolean operand, and
    /// never escapes to the comparison containing it: an inexact division makes
    /// this node 0, not the enclosing comparison false.
    BoolAsInt,

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
    size_t offset;
    std::string name;
  };
  using OpndState = SmallVector<ConstraintNodePtr, 2>;

public:
  // const
  ConstraintNode(ParmValue v) : kind(Kind::Const), state(v) {}
  // var
  /// One dimension of a parameter. There is no kind here: only a quantity ever
  /// reaches this IR, because only IntVar converts to an Expr.
  ///
  /// `offset` is which of the parameter's dimensions this is, since the shared
  /// cell holds the first one. It is zero for everything occupying a single
  /// dimension, which is everything except one axis of an ordering.
  ConstraintNode(llvm::StringRef v, std::shared_ptr<size_t> idx,
                 size_t offset = 0)
      : kind(Kind::Var), state(VarState{std::move(idx), offset, v.str()}) {}
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
  // unary
  ConstraintNode(ConstraintNode::Kind kind, ConstraintNodePtr operand)
      : kind(kind), state(OpndState{std::move(operand)}) {
    assert(kind == Kind::BoolAsInt && "the only unary kind");
    assert(isBoolKind(operands()[0]->kind) && "BoolAsInt takes a truth value");
  }
  // n-ary
  ConstraintNode(ConstraintNode::Kind kind, ArrayRef<ConstraintNodePtr> nodes)
      : kind(kind), state{OpndState(nodes)} {
    assert(nodes.size() == 2 || kind == Kind::Mul || kind == Kind::Add);
  }

  /// Whether a node of this kind evaluates to a truth value rather than a
  /// number. The DSL's `Expr<Type::BOOL>` guarantees this statically; code
  /// working on bare nodes -- the Gecode translation, the evaluator -- has to
  /// ask.
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
  /// The Configuration entry this node reads, which is the parameter's first
  /// one plus this node's offset into it.
  size_t varIdx() const {
    assert(kind == Kind::Var);
    const VarState &var = std::get<VarState>(state);
    return *var.idx + var.offset;
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

/// Evaluate an arithmetic (non-boolean) node against one configuration.
///
/// `a / b` in this IR means *exact* division, so when `exact` is given, it is
/// cleared if some division below `node` did not come out exact. The quotient
/// returned is then the truncated one and is meaningless; a boolean context
/// must consult `exact` rather than trust it (evalBoolNode does). Passing null
/// asks only for the quotient, which is what a caller wanting a plain number
/// out of a space expression wants.
ParmValue evalNode(const ConstraintNode &node, const ConfWrapper &c,
                   bool *exact = nullptr);

/// Evaluate a boolean node against one configuration.
bool evalBoolNode(const ConstraintNode &node, const ConfWrapper &c);

/// Wrap a boolean node as a Constraint, so a tree can be registered with
/// ConfigSpace::addConstraint like any other predicate.
Constraint toConstraint(ConstraintNodePtr node);

/// Human-readable rendering, e.g. "((8192 / gemv.M0) * (16384 / gemv.K0)) ==
/// (dpus * tasklets)". Used for the constraint descriptions debugIsValid()
/// reports.
std::string describeNode(const ConstraintNode &node);

} // namespace mlir::cinm::constraints
