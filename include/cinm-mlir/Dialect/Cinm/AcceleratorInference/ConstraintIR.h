#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <string>

namespace mlir::cinm {

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

enum class CmpKind { Le, Ge, Lt, Gt, Eq, Ne };

const char *cmpSymbol(CmpKind k);

struct ConstraintNode;
using ConstraintNodePtr = std::shared_ptr<const ConstraintNode>;

struct ConstraintNode {
  enum class Kind {
    Const, ///< literal; `value`
    Var,   ///< search parameter; `varIdx` / `varName`
    Add,   ///< n-ary sum
    Mul,   ///< n-ary product
    Sub,   ///< binary difference
    Div,   ///< binary quotient; also *asserts* exact divisibility (see below)
    Cmp,   ///< binary comparison; `cmp`
  };

  Kind kind;
  /// Kind::Const
  ParmValue value = 0;
  /// Kind::Var — the same cell SpaceVar holds, so the index resolves once
  /// SpaceBuilder::buildInto() has run and every handle sees it.
  std::shared_ptr<size_t> varIdx;
  std::string varName;
  /// Kind::Cmp
  CmpKind cmp = CmpKind::Eq;
  /// Add/Mul are n-ary; Sub/Div/Cmp are binary. n-ary matters: it is what lets
  /// prod(extent/block) be one node with a child per iteration dimension
  /// instead of a left-leaning tree the analyser would have to re-flatten.
  llvm::SmallVector<ConstraintNodePtr, 2> operands;
};

// ===----------------------------------------------------------------------===//
// Node construction
// ===----------------------------------------------------------------------===//

ConstraintNodePtr makeConstNode(ParmValue v);
ConstraintNodePtr makeVarNode(std::shared_ptr<size_t> idx, llvm::StringRef name);
ConstraintNodePtr makeNaryNode(ConstraintNode::Kind kind,
                               llvm::SmallVector<ConstraintNodePtr, 2> operands);
ConstraintNodePtr makeBinNode(ConstraintNode::Kind kind, ConstraintNodePtr lhs,
                              ConstraintNodePtr rhs);
ConstraintNodePtr makeCmpNode(CmpKind cmp, ConstraintNodePtr lhs,
                              ConstraintNodePtr rhs);

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

/// Evaluate an arithmetic (non-Cmp) node over a whole batch.
ParmVector evalNodeVec(const ConstraintNode &node, const ConfigurationVector &c);

/// Evaluate an arithmetic (non-Cmp) node for a single configuration. Used by
/// callers that need a plain number out of a space expression rather than a
/// predicate (see SpaceValue in the UPMEM plugin).
ParmValue evalNodeScalar(const ConstraintNode &node, const ConfWrapper &c);

/// Evaluate a Cmp node over a batch, AND-ing the result into `valid`.
void evalCmpNodeInto(const ConstraintNode &node, const ConfigurationVector &c,
                     arma::urowvec &valid);

/// Wrap a Cmp node as a VecConstraint, so a tree can be registered with
/// ConfigSpace::addConstraint like any other predicate.
VecConstraint toVecConstraint(ConstraintNodePtr node);

/// Human-readable rendering, e.g. "((8192 / gemv.M0) * (16384 / gemv.K0)) ==
/// (dpus * tasklets)". Used for the constraint descriptions debugIsValid()
/// reports.
std::string describeNode(const ConstraintNode &node);

} // namespace mlir::cinm
