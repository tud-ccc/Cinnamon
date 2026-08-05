#pragma once

#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>

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
ConstraintNodePtr makeVarNode(std::shared_ptr<size_t> idx,
                              llvm::StringRef name);
ConstraintNodePtr
makeNaryNode(ConstraintNode::Kind kind,
             llvm::SmallVector<ConstraintNodePtr, 2> operands);
ConstraintNodePtr makeBinNode(ConstraintNode::Kind kind, ConstraintNodePtr lhs,
                              ConstraintNodePtr rhs);
ConstraintNodePtr makeCmpNode(CmpKind cmp, ConstraintNodePtr lhs,
                              ConstraintNodePtr rhs);

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

/// Evaluate an arithmetic (non-Cmp) node over a whole batch.
ParmVector evalNodeVec(const ConstraintNode &node,
                       const ConfigurationVector &c);

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

// ===----------------------------------------------------------------------===//
// Analysis — Form A (product equality)
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
/// canonical form of a Form A constraint; see docs/ConstraintAnalysisDesign.md.
struct ProductEquality {
  Monomial lhs, rhs;
};

/// Reduce an arithmetic node to `(numer) / (denom)` as monomials. Fails
/// (returns nullopt) on anything that is not a rational monomial — in
/// particular on any Add or Sub, which is why capacity bounds (Form C) are not
/// matched here.
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
// Analysis — interval bounds (Form C)
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

/// Whether a comparison can still be satisfied by some completion of the
/// current partial assignment. False means every completion violates it, so
/// the caller may prune. True is the safe answer: it never prunes a subtree
/// that might contain a solution.
///
/// This is what lets a capacity bound cut the search rather than filter it
/// afterwards: `sum of tile products <= MRAM` is monotone, so once the
/// smallest possible completion exceeds the limit the subtree is dead.
bool cmpMayHold(const ConstraintNode &node, const VarBounds &bounds);

} // namespace mlir::cinm
