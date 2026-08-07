#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <llvm/Support/ErrorHandling.h>

namespace mlir::cinm::constraints {
using Kind = ConstraintNode::Kind;

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

static bool containsDivision(const ConstraintNode &node) {
  if (node.kind == Kind::Div)
    return true;
  return llvm::any_of(node.operands(), [](const ConstraintNodePtr &child) {
    return containsDivision(*child);
  });
}

ParmVector evalNodeVec(const ConstraintNode &node, const ConfigurationVector &c,
                       arma::urowvec *exact) {
  using Kind = Kind;
  switch (node.kind) {
  case Kind::Const: {
    ParmVector r(c.size());
    r.fill(node.constValue());
    return r;
  }
  case Kind::Var:
    return c[node.varIdx()];
  case Kind::Add: {
    ParmVector acc = c.zeros();
    for (const auto &op : node.operands())
      acc += evalNodeVec(*op, c, exact);
    return acc;
  }
  case Kind::Mul: {
    ParmVector acc = c.ones();
    for (const auto &op : node.operands())
      acc %=
          evalNodeVec(*op, c, exact); // `%` is Armadillo's elementwise multiply
    return acc;
  }
  case Kind::Div: {
    const ParmVector num = evalNodeVec(*node.operands()[0], c, exact);
    const ParmVector den = evalNodeVec(*node.operands()[1], c, exact);
    // `a / b` asserts that b divides a. Record where it does not, so the
    // enclosing comparison can come out false rather than compare a truncated
    // quotient -- 1024 / 768 is not 1.
    if (exact)
      *exact %= vecDivides(den, num);
    // The quotient still has to be computed for every lane: a vectorized
    // evaluation cannot short-circuit past the bad ones, and a plain
    // elementwise division would be UB on a zero divisor.
    return vecSafeDiv(num, den);
  }
  default:
    llvm_unreachable(
        "boolean node evaluated as arithmetic; use evalBoolNodeVec");
  }
}

arma::urowvec evalBoolNodeVec(const ConstraintNode &node,
                              const ConfigurationVector &c) {
  if (node.kind == Kind::Implies) {
    // `a => b` is `!a | b`. Both sides are evaluated for every lane: a
    // vectorized evaluation has no short-circuit, and the antecedent's own
    // operands are arithmetic, so there is nothing unsafe to guard against.
    const arma::urowvec ante = evalBoolNodeVec(*node.operands()[0], c);
    const arma::urowvec cons = evalBoolNodeVec(*node.operands()[1], c);
    return (ante == 0) || (cons != 0);
  }

  if (node.kind == Kind::Divides) {
    // A test, so it never has to reject a lane for being inexact -- being
    // inexact is the answer it reports.
    arma::urowvec exact(c.size(), arma::fill::ones);
    const ParmVector divisor = evalNodeVec(*node.operands()[0], c, &exact);
    const ParmVector dividend = evalNodeVec(*node.operands()[1], c, &exact);
    arma::urowvec result = vecDivides(divisor, dividend);
    // ...though a `/` *inside* one of its operands still has to be exact for
    // the operand to mean anything.
    return result % exact;
  }

  assert(ConstraintNode::isBoolKind(node.kind) && "expected a boolean node");

  // A comparison is where an inexact division becomes observable, and where it
  // is discharged: `a / b` means "b divides a and the quotient is", so a lane
  // whose division does not come out exact makes the comparison *false*,
  // whatever the truncated quotient happens to compare to.
  //
  // Doing it here rather than at the root is what makes the answer right under
  // an implication: an inexact division in the antecedent falsifies the
  // antecedent, which satisfies the implication, and clearing the lane at the
  // root would have rejected it instead.
  const bool hasDiv = containsDivision(node);
  arma::urowvec exact;
  if (hasDiv)
    exact = arma::urowvec(c.size(), arma::fill::ones);
  arma::urowvec *exactPtr = hasDiv ? &exact : nullptr;

  const ParmVector lhs = evalNodeVec(*node.operands()[0], c, exactPtr);
  const ParmVector rhs = evalNodeVec(*node.operands()[1], c, exactPtr);
  arma::urowvec result = [&]() -> arma::urowvec {
    switch (node.kind) {
    case Kind::Le:
      return lhs <= rhs;
    case Kind::Ge:
      return lhs >= rhs;
    case Kind::Lt:
      return lhs < rhs;
    case Kind::Gt:
      return lhs > rhs;
    case Kind::Eq:
      return lhs == rhs;
    case Kind::Ne:
      return lhs != rhs;
    default:
      llvm_unreachable("unknown ConstraintNode::Kind");
    }
  }();
  if (hasDiv)
    result %= exact; // `%=` is elementwise multiply, i.e. AND over 0/1 masks
  return result;
}

void evalBoolNodeInto(const ConstraintNode &node, const ConfigurationVector &c,
                      arma::urowvec &valid) {
  // `%=` is Armadillo's elementwise multiply, i.e. AND over 0/1 masks: a lane
  // already cleared by an earlier constraint stays cleared.
  valid %= evalBoolNodeVec(node, c);
}

VecConstraint toVecConstraint(ConstraintNodePtr node) {
  assert(node && ConstraintNode::isBoolKind(node->kind) &&
         "a constraint must be a boolean expression");
  return [node = std::move(node)](const ConfigurationVector &c,
                                  arma::urowvec &valid) {
    evalBoolNodeInto(*node, c, valid);
  };
}

// ===----------------------------------------------------------------------===//
// Printing
// ===----------------------------------------------------------------------===//

std::string describeNode(const ConstraintNode &node) {
  using Kind = ConstraintNode::Kind;
  auto joinOperands = [&](const char *sep) {
    std::string s = "(";
    for (size_t i = 0; i < node.operands().size(); ++i) {
      if (i)
        s += sep;
      s += describeNode(*node.operands()[i]);
    }
    return s + ")";
  };

  switch (node.kind) {

  case Kind::Const:
    return std::to_string(node.constValue());
  case Kind::Var:
    return std::string(node.varName());
  case Kind::Add:
    return joinOperands(" + ");
  case Kind::Mul:
    return joinOperands(" * ");
  case Kind::Div:
    return joinOperands(" / ");
  case ConstraintNode::Kind::Le:
    return joinOperands(" <= ");
  case ConstraintNode::Kind::Ge:
    return joinOperands(" >= ");
  case ConstraintNode::Kind::Lt:
    return joinOperands(" < ");
  case ConstraintNode::Kind::Gt:
    return joinOperands(" > ");
  case ConstraintNode::Kind::Eq:
    return joinOperands(" == ");
  case ConstraintNode::Kind::Ne:
    return joinOperands(" != ");
  case Kind::Divides:
    return joinOperands(" | ");
  case Kind::Implies:
    return joinOperands(" => ");
  }
}

} // namespace mlir::cinm::constraints
