#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <llvm/Support/ErrorHandling.h>

namespace mlir::cinm::constraints {
using Kind = ConstraintNode::Kind;

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

ParmValue evalNode(const ConstraintNode &node, const ConfWrapper &c,
                   bool *exact) {
  using Kind = Kind;
  switch (node.kind) {
  case Kind::Const:
    return node.constValue();
  case Kind::Var:
    return c[node.varIdx()];
  case Kind::Add: {
    ParmValue acc = 0;
    for (const auto &op : node.operands())
      acc += evalNode(*op, c, exact);
    return acc;
  }
  case Kind::Mul: {
    ParmValue acc = 1;
    for (const auto &op : node.operands())
      acc *= evalNode(*op, c, exact);
    return acc;
  }
  case Kind::BoolAsInt:
    // evalBoolNode applies this subtree's own exactness, so `exact` is
    // deliberately not threaded through: nothing below here may falsify the
    // comparison above.
    return evalBoolNode(*node.operands()[0], c) ? 1 : 0;
  case Kind::Div: {
    const ParmValue num = evalNode(*node.operands()[0], c, exact);
    const ParmValue den = evalNode(*node.operands()[1], c, exact);
    // `a / b` asserts that b divides a. Record where it does not, so the
    // enclosing comparison can come out false rather than compare a truncated
    // quotient -- 1024 / 768 is not 1.
    if (exact && (den == 0 || num % den != 0))
      *exact = false;
    return den ? num / den : 0;
  }
  default:
    llvm_unreachable("boolean node evaluated as arithmetic; use evalBoolNode");
  }
}

bool evalBoolNode(const ConstraintNode &node, const ConfWrapper &c) {
  if (node.kind == Kind::Implies)
    return !evalBoolNode(*node.operands()[0], c) ||
           evalBoolNode(*node.operands()[1], c);

  if (node.kind == Kind::Divides) {
    // A test, so it never has to reject for being inexact -- being inexact is
    // the answer it reports.
    bool exact = true;
    const ParmValue divisor = evalNode(*node.operands()[0], c, &exact);
    const ParmValue dividend = evalNode(*node.operands()[1], c, &exact);
    // ...though a `/` *inside* one of its operands still has to be exact for
    // the operand to mean anything.
    return exact && divisor != 0 && dividend % divisor == 0;
  }

  assert(ConstraintNode::isBoolKind(node.kind) && "expected a boolean node");

  // A comparison is where an inexact division becomes observable, and where it
  // is discharged: `a / b` means "b divides a and the quotient is", so an
  // inexact division makes the comparison *false*, whatever the truncated
  // quotient happens to compare to.
  //
  // Doing it here rather than at the root is what makes the answer right under
  // an implication: an inexact division in the antecedent falsifies the
  // antecedent, which satisfies the implication, and rejecting at the root
  // would have thrown the configuration out instead.
  bool exact = true;
  const ParmValue lhs = evalNode(*node.operands()[0], c, &exact);
  const ParmValue rhs = evalNode(*node.operands()[1], c, &exact);
  if (!exact)
    return false;

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
}

Constraint toConstraint(ConstraintNodePtr node) {
  assert(node && ConstraintNode::isBoolKind(node->kind) &&
         "a constraint must be a boolean expression");
  return [node = std::move(node)](const ConfWrapper c) {
    return evalBoolNode(*node, c);
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
  case Kind::BoolAsInt:
    return "int(" + describeNode(*node.operands()[0]) + ")";
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
