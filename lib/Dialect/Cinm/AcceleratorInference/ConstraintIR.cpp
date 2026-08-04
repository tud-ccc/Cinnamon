#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <llvm/Support/ErrorHandling.h>

namespace mlir::cinm {

const char *cmpSymbol(CmpKind k) {
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

// ===----------------------------------------------------------------------===//
// Construction
// ===----------------------------------------------------------------------===//

static ConstraintNodePtr make(ConstraintNode n) {
  return std::make_shared<const ConstraintNode>(std::move(n));
}

ConstraintNodePtr makeConstNode(ParmValue v) {
  ConstraintNode n;
  n.kind = ConstraintNode::Kind::Const;
  n.value = v;
  return make(std::move(n));
}

ConstraintNodePtr makeVarNode(std::shared_ptr<size_t> idx,
                              llvm::StringRef name) {
  ConstraintNode n;
  n.kind = ConstraintNode::Kind::Var;
  n.varIdx = std::move(idx);
  n.varName = name.str();
  return make(std::move(n));
}

ConstraintNodePtr
makeNaryNode(ConstraintNode::Kind kind,
             llvm::SmallVector<ConstraintNodePtr, 2> operands) {
  assert((kind == ConstraintNode::Kind::Add ||
          kind == ConstraintNode::Kind::Mul) &&
         "only Add and Mul are n-ary");
  ConstraintNode n;
  n.kind = kind;
  n.operands = std::move(operands);
  return make(std::move(n));
}

ConstraintNodePtr makeBinNode(ConstraintNode::Kind kind, ConstraintNodePtr lhs,
                              ConstraintNodePtr rhs) {
  ConstraintNode n;
  n.kind = kind;
  n.operands = {std::move(lhs), std::move(rhs)};
  return make(std::move(n));
}

ConstraintNodePtr makeCmpNode(CmpKind cmp, ConstraintNodePtr lhs,
                              ConstraintNodePtr rhs) {
  ConstraintNode n;
  n.kind = ConstraintNode::Kind::Cmp;
  n.cmp = cmp;
  n.operands = {std::move(lhs), std::move(rhs)};
  return make(std::move(n));
}

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

ParmVector evalNodeVec(const ConstraintNode &node,
                       const ConfigurationVector &c) {
  using Kind = ConstraintNode::Kind;
  switch (node.kind) {
  case Kind::Const: {
    ParmVector r(c.size());
    r.fill(node.value);
    return r;
  }
  case Kind::Var:
    return c[*node.varIdx];
  case Kind::Add: {
    ParmVector acc = c.zeros();
    for (const auto &op : node.operands)
      acc += evalNodeVec(*op, c);
    return acc;
  }
  case Kind::Mul: {
    ParmVector acc = c.ones();
    for (const auto &op : node.operands)
      acc %= evalNodeVec(*op, c); // `%` is Armadillo's elementwise multiply
    return acc;
  }
  case Kind::Sub:
    return evalNodeVec(*node.operands[0], c) - evalNodeVec(*node.operands[1], c);
  case Kind::Div:
    // Matches the old OpDiv (`b ? a / b : 0`). A vectorized evaluation touches
    // every lane and cannot short-circuit past a zero divisor the way a scalar
    // predicate would, so the guard is not optional.
    return vecSafeDiv(evalNodeVec(*node.operands[0], c),
                      evalNodeVec(*node.operands[1], c));
  case Kind::Cmp:
    llvm_unreachable("Cmp is not an arithmetic node; use evalCmpNodeInto");
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

ParmValue evalNodeScalar(const ConstraintNode &node, const ConfWrapper &c) {
  using Kind = ConstraintNode::Kind;
  switch (node.kind) {
  case Kind::Const:
    return node.value;
  case Kind::Var:
    return c[*node.varIdx];
  case Kind::Add: {
    ParmValue acc = 0;
    for (const auto &op : node.operands)
      acc += evalNodeScalar(*op, c);
    return acc;
  }
  case Kind::Mul: {
    ParmValue acc = 1;
    for (const auto &op : node.operands)
      acc *= evalNodeScalar(*op, c);
    return acc;
  }
  case Kind::Sub:
    return evalNodeScalar(*node.operands[0], c) -
           evalNodeScalar(*node.operands[1], c);
  case Kind::Div: {
    ParmValue d = evalNodeScalar(*node.operands[1], c);
    return d ? evalNodeScalar(*node.operands[0], c) / d : 0;
  }
  case Kind::Cmp:
    llvm_unreachable("Cmp is not an arithmetic node");
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

void evalCmpNodeInto(const ConstraintNode &node, const ConfigurationVector &c,
                     arma::urowvec &valid) {
  assert(node.kind == ConstraintNode::Kind::Cmp && "expected a Cmp node");
  const ParmVector lhs = evalNodeVec(*node.operands[0], c);
  const ParmVector rhs = evalNodeVec(*node.operands[1], c);
  // `%=` is Armadillo's elementwise multiply, i.e. AND over 0/1 masks: a lane
  // already cleared by an earlier constraint stays cleared.
  switch (node.cmp) {
  case CmpKind::Le:
    valid %= (lhs <= rhs);
    return;
  case CmpKind::Ge:
    valid %= (lhs >= rhs);
    return;
  case CmpKind::Lt:
    valid %= (lhs < rhs);
    return;
  case CmpKind::Gt:
    valid %= (lhs > rhs);
    return;
  case CmpKind::Eq:
    valid %= (lhs == rhs);
    return;
  case CmpKind::Ne:
    valid %= (lhs != rhs);
    return;
  }
  llvm_unreachable("unknown CmpKind");
}

VecConstraint toVecConstraint(ConstraintNodePtr node) {
  assert(node && node->kind == ConstraintNode::Kind::Cmp &&
         "a constraint must be a comparison");
  return [node = std::move(node)](const ConfigurationVector &c,
                                  arma::urowvec &valid) {
    evalCmpNodeInto(*node, c, valid);
  };
}

// ===----------------------------------------------------------------------===//
// Printing
// ===----------------------------------------------------------------------===//

std::string describeNode(const ConstraintNode &node) {
  using Kind = ConstraintNode::Kind;
  auto joinOperands = [&](const char *sep) {
    std::string s = "(";
    for (size_t i = 0; i < node.operands.size(); ++i) {
      if (i)
        s += sep;
      s += describeNode(*node.operands[i]);
    }
    return s + ")";
  };

  switch (node.kind) {
  case Kind::Const:
    return std::to_string(node.value);
  case Kind::Var:
    return node.varName;
  case Kind::Add:
    return joinOperands(" + ");
  case Kind::Mul:
    return joinOperands(" * ");
  case Kind::Sub:
    return joinOperands(" - ");
  case Kind::Div:
    return joinOperands(" / ");
  case Kind::Cmp:
    return describeNode(*node.operands[0]) + cmpSymbol(node.cmp) +
           describeNode(*node.operands[1]);
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

} // namespace mlir::cinm
