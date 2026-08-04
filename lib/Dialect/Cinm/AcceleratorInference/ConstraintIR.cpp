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

// ===----------------------------------------------------------------------===//
// Analysis — Form A
// ===----------------------------------------------------------------------===//

namespace {
/// numer/denom, each a monomial. Multiplying two rationals multiplies both
/// parts; dividing crosses them over.
struct Rational {
  Monomial numer, denom;
};

void mulInto(Monomial &acc, const Monomial &m) {
  acc.coeff *= m.coeff;
  acc.vars.append(m.vars.begin(), m.vars.end());
}

void sortVars(Monomial &m) { llvm::sort(m.vars); }
} // namespace

static std::optional<Rational> normalizeImpl(const ConstraintNode &node) {
  using Kind = ConstraintNode::Kind;
  switch (node.kind) {
  case Kind::Const: {
    Rational r;
    r.numer.coeff = node.value;
    return r;
  }
  case Kind::Var: {
    Rational r;
    r.numer.vars.push_back(*node.varIdx);
    return r;
  }
  case Kind::Mul: {
    Rational acc;
    for (const auto &op : node.operands) {
      auto sub = normalizeImpl(*op);
      if (!sub)
        return std::nullopt;
      mulInto(acc.numer, sub->numer);
      mulInto(acc.denom, sub->denom);
    }
    return acc;
  }
  case Kind::Div: {
    auto lhs = normalizeImpl(*node.operands[0]);
    auto rhs = normalizeImpl(*node.operands[1]);
    if (!lhs || !rhs)
      return std::nullopt;
    Rational r;
    r.numer = lhs->numer;
    mulInto(r.numer, rhs->denom);
    r.denom = lhs->denom;
    mulInto(r.denom, rhs->numer);
    return r;
  }
  // A sum is not a monomial. This is the line that keeps capacity bounds
  // (Form C) out of Form A rather than silently mis-analysing them.
  case Kind::Add:
  case Kind::Sub:
  case Kind::Cmp:
    return std::nullopt;
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

std::optional<std::pair<Monomial, Monomial>>
normalizeRationalMonomial(const ConstraintNode &node) {
  auto r = normalizeImpl(node);
  if (!r)
    return std::nullopt;
  sortVars(r->numer);
  sortVars(r->denom);
  return std::make_pair(r->numer, r->denom);
}

std::optional<ProductEquality>
matchProductEquality(const ConstraintNode &node) {
  if (node.kind != ConstraintNode::Kind::Cmp || node.cmp != CmpKind::Eq)
    return std::nullopt;
  auto lhs = normalizeImpl(*node.operands[0]);
  auto rhs = normalizeImpl(*node.operands[1]);
  if (!lhs || !rhs)
    return std::nullopt;

  // Cross-multiply to clear denominators:
  //   lhsNum/lhsDen == rhsNum/rhsDen   <=>   lhsNum*rhsDen == rhsNum*lhsDen
  ProductEquality eq;
  eq.lhs = lhs->numer;
  mulInto(eq.lhs, rhs->denom);
  eq.rhs = rhs->numer;
  mulInto(eq.rhs, lhs->denom);
  sortVars(eq.lhs);
  sortVars(eq.rhs);
  return eq;
}

std::string describeMonomial(const Monomial &m,
                             llvm::ArrayRef<std::string> paramNames) {
  std::string s;
  bool first = true;
  if (m.coeff != 1 || m.vars.empty()) {
    s += std::to_string(m.coeff);
    first = false;
  }
  for (size_t v : m.vars) {
    if (!first)
      s += " * ";
    s += v < paramNames.size() ? paramNames[v] : ("<" + std::to_string(v) + ">");
    first = false;
  }
  return s;
}

} // namespace mlir::cinm
