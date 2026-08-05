#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <llvm/Support/ErrorHandling.h>

namespace mlir::cinm::constraints {

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

ConstraintNodePtr makeImpliesNode(ConstraintNodePtr antecedent,
                                  ConstraintNodePtr consequent) {
  assert(antecedent && consequent && isBoolKind(antecedent->kind) &&
         isBoolKind(consequent->kind) &&
         "an implication relates two boolean nodes");
  ConstraintNode n;
  n.kind = ConstraintNode::Kind::Implies;
  n.operands = {std::move(antecedent), std::move(consequent)};
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
    return evalNodeVec(*node.operands[0], c) -
           evalNodeVec(*node.operands[1], c);
  case Kind::Div:
    // Matches the old OpDiv (`b ? a / b : 0`). A vectorized evaluation touches
    // every lane and cannot short-circuit past a zero divisor the way a scalar
    // predicate would, so the guard is not optional.
    return vecSafeDiv(evalNodeVec(*node.operands[0], c),
                      evalNodeVec(*node.operands[1], c));
  case Kind::Cmp:
  case Kind::Implies:
    llvm_unreachable(
        "boolean node evaluated as arithmetic; use evalBoolNodeVec");
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
  case Kind::Implies:
    llvm_unreachable("boolean node evaluated as arithmetic");
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

arma::urowvec evalBoolNodeVec(const ConstraintNode &node,
                              const ConfigurationVector &c) {
  if (node.kind == ConstraintNode::Kind::Implies) {
    // `a => b` is `!a | b`. Both sides are evaluated for every lane: a
    // vectorized evaluation has no short-circuit, and the antecedent's own
    // operands are arithmetic, so there is nothing unsafe to guard against.
    const arma::urowvec ante = evalBoolNodeVec(*node.operands[0], c);
    const arma::urowvec cons = evalBoolNodeVec(*node.operands[1], c);
    return (ante == 0) || (cons != 0);
  }

  assert(node.kind == ConstraintNode::Kind::Cmp && "expected a boolean node");
  const ParmVector lhs = evalNodeVec(*node.operands[0], c);
  const ParmVector rhs = evalNodeVec(*node.operands[1], c);
  switch (node.cmp) {
  case CmpKind::Le:
    return lhs <= rhs;
  case CmpKind::Ge:
    return lhs >= rhs;
  case CmpKind::Lt:
    return lhs < rhs;
  case CmpKind::Gt:
    return lhs > rhs;
  case CmpKind::Eq:
    return lhs == rhs;
  case CmpKind::Ne:
    return lhs != rhs;
  }
  llvm_unreachable("unknown CmpKind");
}

void evalBoolNodeInto(const ConstraintNode &node, const ConfigurationVector &c,
                      arma::urowvec &valid) {
  // `%=` is Armadillo's elementwise multiply, i.e. AND over 0/1 masks: a lane
  // already cleared by an earlier constraint stays cleared.
  valid %= evalBoolNodeVec(node, c);
}

VecConstraint toVecConstraint(ConstraintNodePtr node) {
  assert(node && isBoolKind(node->kind) &&
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
  case Kind::Implies:
    return "(" + describeNode(*node.operands[0]) + " => " +
           describeNode(*node.operands[1]) + ")";
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
  case Kind::Implies:
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

// ===----------------------------------------------------------------------===//
// Analysis — interval bounds
// ===----------------------------------------------------------------------===//

static const Interval kUnbounded = {0, 0, false};

Interval evalNodeBounds(const ConstraintNode &node, const VarBounds &bounds) {
  using Kind = ConstraintNode::Kind;
  switch (node.kind) {
  case Kind::Const:
    return {node.value, node.value, true};
  case Kind::Var:
    return bounds(*node.varIdx);
  case Kind::Add: {
    Interval acc{0, 0, true};
    for (const auto &op : node.operands) {
      Interval s = evalNodeBounds(*op, bounds);
      if (!s.valid)
        return kUnbounded;
      acc.lo += s.lo;
      acc.hi += s.hi;
    }
    return acc;
  }
  case Kind::Mul: {
    // Only sound for non-negative operands; every search parameter here is a
    // size or a count, but check rather than assume.
    Interval acc{1, 1, true};
    for (const auto &op : node.operands) {
      Interval s = evalNodeBounds(*op, bounds);
      if (!s.valid || s.lo < 0)
        return kUnbounded;
      acc.lo *= s.lo;
      acc.hi *= s.hi;
    }
    return acc;
  }
  case Kind::Sub: {
    Interval a = evalNodeBounds(*node.operands[0], bounds);
    Interval b = evalNodeBounds(*node.operands[1], bounds);
    if (!a.valid || !b.valid)
      return kUnbounded;
    return {a.lo - b.hi, a.hi - b.lo, true};
  }
  case Kind::Div: {
    // Decreasing in the divisor, so the bounds cross over. A divisor whose
    // range includes zero yields nothing usable.
    Interval a = evalNodeBounds(*node.operands[0], bounds);
    Interval b = evalNodeBounds(*node.operands[1], bounds);
    if (!a.valid || !b.valid || a.lo < 0 || b.lo <= 0)
      return kUnbounded;
    return {a.lo / b.hi, a.hi / b.lo, true};
  }
  case Kind::Cmp:
  case Kind::Implies:
    return kUnbounded;
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

bool boolMayHold(const ConstraintNode &node, const VarBounds &bounds) {
  if (node.kind == ConstraintNode::Kind::Implies)
    // Satisfiable unless the antecedent is forced and the consequent
    // impossible.
    return !boolMustHold(*node.operands[0], bounds) ||
           boolMayHold(*node.operands[1], bounds);

  assert(node.kind == ConstraintNode::Kind::Cmp && "expected a boolean node");
  Interval l = evalNodeBounds(*node.operands[0], bounds);
  Interval r = evalNodeBounds(*node.operands[1], bounds);
  if (!l.valid || !r.valid)
    return true; // no information; never prune on a guess

  switch (node.cmp) {
  case CmpKind::Le:
    return l.lo <= r.hi;
  case CmpKind::Lt:
    return l.lo < r.hi;
  case CmpKind::Ge:
    return l.hi >= r.lo;
  case CmpKind::Gt:
    return l.hi > r.lo;
  case CmpKind::Eq:
    return l.lo <= r.hi && r.lo <= l.hi; // the ranges must overlap
  case CmpKind::Ne:
    // Only unsatisfiable when both sides are pinned to the same value.
    return !(l.lo == l.hi && r.lo == r.hi && l.lo == r.lo);
  }
  llvm_unreachable("unknown CmpKind");
}

bool boolMustHold(const ConstraintNode &node, const VarBounds &bounds) {
  if (node.kind == ConstraintNode::Kind::Implies)
    return !boolMayHold(*node.operands[0], bounds) ||
           boolMustHold(*node.operands[1], bounds);

  assert(node.kind == ConstraintNode::Kind::Cmp && "expected a boolean node");
  Interval l = evalNodeBounds(*node.operands[0], bounds);
  Interval r = evalNodeBounds(*node.operands[1], bounds);
  if (!l.valid || !r.valid)
    return false; // no information; never claim more than is known

  switch (node.cmp) {
  case CmpKind::Le:
    return l.hi <= r.lo;
  case CmpKind::Lt:
    return l.hi < r.lo;
  case CmpKind::Ge:
    return l.lo >= r.hi;
  case CmpKind::Gt:
    return l.lo > r.hi;
  case CmpKind::Eq:
    // Both sides pinned to one value, and the same one.
    return l.lo == l.hi && r.lo == r.hi && l.lo == r.lo;
  case CmpKind::Ne:
    // The ranges must not overlap at all.
    return l.hi < r.lo || r.hi < l.lo;
  }
  llvm_unreachable("unknown CmpKind");
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
    s +=
        v < paramNames.size() ? paramNames[v] : ("<" + std::to_string(v) + ">");
    first = false;
  }
  return s;
}

} // namespace mlir::cinm::constraints
