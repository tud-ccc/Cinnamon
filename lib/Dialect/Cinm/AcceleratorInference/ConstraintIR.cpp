#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/ConstraintIR.h"

#include <llvm/Support/ErrorHandling.h>

namespace mlir::cinm::constraints {
using Kind = ConstraintNode::Kind;

// ===----------------------------------------------------------------------===//
// Evaluation
// ===----------------------------------------------------------------------===//

bool containsDivision(const ConstraintNode &node) {
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

ParmValue evalNodeScalar(const ConstraintNode &node, const ConfWrapper &c) {
  using Kind = Kind;
  switch (node.kind) {
  case Kind::Const:
    return node.constValue();
  case Kind::Var:
    return c[node.varIdx()];
  case Kind::Add: {
    ParmValue acc = 0;
    for (const auto &op : node.operands())
      acc += evalNodeScalar(*op, c);
    return acc;
  }
  case Kind::Mul: {
    ParmValue acc = 1;
    for (const auto &op : node.operands())
      acc *= evalNodeScalar(*op, c);
    return acc;
  }
  case Kind::Div: {
    ParmValue d = evalNodeScalar(*node.operands()[1], c);
    return d ? evalNodeScalar(*node.operands()[0], c) / d : 0;
  }
  default:
    llvm_unreachable("boolean node evaluated as arithmetic");
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

// ===----------------------------------------------------------------------===//
// Analysis — identities (product equalities)
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
    r.numer.coeff = node.constValue();
    return r;
  }
  case Kind::Var: {
    Rational r;
    r.numer.vars.push_back(node.varIdx());
    return r;
  }
  case Kind::Mul: {
    Rational acc;
    for (const auto &op : node.operands()) {
      auto sub = normalizeImpl(*op);
      if (!sub)
        return std::nullopt;
      mulInto(acc.numer, sub->numer);
      mulInto(acc.denom, sub->denom);
    }
    return acc;
  }
  case Kind::Div: {
    auto lhs = normalizeImpl(*node.operands()[0]);
    auto rhs = normalizeImpl(*node.operands()[1]);
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
  // (inequalities) out of identity matching rather than silently
  // mis-analysing them.
  default:
    return std::nullopt;
  }
  llvm_unreachable("unknown Kind");
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
  if (node.kind != Kind::Eq)
    return std::nullopt;
  auto lhs = normalizeImpl(*node.operands()[0]);
  auto rhs = normalizeImpl(*node.operands()[1]);
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
  switch (node.kind) {
  case Kind::Const:
    return {node.constValue(), node.constValue(), true};
  case Kind::Var:
    return bounds(node.varIdx());
  case Kind::Add: {
    Interval acc{0, 0, true};
    for (const auto &op : node.operands()) {
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
    for (const auto &op : node.operands()) {
      Interval s = evalNodeBounds(*op, bounds);
      if (!s.valid || s.lo < 0)
        return kUnbounded;
      acc.lo *= s.lo;
      acc.hi *= s.hi;
    }
    return acc;
  }
  case Kind::Div: {
    // Decreasing in the divisor, so the bounds cross over. A divisor whose
    // range includes zero yields nothing usable.
    Interval a = evalNodeBounds(*node.operands()[0], bounds);
    Interval b = evalNodeBounds(*node.operands()[1], bounds);
    if (!a.valid || !b.valid || a.lo < 0 || b.lo <= 0)
      return kUnbounded;
    return {a.lo / b.hi, a.hi / b.lo, true};
  }
  default:
    return kUnbounded;
  }
  llvm_unreachable("unknown ConstraintNode::Kind");
}

/// Whether every division below `node` comes out exact, as far as `bounds`
/// can tell: true/false when decided, nullopt when not.
///
/// Only a division whose two operands are both pinned to a single value can be
/// decided this way -- but that is the case that matters, because at a full
/// assignment every operand is pinned, which is what lets a component enforce a
/// dividing comparison outright instead of leaving it to be filtered later.
static std::optional<bool> divisionsExact(const ConstraintNode &node,
                                          const VarBounds &bounds) {
  bool allKnown = true;
  if (node.kind == Kind::Div) {
    Interval num = evalNodeBounds(*node.operands()[0], bounds);
    Interval den = evalNodeBounds(*node.operands()[1], bounds);
    if (num.valid && den.valid && num.lo == num.hi && den.lo == den.hi) {
      if (den.lo == 0 || num.lo % den.lo != 0)
        return false;
    } else {
      allKnown = false;
    }
  }
  for (const auto &child : node.operands()) {
    std::optional<bool> sub = divisionsExact(*child, bounds);
    if (sub && !*sub)
      return false;
    if (!sub)
      allKnown = false;
  }
  if (!allKnown)
    return std::nullopt;
  return true;
}

/// A divisibility test decided by the bounds, or nullopt if they do not decide
/// it. Only the fully-pinned case is decided -- which is the case that matters,
/// since at a full assignment every operand is pinned.
static std::optional<bool> dividesDecided(const ConstraintNode &node,
                                          const VarBounds &bounds) {
  Interval d = evalNodeBounds(*node.operands()[0], bounds);
  Interval n = evalNodeBounds(*node.operands()[1], bounds);
  if (!d.valid || !n.valid || d.lo != d.hi || n.lo != n.hi)
    return std::nullopt;
  return d.lo != 0 && n.lo % d.lo == 0;
}

bool boolMayHold(const ConstraintNode &node, const VarBounds &bounds) {
  if (node.kind == Kind::Divides) {
    if (divisionsExact(node, bounds) == std::optional<bool>(false))
      return false;
    std::optional<bool> decided = dividesDecided(node, bounds);
    return !decided || *decided;
  }
  if (node.kind == Kind::Implies)
    // Satisfiable unless the antecedent is forced and the consequent
    // impossible.
    return !boolMustHold(*node.operands()[0], bounds) ||
           boolMayHold(*node.operands()[1], bounds);

  assert(ConstraintNode::isBoolKind(node.kind) && "expected a boolean node");
  // A division known not to come out exact makes the comparison false outright,
  // whatever the truncated quotient compares to (see evalBoolNodeVec).
  if (divisionsExact(node, bounds) == std::optional<bool>(false))
    return false;

  Interval l = evalNodeBounds(*node.operands()[0], bounds);
  Interval r = evalNodeBounds(*node.operands()[1], bounds);
  if (!l.valid || !r.valid)
    return true; // no information; never prune on a guess

  switch (node.kind) {
  case Kind::Le:
    return l.lo <= r.hi;
  case Kind::Lt:
    return l.lo < r.hi;
  case Kind::Ge:
    return l.hi >= r.lo;
  case Kind::Gt:
    return l.hi > r.lo;
  case Kind::Eq:
    return l.lo <= r.hi && r.lo <= l.hi; // the ranges must overlap
  case Kind::Ne:
    // Only unsatisfiable when both sides are pinned to the same value.
    return !(l.lo == l.hi && r.lo == r.hi && l.lo == r.lo);
  default:
    llvm_unreachable("unknown ConstraintNode::Kind");
  }
}

bool boolMustHold(const ConstraintNode &node, const VarBounds &bounds) {
  if (node.kind == Kind::Divides) {
    if (divisionsExact(node, bounds) != std::optional<bool>(true))
      return false;
    std::optional<bool> decided = dividesDecided(node, bounds);
    return decided && *decided;
  }
  if (node.kind == Kind::Implies)
    return !boolMayHold(*node.operands()[0], bounds) ||
           boolMustHold(*node.operands()[1], bounds);

  assert(ConstraintNode::isBoolKind(node.kind) && "expected a boolean node");
  // evalNodeBounds divides by truncation, so the interval it gives for a
  // quotient covers values the comparison would reject as inexact. Harmless for
  // boolMayHold, whose intervals only need to be a superset; here it would
  // claim a comparison holds on lanes where it does not. So every division has
  // to be known exact before the intervals mean anything.
  if (divisionsExact(node, bounds) != std::optional<bool>(true))
    return false;

  Interval l = evalNodeBounds(*node.operands()[0], bounds);
  Interval r = evalNodeBounds(*node.operands()[1], bounds);
  if (!l.valid || !r.valid)
    return false; // no information; never claim more than is known

  switch (node.kind) {
  case Kind::Le:
    return l.hi <= r.lo;
  case Kind::Lt:
    return l.hi < r.lo;
  case Kind::Ge:
    return l.lo >= r.hi;
  case Kind::Gt:
    return l.lo > r.hi;
  case Kind::Eq:
    // Both sides pinned to one value, and the same one.
    return l.lo == l.hi && r.lo == r.hi && l.lo == r.lo;
  case Kind::Ne:
    // The ranges must not overlap at all.
    return l.hi < r.lo || r.hi < l.lo;
  default:
    llvm_unreachable("unknown ConstraintNode::Kind");
  }
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
