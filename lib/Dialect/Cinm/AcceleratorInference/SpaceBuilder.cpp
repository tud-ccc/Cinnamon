#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <algorithm>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// SpaceExpr — construction and evaluation
// ===----------------------------------------------------------------------===//

using Node    = detail::ExprNode;
using NodePtr = detail::NodePtr;

namespace {
NodePtr makeNode(Node::Kind k, int64_t v = 0) {
  auto n    = std::make_shared<Node>();
  n->kind   = k;
  n->constVal = v;
  return n;
}

NodePtr binNode(Node::Kind k, NodePtr lhs, NodePtr rhs) {
  auto n  = std::make_shared<Node>();
  n->kind = k;
  n->lhs  = std::move(lhs);
  n->rhs  = std::move(rhs);
  return n;
}

int64_t evalNode(const Node &root, const ConfWrapper &c) {
  struct Frame { const Node *node; bool visited; };
  llvm::SmallVector<Frame, 16> work;
  llvm::SmallVector<int64_t, 8> vals;
  work.push_back({&root, false});

  while (!work.empty()) {
    const auto [node, visited] = work.back();
    work.pop_back();

    if (node->kind == Node::Const) { vals.push_back(node->constVal); continue; }
    if (node->kind == Node::Var)   { vals.push_back(node->var.get(c)); continue; }

    if (!visited) {
      work.push_back({node, true});
      work.push_back({node->rhs.get(), false}); // right evaluated before revisit
      work.push_back({node->lhs.get(), false}); // left evaluated first (on top)
    } else {
      int64_t rhs = vals.pop_back_val();
      int64_t lhs = vals.pop_back_val();
      switch (node->kind) {
      case Node::Add: vals.push_back(lhs + rhs); break;
      case Node::Sub: vals.push_back(lhs - rhs); break;
      case Node::Mul: vals.push_back(lhs * rhs); break;
      case Node::Div: vals.push_back(rhs != 0 ? lhs / rhs : 0); break;
      default: llvm_unreachable("unknown binary ExprNode kind");
      }
    }
  }
  return vals.back();
}
} // namespace

SpaceExpr::SpaceExpr(int64_t constant) {
  auto n       = makeNode(Node::Const, constant);
  root_        = std::move(n);
}

SpaceExpr::SpaceExpr(const SpaceVar &var) {
  auto n  = std::make_shared<Node>();
  n->kind = Node::Var;
  n->var  = var;
  root_   = std::move(n);
}

int64_t SpaceExpr::eval(const ConfWrapper &c) const {
  return evalNode(*root_, c);
}

// ===----------------------------------------------------------------------===//
// Arithmetic operators
// ===----------------------------------------------------------------------===//

SpaceExpr operator+(SpaceExpr lhs, SpaceExpr rhs) {
  return SpaceExpr(binNode(Node::Add, lhs.root_, rhs.root_));
}
SpaceExpr operator-(SpaceExpr lhs, SpaceExpr rhs) {
  return SpaceExpr(binNode(Node::Sub, lhs.root_, rhs.root_));
}
SpaceExpr operator*(SpaceExpr lhs, SpaceExpr rhs) {
  return SpaceExpr(binNode(Node::Mul, lhs.root_, rhs.root_));
}
SpaceExpr operator/(SpaceExpr lhs, SpaceExpr rhs) {
  return SpaceExpr(binNode(Node::Div, lhs.root_, rhs.root_));
}

// ===----------------------------------------------------------------------===//
// Comparison operators → ConstraintExpr
// ===----------------------------------------------------------------------===//

ConstraintExpr operator<=(SpaceExpr lhs, SpaceExpr rhs) {
  return ConstraintExpr(ConstraintExpr::Le, std::move(lhs), std::move(rhs));
}
ConstraintExpr operator>=(SpaceExpr lhs, SpaceExpr rhs) {
  return ConstraintExpr(ConstraintExpr::Ge, std::move(lhs), std::move(rhs));
}
ConstraintExpr operator<(SpaceExpr lhs, SpaceExpr rhs) {
  return ConstraintExpr(ConstraintExpr::Lt, std::move(lhs), std::move(rhs));
}
ConstraintExpr operator>(SpaceExpr lhs, SpaceExpr rhs) {
  return ConstraintExpr(ConstraintExpr::Gt, std::move(lhs), std::move(rhs));
}
ConstraintExpr operator==(SpaceExpr lhs, SpaceExpr rhs) {
  return ConstraintExpr(ConstraintExpr::Eq, std::move(lhs), std::move(rhs));
}
ConstraintExpr operator!=(SpaceExpr lhs, SpaceExpr rhs) {
  return ConstraintExpr(ConstraintExpr::Ne, std::move(lhs), std::move(rhs));
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder — dimension declaration
// ===----------------------------------------------------------------------===//

SpaceVar SpaceBuilder::intRange(llvm::StringRef name, int64_t lo, int64_t hi) {
  SpaceVar v(name, hi);
  dims_.push_back({v, DimEntry::IntRange, lo, hi, {}});
  return v;
}

SpaceVar SpaceBuilder::pow2Range(llvm::StringRef name, int expLo, int expHi) {
  SpaceVar v(name, int64_t{1} << expHi);
  dims_.push_back({v, DimEntry::Pow2, expLo, expHi, {}});
  return v;
}

SpaceVar SpaceBuilder::divisorsOf(llvm::StringRef name, int64_t n) {
  SpaceVar v(name, n);
  dims_.push_back({v, DimEntry::DivisorsOfConst, 1, n, {n}});
  return v;
}

SpaceVar SpaceBuilder::divisorsOf(llvm::StringRef name, SpaceVar src) {
  SpaceVar v(name, src.maxVal());
  dims_.push_back({v, DimEntry::IntRange, 1, src.maxVal(), {}});
  multiples_.push_back({v.name_, src.name_});
  return v;
}

SpaceBuilder::DimEntry &SpaceBuilder::findEntry(const SpaceVar &v) {
  for (auto &e : dims_)
    if (e.var.idx_ == v.idx_)
      return e;
  llvm_unreachable("SpaceVar not found in SpaceBuilder");
}

SpaceVar SpaceBuilder::findVarByName(llvm::StringRef name) const {
  for (const auto &e : dims_)
    if (e.var.name_ == name.str())
      return e.var;
  llvm_unreachable("dim name not found in SpaceBuilder");
}

int SpaceBuilder::dimIndexByName(llvm::StringRef name) const {
  for (int i = 0; i < (int)dims_.size(); ++i)
    if (dims_[i].var.name_ == name.str())
      return i;
  return -1;
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder — constraint declaration
// ===----------------------------------------------------------------------===//

void SpaceBuilder::mustDivide(SpaceVar v, int64_t n) {
  if (!ShapedType::isDynamic(n))
    findEntry(v).divisorFilters.push_back(n);
}

void SpaceBuilder::mustDivide(SpaceVar parent, SpaceVar child) {
  multiples_.push_back({parent.name_, child.name_});
}

void SpaceBuilder::require(Constraint pred) {
  predicates_.push_back(std::move(pred));
}

// ===----------------------------------------------------------------------===//
// Expression constraint extraction
// ===----------------------------------------------------------------------===//

void SpaceBuilder::addDivConstraint(const NodePtr &num, const NodePtr &den) {
  if (num->kind == Node::Const && den->kind == Node::Var) {
    // den must divide num  (static filter on den's values)
    LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   div constraint: " << den->var.name()
                            << " | " << num->constVal << "  (static filter)\n");
    mustDivide(den->var, num->constVal);
  } else if (num->kind == Node::Var && den->kind == Node::Var) {
    // den divides num at runtime  (structural constraint)
    LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   div constraint: " << den->var.name()
                            << " | " << num->var.name() << "  (structural)\n");
    mustDivide(den->var, num->var);
  } else {
    // compound expressions: register a dynamic divisibility predicate
    LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   div constraint: compound  (dynamic predicate)\n");
    SpaceExpr numExpr(num), denExpr(den);
    predicates_.push_back([numExpr, denExpr](const ConfWrapper &c) {
      auto dv = denExpr.eval(c);
      return dv != 0 && numExpr.eval(c) % dv == 0;
    });
  }
}

void SpaceBuilder::extractDivConstraints(const NodePtr &node) {
  if (!node || node->kind == Node::Const || node->kind == Node::Var)
    return;
  if (node->kind == Node::Div)
    addDivConstraint(node->lhs, node->rhs);
  extractDivConstraints(node->lhs);
  extractDivConstraints(node->rhs);
}

void SpaceBuilder::require(SpaceExpr expr) {
  extractDivConstraints(expr.root_);
}

void SpaceBuilder::require(ConstraintExpr expr) {
  extractDivConstraints(expr.lhs_.root_);
  extractDivConstraints(expr.rhs_.root_);
  auto lhs  = expr.lhs_;
  auto rhs  = expr.rhs_;
  auto kind = expr.kind_;
  predicates_.push_back([lhs, rhs, kind](const ConfWrapper &c) {
    auto lv = lhs.eval(c), rv = rhs.eval(c);
    switch (kind) {
    case ConstraintExpr::Le: return lv <= rv;
    case ConstraintExpr::Ge: return lv >= rv;
    case ConstraintExpr::Lt: return lv < rv;
    case ConstraintExpr::Gt: return lv > rv;
    case ConstraintExpr::Eq: return lv == rv;
    case ConstraintExpr::Ne: return lv != rv;
    }
    return false;
  });
}

// ===----------------------------------------------------------------------===//
// SpaceBuilder::buildInto
// ===----------------------------------------------------------------------===//

void SpaceBuilder::buildInto(ConfigSpace &space) {
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space] building config space:\n");

  // Phase 1: build each SearchParam, deduplicate + apply static filters, addDim.
  for (auto &entry : dims_) {
    SearchParam param = [&]() -> SearchParam {
      switch (entry.kind) {
      case DimEntry::IntRange:
      case DimEntry::DivisorsOfConst:
        return makeRange(entry.var.name_, entry.lo, entry.hi);
      case DimEntry::Pow2:
        return makePow2Range(entry.var.name_, entry.lo, entry.hi);
      }
      llvm_unreachable("unknown DimKind");
    }();

    std::sort(entry.divisorFilters.begin(), entry.divisorFilters.end());
    entry.divisorFilters.erase(
        std::unique(entry.divisorFilters.begin(), entry.divisorFilters.end()),
        entry.divisorFilters.end());
    for (int64_t n : entry.divisorFilters)
      param.keepDivisorsOf(n);

    LLVM_DEBUG({
      llvm::dbgs() << "[cinm-space]   dim '" << entry.var.name_ << "': ";
      switch (entry.kind) {
      case DimEntry::IntRange:        llvm::dbgs() << "int[" << entry.lo << ".." << entry.hi << "]"; break;
      case DimEntry::DivisorsOfConst: llvm::dbgs() << "divisors[" << entry.lo << ".." << entry.hi << "]"; break;
      case DimEntry::Pow2:            llvm::dbgs() << "pow2[2^" << entry.lo << "..2^" << entry.hi << "]"; break;
      }
      if (!entry.divisorFilters.empty()) {
        llvm::dbgs() << "  filters=divisorsOf{";
        for (size_t i = 0; i < entry.divisorFilters.size(); ++i) {
          if (i) llvm::dbgs() << ",";
          llvm::dbgs() << entry.divisorFilters[i];
        }
        llvm::dbgs() << "}";
      }
      llvm::dbgs() << "\n";
    });

    *entry.var.idx_ = space.addDim(std::move(param));
  }

  // Phase 2: analyze and commit structural multiples constraints.
  // Deduplicate first.
  std::sort(multiples_.begin(), multiples_.end());
  multiples_.erase(std::unique(multiples_.begin(), multiples_.end()),
                   multiples_.end());

  // Build lookup for the full set.
  std::set<std::pair<std::string, std::string>> multsSet;
  for (auto &m : multiples_)
    multsSet.insert({m.parent, m.child});

  // childSet tracks dims already committed as structural children; these cannot
  // be structural parents (the encoding has no slot for them, so their child
  // would never be decoded).
  std::set<std::string> childSet;

  // Pairs replaced by fallback constraints (indices valid after phase 1).
  std::vector<std::pair<SpaceVar, SpaceVar>> equalityFallbacks;    // A == B
  std::vector<std::pair<SpaceVar, SpaceVar>> dynamicDivFallbacks;  // child % parent == 0

  std::set<std::pair<std::string, std::string>> handled;

  for (auto &m : multiples_) {
    if (handled.count({m.parent, m.child}))
      continue;

    // Note if parent was declared after child (informational: encoding is
    // index-agnostic and handles this correctly).
    LLVM_DEBUG({
      int pi = dimIndexByName(m.parent), ci = dimIndexByName(m.child);
      if (pi > ci)
        llvm::dbgs() << "[cinm-space]   note: '" << m.parent
                     << "' (dim " << pi << ") declared after child '"
                     << m.child << "' (dim " << ci << ") — OK for encoding\n";
    });

    // Detect mutual divisibility: A|B AND B|A → implies A == B.
    if (multsSet.count({m.child, m.parent})) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-space]   WARNING: mutual divisibility '"
                 << m.parent << "' | '" << m.child << "' AND '" << m.child
                 << "' | '" << m.parent
                 << "'  (implies equality; replacing both with dynamic A==B)\n");
      handled.insert({m.parent, m.child});
      handled.insert({m.child, m.parent});
      equalityFallbacks.push_back(
          {findVarByName(m.parent), findVarByName(m.child)});
      continue;
    }

    // Detect chains: parent is already a structural child — the encoding has
    // no slot for it, so its own child would never be decoded.
    if (childSet.count(m.parent)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-space]   WARNING: chained divisibility '"
                 << m.parent << "' | '" << m.child << "' where '" << m.parent
                 << "' is already a structural child"
                 << "  (converting to dynamic predicate)\n");
      handled.insert({m.parent, m.child});
      dynamicDivFallbacks.push_back(
          {findVarByName(m.parent), findVarByName(m.child)});
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   structural: '" << m.parent
                            << "' | '" << m.child << "'\n");
    space.addMultiplesConstraint(m.parent, m.child);
    childSet.insert(m.child);
  }

  // Add fallback dynamic predicates (indices set during phase 1).
  for (auto [va, vb] : equalityFallbacks)
    space.addConstraint(
        [va, vb](const ConfWrapper &c) { return va[c] == vb[c]; });
  for (auto [parent, child] : dynamicDivFallbacks)
    space.addConstraint(
        [parent, child](const ConfWrapper &c) { return child[c] % parent[c] == 0; });

  // Phase 3: dynamic predicates.
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   dynamic predicates: "
                          << predicates_.size() << "\n");
  for (auto &pred : predicates_)
    space.addConstraint(Constraint(pred));
}

} // namespace mlir::cinm
