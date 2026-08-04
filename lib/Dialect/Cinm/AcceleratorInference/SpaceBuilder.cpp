#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/SpaceBuilder.h"

#include <algorithm>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <set>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

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

void SpaceBuilder::require(Constraint pred, llvm::StringRef description) {
  predicates_.push_back({description.str(), std::move(pred)});
}

void SpaceBuilder::requireVec(VecConstraint pred, llvm::StringRef description) {
  vecPredicates_.push_back({description.str(), std::move(pred)});
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
      case DimEntry::IntRange:
        llvm::dbgs() << "int[" << entry.lo << ".." << entry.hi << "]";
        break;
      case DimEntry::DivisorsOfConst:
        llvm::dbgs() << "divisors[" << entry.lo << ".." << entry.hi << "]";
        break;
      case DimEntry::Pow2:
        llvm::dbgs() << "pow2[2^" << entry.lo << "..2^" << entry.hi << "]";
        break;
      }
      if (!entry.divisorFilters.empty()) {
        llvm::dbgs() << "  filters=divisorsOf{";
        for (size_t i = 0; i < entry.divisorFilters.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
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

  // childSet tracks dims already committed as structural children.
  std::set<std::string> childSet;

  std::vector<std::pair<SpaceVar, SpaceVar>> equalityFallbacks;
  std::vector<std::pair<SpaceVar, SpaceVar>> dynamicDivFallbacks;

  std::set<std::pair<std::string, std::string>> handled;

  for (auto &m : multiples_) {
    if (handled.count({m.parent, m.child}))
      continue;

    LLVM_DEBUG({
      int pi = dimIndexByName(m.parent), ci = dimIndexByName(m.child);
      if (pi > ci)
        llvm::dbgs() << "[cinm-space]   note: '" << m.parent << "' (dim " << pi
                     << ") declared after child '" << m.child << "' (dim " << ci
                     << ") — OK for encoding\n";
    });

    // Detect mutual divisibility: A|B AND B|A → implies A == B.
    if (multsSet.count({m.child, m.parent})) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-space]   WARNING: mutual divisibility '" << m.parent
                 << "' | '" << m.child << "' AND '" << m.child << "' | '"
                 << m.parent
                 << "'  (implies equality; replacing both with dynamic A==B)\n");
      handled.insert({m.parent, m.child});
      handled.insert({m.child, m.parent});
      equalityFallbacks.push_back(
          {findVarByName(m.parent), findVarByName(m.child)});
      continue;
    }

    // Detect chains: parent is already a structural child.
    if (childSet.count(m.parent)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cinm-space]   WARNING: chained divisibility '" << m.parent
                 << "' | '" << m.child << "' where '" << m.parent
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

  // Add fallback dynamic predicates.
  for (auto [va, vb] : equalityFallbacks)
    space.addConstraint(
        [va, vb](const ConfWrapper &c) { return va[c] == vb[c]; },
        va.name().str() + " == " + vb.name().str());
  for (auto [parent, child] : dynamicDivFallbacks)
    space.addConstraint(
        [parent, child](const ConfWrapper &c) {
          return child[c] % parent[c] == 0;
        },
        parent.name().str() + " | " + child.name().str());

  // Phase 3: dynamic predicates.
  LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   dynamic predicates: "
                          << predicates_.size() << "\n");
  for (auto &[desc, pred] : predicates_)
    space.addConstraint(Constraint(pred), desc);

  LLVM_DEBUG(llvm::dbgs() << "[cinm-space]   vectorized predicates: "
                          << vecPredicates_.size() << "\n");
  for (auto &[desc, pred] : vecPredicates_)
    space.addVecConstraint(VecConstraint(pred), desc);
}

} // namespace mlir::cinm
