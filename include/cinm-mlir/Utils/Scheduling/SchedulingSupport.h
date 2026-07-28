#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <string>
#include <utility>
#include <variant>

namespace mlir::cinm::utils {

void evolveName(llvm::StringRef name, std::string &result);

class NameInventor {
  using SetType = llvm::SmallSet<llvm::StringRef, 6>;
  SetType usedNames;
  StringRef prefix;
  MLIRContext *context;

public:
  NameInventor(const NameInventor &) = delete;

  NameInventor(SetType &&names, MLIRContext *ctx, StringRef prefix)
      : usedNames(std::move(names)), prefix(prefix), context(ctx) {}
  NameInventor(MLIRContext *ctx, StringRef prefix = "")
      : usedNames(), prefix(prefix), context(ctx) {}

  void addUsedName(llvm::StringRef ref) { usedNames.insert(ref); }

  /// Spawn a unique name. The name will be remembered so subsequent calls
  /// won't collide. Returns the hint unchanged if unused, otherwise appends
  /// a numeric suffix.
  mlir::StringAttr getUniqueName(llvm::StringRef hint = "");

  /// Return a NameInventor seeded from all StringAttr values in the block
  /// containing `loc` that start with `hint`.
  static NameInventor getNameInventor(mlir::Operation *loc,
                                      llvm::StringRef hint);
};

/// If the given symbol is null, invent a new symbol distinct from all other
/// string attributes in the block containing the given operation.
inline StringAttr inventFreshName(Operation *loc, llvm::StringRef hint) {
  NameInventor nameInventor = NameInventor::getNameInventor(loc, hint);
  return nameInventor.getUniqueName("");
}

template <typename T>
using Maybe = std::variant<T, DiagnosedSilenceableFailure>;

/// Check the result of a Maybe<T> and return early if it failed.
#define TRY_GET(expr)                                                          \
  ({                                                                           \
    auto &&_result = (expr);                                                   \
    if (std::holds_alternative<mlir::DiagnosedSilenceableFailure>(_result)) {  \
      return std::move(std::get<1>(_result));                                  \
    }                                                                          \
    std::get<0>(std::move(_result));                                           \
  })

/// Check a DiagnosedSilenceableFailure result and return early if failed.
#define TRY(expr)                                                              \
  ({                                                                           \
    auto &&_result = (expr);                                                   \
    if (!_result.succeeded()) {                                                \
      return std::move(_result);                                               \
    }                                                                          \
  })

/// Check a DiagnosedSilenceableFailure and return llvm::failure() if failed.
#define TRY_REPORT(expr)                                                       \
  ({                                                                           \
    auto &&_result = (expr);                                                   \
    if (!_result.succeeded()) {                                                \
      (void)_result.checkAndReport();                                          \
      return llvm::failure();                                                  \
    }                                                                          \
  })

/// Check a DiagnosedSilenceableFailure inside a walk, interrupting on failure.
#define TRY_IN_WALK(out, expr)                                                 \
  ({                                                                           \
    auto &&_result = (expr);                                                   \
    if (!_result.succeeded()) {                                                \
      out = std::move(_result);                                                \
      return WalkResult::interrupt();                                          \
    }                                                                          \
  })

#define TRY_GET_OR(out, expr, orElse)                                          \
  ({                                                                           \
    auto &&_result = (expr);                                                   \
    if (!_result.succeeded()) {                                                \
      out = std::move(_result);                                                \
      return (orElse);                                                         \
    }                                                                          \
  })

template <typename Op, typename Res = Op>
static llvm::SmallVector<Res> collect(Operation *root) {
  llvm::SmallVector<Res> results;
  root->walk([&](Op op) { results.push_back(llvm::cast<Res>(op)); });
  return results;
}

struct AffineFun {
  int64_t intersect;
  int64_t slope;

  constexpr int64_t getValue(int64_t x) const { return intersect + slope * x; }

  inline llvm::FailureOr<uint64_t> getIntersectWithY(int64_t y) const {
    if (slope == 0)
      return llvm::failure();
    auto x = (y - intersect) / slope;
    if (x < 0)
      return llvm::failure();
    return static_cast<uint64_t>(x);
  }
};

/// The four top-level buckets an estimated cost can be attributed to.
///   Kernel       — accelerator kernel launch/execution time
///   Cpu          — host CPU time
///   Transfer     — host->accelerator transfers
///   TransferBack — accelerator->host transfers
enum class CostCategory : uint8_t { Kernel, Cpu, Transfer, TransferBack };

constexpr size_t kNumCostCategories = 4;

inline llvm::StringRef costCategoryName(CostCategory c) {
  switch (c) {
  case CostCategory::Kernel:
    return "kernel";
  case CostCategory::Cpu:
    return "cpu";
  case CostCategory::Transfer:
    return "transfer";
  case CostCategory::TransferBack:
    return "transfer_back";
  }
  return "?";
}

/// Breakdown of an estimated program cost (in milliseconds) into the four
/// CostCategory buckets. Within a category, callers may further tag a cost
/// with an arbitrary sub-label (e.g. Cpu "copy" vs "other", Transfer
/// "scatter" vs "broadcast") for finer-grained reporting; same
/// category+label pairs accumulate, distinct labels under the same category
/// are tracked separately but still roll up into that category's total.
/// Combinators (loops, sequences of ops) combine costs label-wise so that
/// e.g. a loop trip count scales every label independently rather than an
/// opaque aggregate.
class SimCost {
public:
  SimCost() = default;

  /// Cost attributed to `category`, optionally tagged with a sub-label for
  /// finer-grained reporting (e.g. forCategory(Cpu, 1.2, "copy")). An empty
  /// label rolls the cost up under the bare category name.
  static SimCost forCategory(CostCategory category, double value,
                             llvm::StringRef label = {}) {
    SimCost c;
    if (value != 0.0)
      c.add(category, label, value);
    return c;
  }
  static SimCost forKernel(double v, llvm::StringRef label = {}) {
    return forCategory(CostCategory::Kernel, v, label);
  }
  static SimCost forCpu(double v, llvm::StringRef label = {}) {
    return forCategory(CostCategory::Cpu, v, label);
  }
  static SimCost forTransfer(double v, llvm::StringRef label = {}) {
    return forCategory(CostCategory::Transfer, v, label);
  }
  static SimCost forTransferBack(double v, llvm::StringRef label = {}) {
    return forCategory(CostCategory::TransferBack, v, label);
  }

  /// Sum of every category/label.
  double total() const {
    double t = 0.0;
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        t += e.second;
    return t;
  }
  /// Sum of every label under `category`.
  double categoryTotal(CostCategory category) const {
    double t = 0.0;
    for (auto &e : buckets[static_cast<size_t>(category)])
      t += e.second;
    return t;
  }
  bool isFinite() const {
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        if (!std::isfinite(e.second))
          return false;
    return true;
  }

  /// Invokes `fn(CostCategory, StringRef label, double value)` for every
  /// entry, for reporting/debugging.
  template <typename Fn>
  void forEachEntry(Fn &&fn) const {
    for (size_t i = 0; i < kNumCostCategories; ++i)
      for (auto &e : buckets[i])
        fn(static_cast<CostCategory>(i), llvm::StringRef(e.first), e.second);
  }

  SimCost &operator+=(const SimCost &o) {
    for (size_t i = 0; i < kNumCostCategories; ++i)
      for (auto &e : o.buckets[i])
        add(static_cast<CostCategory>(i), e.first, e.second);
    return *this;
  }
  SimCost &operator*=(double scale) {
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        e.second *= scale;
    return *this;
  }
  SimCost &operator/=(double scale) { return *this *= (1.0 / scale); }

private:
  // Per category, the (label, value) pairs tagged under it. Most categories
  // only ever see a handful of distinct labels, so a small inline vector
  // avoids hashing/heap allocation in the common case.
  using Bucket = llvm::SmallVector<std::pair<std::string, double>, 2>;
  std::array<Bucket, kNumCostCategories> buckets;

  void add(CostCategory category, llvm::StringRef label, double value) {
    auto &bucket = buckets[static_cast<size_t>(category)];
    for (auto &e : bucket)
      if (e.first == label) {
        e.second += value;
        return;
      }
    bucket.emplace_back(label.str(), value);
  }
};

inline SimCost operator+(SimCost a, const SimCost &b) { return a += b; }
inline SimCost operator*(SimCost a, double scale) { return a *= scale; }
inline SimCost operator*(double scale, SimCost a) { return a *= scale; }
inline SimCost operator/(SimCost a, double scale) { return a /= scale; }

} // namespace mlir::cinm::utils
