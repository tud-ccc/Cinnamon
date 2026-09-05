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
///
/// An entry may additionally be marked *excluded* (markExcluded): work the
/// program performs, and that the breakdown keeps reporting under its own
/// category and label, but that the totals do not charge -- a transfer of
/// data that is the same on every inference is paid once by a serving
/// deployment, not per inference. Excluding is a statement about how often
/// the cost is paid, not about whether it happened, so the entry stays
/// visible to anything scoring the model against a run.
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
      c.add(category, label, value, /*excluded=*/false);
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

  /// Mark every entry as excluded from the totals, keeping it under the
  /// category and label it was recorded with. See the class comment.
  SimCost &markExcluded() {
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        e.excluded = true;
    return *this;
  }

  /// Sum of every category/label, minus the excluded entries.
  ///
  /// A non-finite entry makes the whole total non-finite whether or not it is
  /// excluded, and the offending value is what comes back. Excluding a cost
  /// says how often the program pays it, not whether the number is known: an
  /// entry a model could not compute leaves the rest of the walk unreliable
  /// too, and silently reporting the finite remainder would score a
  /// configuration whose cost is unknown as one that is cheap.
  double total() const {
    double t = 0.0;
    for (auto &bucket : buckets)
      for (auto &e : bucket) {
        if (!std::isfinite(e.value))
          return e.value;
        if (!e.excluded)
          t += e.value;
      }
    return t;
  }
  /// Sum of the excluded entries -- what the program pays that total() does
  /// not charge it.
  double excludedTotal() const {
    double t = 0.0;
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        if (e.excluded)
          t += e.value;
    return t;
  }
  /// Sum of every label under `category`, minus the excluded entries.
  double categoryTotal(CostCategory category) const {
    double t = 0.0;
    for (auto &e : buckets[static_cast<size_t>(category)])
      if (!e.excluded)
        t += e.value;
    return t;
  }
  bool isFinite() const {
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        if (!std::isfinite(e.value))
          return false;
    return true;
  }

  /// Invokes `fn(CostCategory, StringRef label, double value, bool excluded)`
  /// for every entry, excluded ones included, for reporting/debugging.
  template <typename Fn> void forEachEntry(Fn &&fn) const {
    for (size_t i = 0; i < kNumCostCategories; ++i)
      for (auto &e : buckets[i])
        fn(static_cast<CostCategory>(i), llvm::StringRef(e.label), e.value,
           e.excluded);
  }

  SimCost &operator+=(const SimCost &o) {
    for (size_t i = 0; i < kNumCostCategories; ++i)
      for (auto &e : o.buckets[i])
        add(static_cast<CostCategory>(i), e.label, e.value, e.excluded);
    return *this;
  }
  SimCost &operator*=(double scale) {
    for (auto &bucket : buckets)
      for (auto &e : bucket)
        e.value *= scale;
    return *this;
  }
  SimCost &operator/=(double scale) { return *this *= (1.0 / scale); }

private:
  struct Entry {
    std::string label;
    double value;
    bool excluded;
  };
  // Per category, the entries tagged under it. Most categories only ever see
  // a handful of distinct labels, so a small inline vector avoids
  // hashing/heap allocation in the common case.
  using Bucket = llvm::SmallVector<Entry, 2>;
  std::array<Bucket, kNumCostCategories> buckets;

  // Excluded and charged cost under one label stay separate entries: they
  // answer different questions and each side has readers.
  void add(CostCategory category, llvm::StringRef label, double value,
           bool excluded) {
    auto &bucket = buckets[static_cast<size_t>(category)];
    for (auto &e : bucket)
      if (e.label == label && e.excluded == excluded) {
        e.value += value;
        return;
      }
    bucket.push_back({label.str(), value, excluded});
  }
};

inline SimCost operator+(SimCost a, const SimCost &b) { return a += b; }
inline SimCost operator*(SimCost a, double scale) { return a *= scale; }
inline SimCost operator*(double scale, SimCost a) { return a *= scale; }
inline SimCost operator/(SimCost a, double scale) { return a /= scale; }

} // namespace mlir::cinm::utils
