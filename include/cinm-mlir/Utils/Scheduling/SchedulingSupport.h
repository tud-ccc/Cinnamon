#pragma once

#include <cstdint>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
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
    if (std::holds_alternative<mlir::DiagnosedSilenceableFailure>(_result)) { \
      return std::move(std::get<1>(_result));                                  \
    }                                                                          \
    std::get<0>(_result);                                                      \
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
      return (orElse);                                          \
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

} // namespace mlir::cinm::utils
