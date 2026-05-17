#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cstdint>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <variant>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Configuration space types
// ===----------------------------------------------------------------------===//

/// Contiguous integer range [lo, hi] sampled at multiples of step.
struct IntRange {
  int64_t lo, hi;
  int64_t step = 1;
};

/// Explicit discrete value set.
struct ValueList {
  llvm::SmallVector<int64_t> values;
};

/// One dimension of the search space.
struct SearchParam {
  std::string name;
  std::variant<IntRange, ValueList> domain;

  double dlo() const;
  double dhi() const;
  /// Number of distinct values this parameter can take.
  int64_t cardinality() const;
  /// Map a continuous sample in [dlo, dhi] to the nearest valid discrete value.
  int64_t discretize(double v) const;
};

/// A concrete assignment — one int64_t per SearchParam, in ConfigSpace order.
using Configuration = llvm::SmallVector<int64_t>;

struct ConfigSpace;
struct ConfWrapper;

/// Predicate over a configuration; returns true if the configuration is valid.
using Constraint = std::function<bool(const ConfWrapper &)>;

/// Ordered collection of SearchParams that defines the search space.
struct ConfigSpace {
  llvm::SmallVector<SearchParam> params;
  llvm::SmallVector<Constraint> constraints;

  void addRange(std::string name, int64_t lo, int64_t hi, int64_t step = 1);
  /// Add a ValueList of consecutive powers of 2: {2^loExp, ..., 2^hiExp}.
  void addPow2Range(std::string name, int64_t loExp, int64_t hiExp);
  void addValues(std::string name, llvm::SmallVector<int64_t> values);
  /// Register a predicate; configurations for which any constraint returns
  /// false are skipped and never passed to the plugin for evaluation.
  void addConstraint(Constraint constraint);
  void addConstraint(std::optional<Constraint> constraint) {
    if (auto aConstraint = constraint) {
      addConstraint(*aConstraint);
    }
  }

  size_t size() const { return params.size(); }
  const SearchParam &operator[](size_t i) const { return params[i]; }
  SearchParam &operator[](size_t i) { return params[i]; }

  /// Index of param with the given name, or -1.
  int findIndex(llvm::StringRef name) const;
  /// Value of the named param in a configuration, or 0 if not found.
  int64_t get(const Configuration &config, llvm::StringRef name) const;
  /// Return true iff all registered constraints accept this configuration.
  bool isValid(const Configuration &config) const;
};

/// Wrap a space and config for nicer interface.
struct ConfWrapper {
  const ConfigSpace &space;
  const Configuration &conf;
  ConfWrapper(const ConfigSpace &space, const Configuration &conf)
      : space(space), conf(conf) {}

  /// Get the value of a variable
  int64_t operator[](StringRef name) const { return space.get(conf, name); }
};

// ===----------------------------------------------------------------------===//
// Plugin interface
// ===----------------------------------------------------------------------===//

/// Abstract plugin, one implementation per target.
/// Responsible for populating the config space and evaluating configurations.
/// The core framework calls these methods; target-specific logic lives here.
struct InferencePlugin {
  virtual ~InferencePlugin() = default;

  /// Populate the configuration space from the reference clone.
  /// The plugin decides what to add and how to explore the IR — it may walk
  /// the compute body, inspect op shapes, attach attributes to nodes, etc.
  /// Annotations left on the clone are inherited by every per-evaluation clone.
  virtual void initializeSpace(cinm::ComputeBlockOp refClone,
                               ConfigSpace &space) = 0;

  /// Evaluate a configuration. Lower cost is better.
  /// Receives a fresh clone inserted right after the reference clone in the
  /// sandbox module. The plugin may freely annotate, transform, or lower it —
  /// changes do not affect the reference clone or other trials.
  virtual utils::Maybe<double> evaluate(cinm::ComputeBlockOp candidate,
                                        const ConfigSpace &space,
                                        const Configuration &config) = 0;

  /// Discard a losing candidate. Override to also clean up any side resources
  /// (e.g. kernel submodules) created during evaluate().
  /// Default implementation simply erases the op.
  virtual void disposeCandidate(cinm::ComputeBlockOp candidate) {
    candidate->erase();
  }

  /// Called once with the winning candidate (inside the sandbox module).
  /// The plugin must transfer the lowered IR and any side resources
  /// (e.g. kernel submodules) from the candidate into the original module,
  /// then erase the candidate. The sandbox is destroyed after this returns.
  virtual DiagnosedSilenceableFailure
  commitBestCandidate(cinm::ComputeBlockOp original,
                      cinm::ComputeBlockOp bestCandidate) = 0;
};

// ===----------------------------------------------------------------------===//
// Core framework API
// ===----------------------------------------------------------------------===//

/// Build the configuration space by calling plugin.initializeSpace on the
/// reference clone.
ConfigSpace buildConfigSpace(cinm::ComputeBlockOp refClone,
                             InferencePlugin &plugin);

struct InferenceOptions {
  int maxEvals = 50;
};

/// Run Bayesian optimization over the config space.
/// Returns the winning candidate op (still inside the sandbox module).
/// The caller is responsible for committing or disposing it.
utils::Maybe<cinm::ComputeBlockOp>
runInference(cinm::ComputeBlockOp refClone, InferencePlugin &plugin,
             const ConfigSpace &space, const InferenceOptions &opts = {});

/// Full pipeline: clone the parent module into a sandbox → buildConfigSpace →
/// runInference → commitBestCandidate on the original.
/// The original is only modified by commitBestCandidate at the very end.
DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts = {});

} // namespace mlir::cinm
