#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cstdint>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
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
  /// Retain only values that evenly divide n; converts a range to a ValueList.
  SearchParam &keepDivisorsOf(int64_t n);
};

/// Factory functions — build a SearchParam without adding it to a space yet.
/// Use ConfigSpace::addDim to register the result.
SearchParam makeRange(std::string name, int64_t lo, int64_t hi,
                      int64_t step = 1);
SearchParam makePow2Range(std::string name, int64_t loExp, int64_t hiExp);
SearchParam makeValues(std::string name, llvm::SmallVector<int64_t> values);

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

  /// Add a fully-constructed SearchParam; returns its index in the space.
  int64_t addDim(SearchParam &&param) {
    int64_t idx = params.size();
    params.emplace_back(std::move(param));
    return idx;
  }
  /// Convenience wrappers — construct and add in one call.
  int64_t addRange(std::string name, int64_t lo, int64_t hi, int64_t step = 1);
  /// Add a ValueList of consecutive powers of 2: {2^loExp, ..., 2^hiExp}.
  int64_t addPow2Range(std::string name, int64_t loExp, int64_t hiExp);
  int64_t addValues(std::string name, llvm::SmallVector<int64_t> values);
  /// Register a predicate; configurations for which any constraint returns
  /// false are skipped and never passed to the plugin for evaluation.
  void addConstraint(Constraint constraint);

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
  int64_t operator[](int64_t ix) const { return conf[ix]; }
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
  /// Receives a fresh clone of the reference compute block. The clone lives
  /// inside a dedicated trial module (`module { func @host(...) { clone } }`),
  /// so the plugin may run module-scoped passes by calling
  ///   `candidate->getParentOfType<ModuleOp>()`
  /// The trial module is owned and destroyed by the framework after evaluate()
  /// returns; the plugin must not hold references into it.
  virtual utils::Maybe<double> evaluate(cinm::ComputeBlockOp candidate,
                                        const ConfigSpace &space,
                                        const Configuration &config) = 0;

  /// Called once after the best configuration has been found.
  /// The plugin should apply the winning accelerator settings and tile-size
  /// attributes to `original` so that downstream compilation passes pick them
  /// up. No IR from a trial module is available at this point.
  virtual DiagnosedSilenceableFailure
  commitBestCandidate(cinm::ComputeBlockOp original, const ConfigSpace &space,
                      const Configuration &bestConfig) = 0;
};

// ===----------------------------------------------------------------------===//
// Core framework API
// ===----------------------------------------------------------------------===//

/// Build the configuration space by calling plugin.initializeSpace on the
/// reference clone.
ConfigSpace buildConfigSpace(cinm::ComputeBlockOp refClone,
                             InferencePlugin &plugin);

struct InferenceOptions {
  /// Total number of valid evaluations (LHS init + surrogate-guided).
  int maxEvals = 50;
  /// Number of configurations evaluated in the LHS initialisation phase
  /// before the surrogate model takes over.  Must be ≤ maxEvals.
  int nInit = 10;
};

/// Run Bayesian optimization over the config space.
/// Returns the winning Configuration (values for each SearchParam in space).
/// Each trial is evaluated inside an isolated trial module; the framework
/// manages trial module lifetimes.
utils::Maybe<Configuration>
runInference(mlir::ModuleOp refModule, cinm::ComputeBlockOp refClone,
             InferencePlugin &plugin, const ConfigSpace &space,
             const InferenceOptions &opts = {});

/// Full pipeline: clone the parent module into a sandbox → buildConfigSpace →
/// runInference → commitBestCandidate on the original.
/// The original is only modified by commitBestCandidate at the very end.
DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts = {});

} // namespace mlir::cinm
