#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
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
  /// Map a continuous sample in [dlo, dhi] to the nearest valid discrete value.
  int64_t discretize(double v) const;
};

/// A concrete assignment — one int64_t per SearchParam, in ConfigSpace order.
using Configuration = llvm::SmallVector<int64_t>;

/// Ordered collection of SearchParams that defines the search space.
struct ConfigSpace {
  llvm::SmallVector<SearchParam> params;

  void addRange(std::string name, int64_t lo, int64_t hi, int64_t step = 1);
  /// Add a ValueList of consecutive powers of 2: {2^loExp, ..., 2^hiExp}.
  void addPow2Range(std::string name, int64_t loExp, int64_t hiExp);
  void addValues(std::string name, llvm::SmallVector<int64_t> values);

  size_t size() const { return params.size(); }
  const SearchParam &operator[](size_t i) const { return params[i]; }
  SearchParam &operator[](size_t i) { return params[i]; }

  /// Index of param with the given name, or -1.
  int findIndex(llvm::StringRef name) const;
  /// Value of the named param in a configuration, or 0 if not found.
  int64_t get(const Configuration &config, llvm::StringRef name) const;
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
  virtual void initializeSpace(cinm::ComputeOp refClone, ConfigSpace &space) = 0;

  /// Evaluate a configuration. Lower cost is better.
  /// Receives a clone of the reference clone (which already carries any
  /// annotations attached during populateOpParams). The plugin may freely
  /// annotate, transform, or lower it — changes do not affect other trials.
  virtual mlir::FailureOr<double> evaluate(cinm::ComputeOp clonedComputeOp,
                                           const ConfigSpace &space,
                                           const Configuration &config) = 0;

  /// Annotate the original compute op with the best configuration found.
  /// Called once after optimization completes.
  virtual mlir::LogicalResult applyBestConfig(cinm::ComputeOp computeOp,
                                              const ConfigSpace &space,
                                              const Configuration &config) = 0;
};

// ===----------------------------------------------------------------------===//
// Core framework API
// ===----------------------------------------------------------------------===//

/// Build the configuration space by calling plugin.initializeSpace on the reference clone.
ConfigSpace buildConfigSpace(cinm::ComputeOp refClone, InferencePlugin &plugin);

struct InferenceOptions {
  int maxEvals = 50;
};

/// Run Bayesian optimization over the config space. Does not modify computeOp.
mlir::FailureOr<Configuration> runInference(cinm::ComputeOp computeOp,
                                            InferencePlugin &plugin,
                                            const ConfigSpace &space,
                                            const InferenceOptions &opts = {});

/// Full pipeline: clone original → buildConfigSpace → runInference →
/// applyBestConfig on the original. The original is never modified until
/// applyBestConfig is called with the winning configuration.
mlir::LogicalResult inferAcceleratorConfig(cinm::ComputeOp computeOp,
                                           InferencePlugin &plugin,
                                           const InferenceOptions &opts = {});

} // namespace mlir::cinm
