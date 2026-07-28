#pragma once

#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include <cinm-mlir/Utils/Scheduling/SchedulingSupport.h>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <variant>
#include <vector>

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
  std::vector<int64_t> values;
};

/// One dimension of the search space.
struct SearchParam {
  std::string name;
  std::variant<IntRange, ValueList> domain;

  SearchParam(const SearchParam &) = delete;
  SearchParam(SearchParam &&) = default;
  SearchParam(StringRef name, IntRange &&range)
      : name(name.str()), domain(std::move(range)) {}
  SearchParam(StringRef name, ValueList &&list)
      : name(name.str()), domain(std::move(list)) {}
  SearchParam &operator=(SearchParam &&o) = default;

  double dlo() const;
  double dhi() const;
  /// Number of distinct values this parameter can take.
  int64_t cardinality() const;
  /// Map a continuous sample in [dlo, dhi] to the nearest valid discrete value.
  int64_t discretize(double v) const;
  /// Map a value to its feature
  double featurize(int64_t n) const;

  /// Return the i-th distinct value of this parameter (0-indexed).
  int64_t valueAt(size_t subIdx) const;
  /// Return the sub-index of value within this parameter's domain (inverse of
  /// valueAt).
  size_t subIndexOf(int64_t value) const;

  /// Retain only values that evenly divide n; converts a range to a ValueList.
  SearchParam &keepDivisorsOf(int64_t n);
};

/// Factory functions — build a SearchParam without adding it to a space yet.
/// Use ConfigSpace::addDim to register the result.
SearchParam makeRange(StringRef name, int64_t lo, int64_t hi, int64_t step = 1);
SearchParam makePow2Range(StringRef name, int64_t loExp, int64_t hiExp);
SearchParam makeValues(StringRef name, std::vector<int64_t> values);

/// A concrete assignment — one int64_t per SearchParam, in ConfigSpace order.
using Configuration = std::vector<int64_t>;

struct ConfigSpace;
struct ConfWrapper;

/// Predicate over a configuration; returns true if the configuration is valid.
using Constraint = std::function<bool(const ConfWrapper)>;

/// Ordered collection of SearchParams that defines the search space.
struct ConfigSpace {
  std::vector<SearchParam> params;
  /// Each constraint paired with a human-readable description of what it
  /// checks (e.g. "wramRow | mramRow"); empty if the constraint was added
  /// without one. Used by debugIsValid() to report violations.
  std::vector<std::pair<std::string, Constraint>> constraints;

  /// A (parent, child) divisibility pair baked into the encoding.
  /// Every flat index produced by at() satisfies child_value % parent_value ==
  /// 0.
  struct DependentGroup {
    size_t parentIdx;
    size_t childIdx;
    /// childValues[k] = sorted valid child values when parent has sub-index k.
    std::vector<std::vector<int64_t>> childValues;
    /// cumCount[k] = sum of childValues[0..k-1].size(); cumCount.back() =
    /// total.
    std::vector<size_t> cumCount;
    size_t totalCount() const { return cumCount.back(); }
  };
  std::vector<DependentGroup> groups;

  ConfigSpace() = default;
  ConfigSpace(const ConfigSpace &) = delete;

  /// Add a fully-constructed SearchParam; returns its index in the space.
  int64_t addDim(SearchParam &&param) {
    int64_t idx = params.size();
    params.push_back(std::move(param));
    encodingValid_ = false;
    return idx;
  }

  std::pair<int64_t, int64_t> addDims(SmallVector<SearchParam> &&dims) {
    int64_t start = params.size();
    for (auto &dim : dims)
      params.push_back(std::move(dim));
    encodingValid_ = false;
    return {start, (int64_t)params.size()};
  }

  /// Register a predicate; configurations for which any constraint returns
  /// false are skipped and never passed to the plugin for evaluation.
  /// `description` is an optional human-readable label for the constraint,
  /// reported by debugIsValid() when it is violated.
  void addConstraint(Constraint &&constraint, std::string description = "") {
    constraints.emplace_back(std::move(description), std::move(constraint));
  }

  /// Register that params[childIdx] must be a multiple of params[parentIdx].
  /// This eliminates invalid (parent, child) pairs from the flat index space —
  /// at() never produces a config violating this constraint.
  /// parentIdx must be < childIdx and both params must already be in params[].
  void addMultiplesConstraint(StringRef parent, StringRef child);

  size_t size() const { return params.size(); }
  const SearchParam &operator[](size_t i) const { return params[i]; }
  SearchParam &operator[](size_t i) { return params[i]; }

  /// Index of param with the given name, or -1.
  int findIndex(llvm::StringRef name) const;
  /// Value of the named param in a configuration, or 0 if not found.
  int64_t get(const Configuration &config, llvm::StringRef name) const;
  /// Return true iff all registered constraints accept this configuration.
  bool isValid(const Configuration &config) const;
  /// Like isValid(), but also prints the configuration and the description
  /// of every violated constraint to `os` (nothing is printed if the
  /// configuration is valid). Returns the same result as isValid().
  bool debugIsValid(const Configuration &config, raw_ostream &os) const;

  /// Total number of configurations reachable by at() (excludes pairs
  /// eliminated by addMultiplesConstraint, includes remaining invalid configs
  /// that are filtered by isValid()).
  size_t totalSize() const;
  /// Fill conf with the configuration at flat index idx.
  /// idx must be in [0, totalSize()). Constraints are NOT checked.
  void at(size_t idx, Configuration &conf) const;
  /// Convert a configuration to its flat index (inverse of at()).
  size_t indexOf(const Configuration &conf) const;
  /// Iterate all configurations in flat-index order, calling fn(conf, flatIdx)
  /// for each. Return false from fn to stop early. Successive calls update only
  /// the suffix of conf that changed (O(1) amortised per step vs O(S) for
  /// at()).
  void forEach(std::function<bool(const Configuration &, size_t)> fn) const;
  /// Append to result all flat indices one discrete step away in any dimension.
  void neighborIndices(size_t idx, llvm::SmallVectorImpl<size_t> &result) const;

  template <class Out> void dump(Out &out, const Configuration &config) const {
    out << " {";
    for (auto [i, dim, value] : llvm::enumerate(params, config)) {
      out << dim.name << "=" << value << (i + 1 < size() ? ", " : "");
    }
    out << "}";
  }

private:
  /// One slot in the flat-index encoding. Child dims are merged into their
  /// parent's slot and do not appear as separate slots.
  struct EncodingSlot {
    size_t dimIdx;   ///< index into params[] (the independent or parent dim)
    size_t groupIdx; ///< index into groups[], or SIZE_MAX for independent dims
    size_t slotSize; ///< number of distinct sub-indices this slot contributes
  };

  mutable bool encodingValid_ = false;
  mutable std::vector<EncodingSlot> slots_;
  /// suffixProd_[i] = product of slotSizes[i..end]; suffixProd_[slots_.size()]
  /// = 1.
  mutable std::vector<size_t> suffixProd_;

  void ensureEncoding() const;
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

inline raw_ostream &operator<<(raw_ostream &os, const ConfWrapper &wrapper) {
  wrapper.space.dump(os, wrapper.conf);
  return os;
}
inline Diagnostic &operator<<(Diagnostic &os, const ConfWrapper &wrapper) {
  wrapper.space.dump(os, wrapper.conf);
  return os;
}

/// Per-trial context owned by the framework and passed to plugin callbacks.
/// Before evaluate() runs the pipeline, computeBlock is a live clone inside
/// module; after the pipeline lowers it away, computeBlock is invalid.
struct TrialInfo {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  cinm::ComputeBlockOp computeBlock;
  Configuration config;
  const ConfigSpace *space = nullptr;

  ConfWrapper conf() const { return ConfWrapper(*space, config); }
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

  /// Evaluate a configuration. Lower total cost is better.
  /// `trial.computeBlock` is a fresh clone inside a minimal trial module
  /// (`module { func @host { clone } }`). The plugin annotates computeBlock,
  /// runs passes on `trial.module`, then returns a cost breakdown. The
  /// framework owns `trial`; the plugin must not retain references after
  /// returning.
  virtual utils::Maybe<utils::SimCost> evaluate(TrialInfo &trial) = 0;

  /// Called once after the best configuration has been found.
  /// `bestTrial.module` is the fully-lowered module from the winning evaluation
  /// (computeBlock is gone by this point). The plugin should splice the lowered
  /// code into the original module and replace `original` with it.
  virtual DiagnosedSilenceableFailure
  commitBestCandidate(cinm::ComputeBlockOp original, TrialInfo bestTrial);

  /// Return a fresh independent copy of this plugin, safe to use from a
  /// different thread. Called by the framework before parallel exhaustive
  /// search; `initializeSpace` has already run on `this` so any indices or
  /// space-derived state should be copied to the new instance.
  virtual std::unique_ptr<InferencePlugin> clone() const = 0;

  /// Optional hook called on each clone (on the main thread) before parallel
  /// evaluation begins. Use it to eagerly build pipelines or other state that
  /// is cheaper to construct single-threaded.
  virtual void warmUp(mlir::MLIRContext *) {}

  /// Whether this plugin is safe to evaluate concurrently from multiple
  /// threads. If false, exhaustive search will run single-threaded.
  virtual bool supportsMultithreading() const { return true; }

  /// Emit debug statistics (e.g. cache hit rate). Called after exhaustive
  /// search completes. Default is a no-op.
  virtual void printStats() const {}
};

// ===----------------------------------------------------------------------===//
// Core framework API
// ===----------------------------------------------------------------------===//

struct InferenceOptions {
  /// Total number of valid evaluations (LHS init + surrogate-guided).
  int maxEvals = 100;
  /// Number of configurations evaluated in the LHS initialisation phase
  /// before the surrogate model takes over.  Must be ≤ maxEvals.
  int nInit = 20;

  int rngSeed = 42;

  /// If true, dumping stats will dump the entire valid space into the pool.csv.
  /// Otherwise pool.csv only contains the visited points, not the whole space.
  bool dumpFullPool = true;

  /// Number of independent BO seeds to run in one process. When > 1, the
  /// ConfigSpace, the valid-config scan, and the validation set are built once
  /// and shared; each seed then runs concurrently on its own thread (single-
  /// threaded per seed), capped by `numWorkers`. Seed values are derived from
  /// `rngSeed`. Each seed dumps to a `seed_<value>/` subdirectory. Has no
  /// effect in exhaustive or single-solution modes.
  int nSeeds = 1;

  // Surrogate model (BANANAS) hyperparameters.
  double kappa = 2.0; ///< UCB exploration weight
  int epochs = 5000;  ///< Training epochs per ensemble member
  int nEnsemble = 7;  ///< Number of MLP ensemble members
  int hidden = 64;    ///< Hidden layer width
  int depth = 2;      ///< Number of hidden layers

  bool sampleOnlyValid = true;
  /// Max number of candidate configs passed to the surrogate for ranking
  /// each round (neighbors of observed points + random draws).
  size_t nCandidates = 500;
  /// How many discrete steps away from observed points to include as
  /// candidates. 1 = immediate neighbors only; 2 = neighbors-of-neighbors, etc.
  unsigned neighborDepth = 1;
  /// When true, only the outermost frontier (exactly `neighborDepth` steps
  /// away) is added. When false, all points within `neighborDepth` steps are
  /// added.
  bool neighborFrontierOnly = false;
  /// If non-empty, dump the full candidate pool to a CSV file in this
  /// directory at the end of inference. Columns: one per search param,
  /// then observed cost (empty if not evaluated), then mu / sigma / acq
  /// from a final ensemble fit (omitted when fewer than 2 observations).
  std::string dumpDir;

  /// When true, evaluate every valid configuration in the search space
  /// instead of running Bayesian optimisation. Useful for collecting ground-
  /// truth cost data and comparing against BO solutions. The pool is dumped
  /// in the same CSV format as the BO run (surrogate columns are omitted
  /// since no model is trained).
  bool exhaustiveSearch = false;

  /// Number of held-out validation points sampled (via LHS) before BO begins.
  /// These are evaluated once for their true cost and never used as BO training
  /// data. At each snapshot the surrogate's mu/sigma are recorded for them.
  /// Zero disables validation entirely.
  int nValidation = 0;
  /// Record a surrogate snapshot on the validation set every N BO iterations
  /// (Phase 2 iterations only). Has no effect when nValidation == 0.
  int validationInterval = 5;

  /// Transform applied to costs before surrogate training.
  /// Supported: "linear", "log2", "log10", "ln", "sqrt", "cbrt".
  std::string objectiveScale = "log10";

  /// Number of worker threads used for exhaustive search.
  /// 0 (default) means use std::thread::hardware_concurrency().
  unsigned numWorkers = 0;

  /// When set, skip search entirely and evaluate only this single
  /// configuration. The values are in the same order as the ConfigSpace params
  /// populated by the plugin's initializeSpace(). Acts as a third mode
  /// alongside exhaustiveSearch and Bayesian optimisation.
  std::optional<Configuration> evalSingleSolution;
};

/// Entry point for Bayesian inference.
DiagnosedSilenceableFailure
inferAcceleratorConfig(cinm::ComputeBlockOp computeOp, InferencePlugin &plugin,
                       const InferenceOptions &opts = {});

} // namespace mlir::cinm
