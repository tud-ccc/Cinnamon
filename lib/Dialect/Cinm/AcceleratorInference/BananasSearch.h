#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

// Suppress mlpack's own info/warning streams — we only want LLVM diagnostics.
#ifndef MLPACK_NO_STD_COUT_PRINT
#define MLPACK_NO_STD_COUT_PRINT
#endif
#include <mlpack.hpp>

namespace mlir::cinm {

struct ConfigSpace;
struct InferenceOptions;
struct BananasEnsemble; // defined in BananasSearch.cpp
using Configuration = std::vector<int64_t>;

/// Holds a set of configurations, their true costs, and per-iteration surrogate
/// predictions (mu/sigma). This is used to hold a validation set and evaluate
/// surrogate performance during training. Another instance is used to hold the
/// training dataset and evaluate the surrogate fitting.
struct ValidationSet {
  const ConfigSpace *space_;

  std::vector<size_t> indices;   // pool indices of validation configs
  arma::mat encoded;             // D × nVal encoded matrix, built incrementally
  std::vector<double> trueCosts; // true cost for each validation config

  struct Snapshot {
    int iter;
    arma::rowvec mu, sigma;
  };
  std::vector<Snapshot> snapshots; // one entry per recorded iteration

  explicit ValidationSet(const ConfigSpace &space) : space_(&space) {}

  bool empty() const { return indices.empty(); }
  size_t size() const { return indices.size(); }

  /// Record a validation point after evaluating its true cost.
  void record(size_t idx, double cost);
  /// Append a surrogate snapshot for the current BO iteration.
  void recordSnapshot(int iter, arma::rowvec mu, arma::rowvec sigma);
  /// Write one row per (validation config × snapshot) to a CSV file.
  void dumpToCSV(std::filesystem::path path) const;
};

/// Addressable candidate pool backed by the full ConfigSpace Cartesian product.
/// Configurations are not pre-stored; index i maps to the config at
/// ConfigSpace::at(i).
/// validMask_ is a compact BitVector over [0, N) marking which flat indices
/// pass all constraints. All O(nValid) operations iterate validMask_ rather
/// than the full [0, N) range. Xo/yo are pre-allocated to evalBudget, not
/// totalSize().
struct CandidatePool {
  const ConfigSpace *space_;
  size_t N;                           // = space_->totalSize(), cached
  std::unordered_set<size_t> visited; // flat indices of evaluated configs
  std::shared_ptr<llvm::BitVector>
      validMask_; // bit i set iff config at flat index i is valid
  std::shared_ptr<std::vector<size_t>>
      validIndices_;         // flat indices of all valid configs, built once
  size_t nValidVisited_ = 0; // count of valid configs that have been visited

  // Incrementally maintained observation matrices.
  // Preallocated to D×evalBudget / 1×evalBudget; first nObs columns are valid.
  arma::mat Xo;
  arma::mat yo;
  size_t nObs = 0;

  // Per-pool-index observed cost; absent = unvisited or failed evaluation.
  std::unordered_map<size_t, double> costByIdx;
  // BO iteration at which the cost was recorded, keyed by pool index.
  std::unordered_map<size_t, size_t> iterByIdx;
  // Wall-clock evaluation time in milliseconds, keyed by pool index.
  std::unordered_map<size_t, uint64_t> evalTimeByIdx;
  // CPU time (CLOCK_THREAD_CPUTIME_ID) in milliseconds, keyed by pool index.
  std::unordered_map<size_t, uint64_t> cpuTimeByIdx;

  /// Warm-start ensemble: persisted across BO iterations so each call to
  /// nextCandidateIndices fine-tunes from the previous fit rather than
  /// reinitialising from random weights.
  std::unique_ptr<BananasEnsemble> ensemble_;

  bool exhaustive;

  /// Iterate [0, N), pre-mark constraint-violating configs as visited, and
  /// build validMask_. evalBudget sizes Xo/yo (not N).
  static CandidatePool build(const ConfigSpace &space, size_t evalBudget,
                             bool exhaustive = false);

  /// Construct with a precomputed validity mask, skipping the O(N) validity
  /// scan. `validMask` must have been produced by computeValidMask() for the
  /// same ConfigSpace. Used to share the (expensive) scan across seeds while
  /// each seed keeps its own mutable pool state.
  CandidatePool(const ConfigSpace &space, size_t evalBudget,
                std::shared_ptr<llvm::BitVector> validMask,
                std::shared_ptr<std::vector<size_t>> validIndices,
                bool exhaustive = false);
  ~CandidatePool();

  /// Scan the whole Cartesian product once and return a bitmask over [0, N)
  /// with bit i set iff the config at flat index i passes all constraints.
  static void computeValidMask(const ConfigSpace &space,
                               llvm::BitVector &validMask,
                               std::vector<size_t> &validIndices);

  /// Number of valid (constraint-passing) configs in the pool.
  size_t size() const { return validIndices_->size(); }
  size_t nDims() const;
  bool empty() const { return validMask_->none(); }

  /// Return the configuration at flat pool index i (allocated by value).
  Configuration operator[](size_t i) const;

  void markVisited(size_t idx) {
    if (visited.insert(idx).second &&
        validMask_->test(static_cast<unsigned>(idx)))
      ++nValidVisited_;
  }
  bool isVisited(size_t idx) const { return visited.count(idx); }
  bool isValid(size_t idx) const { return validMask_->test(idx); }
  /// Number of valid configs that have been evaluated (or marked visited).
  size_t numVisited() const { return nValidVisited_; }
  /// Flat index of the first valid unvisited config, or N if all visited.
  size_t firstUnvisited() const {
    for (size_t i : *validIndices_)
      if (!visited.count(i))
        return i;
    return N;
  }

  void recordObservation(size_t idx, double cost, size_t iter = 0,
                         std::chrono::milliseconds evalTime = {},
                         uint64_t cpuTimeMs = 0);
  void recordFailedEvaluation(size_t idx, size_t iter = 0);

  /// Select n row-indices from the pool using Latin Hypercube Sampling.
  /// `accept` is invoked for each selected index; it must be thread-safe when
  /// `workers > 1`, as up to `workers` calls run concurrently on a thread pool.
  /// Sampling stops as soon as `n` calls have returned true.
  void sampleInitialSet(size_t n_samples, std::mt19937 &rng,
                        std::function<bool(size_t)> accept,
                        unsigned workers = 1);

  /// Fit a BANANAS MLP ensemble on the observed subset (Xo/yo) and return
  /// the k unvisited pool indices with the lowest UCB acquisition score.
  /// Unvisited entries are derived from the visited bitvector; observations
  /// come from the incrementally maintained Xo/yo matrices (zero-copy view).
  bool nextCandidateIndices(const InferenceOptions &opts, std::mt19937 &rng,
                            std::function<bool(size_t)> accept,
                            ValidationSet &validSet, ValidationSet &trainingSet,
                            int iter);

  /// Dump the full candidate pool to a CSV file at `path`.
  /// Columns: one per search param, then "cost" (empty if not evaluated),
  /// then "mu", "sigma", "acq" from a final ensemble fit (columns omitted
  /// when fewer than 2 observations are available).
  void dumpToCSV(const ConfigSpace &space, const InferenceOptions &opts,
                 std::filesystem::path path) const;

  /// Write a JSON sidecar at `path` summarising the search space:
  /// total_size (Cartesian product), n_valid (constraint-passing configs),
  /// and a per-parameter description (name, type, domain, cardinality).
  void dumpMetadataJSON(const ConfigSpace &space,
                        std::filesystem::path path) const;

private:
  /// Insert idx into result if it is unvisited, not already present, and valid.
  bool tryInsert(std::unordered_set<size_t> &result, size_t idx);
  /// Add up to `target` random valid-unvisited indices to `result`.
  void fillRandom(std::unordered_set<size_t> &result, size_t target,
                  std::mt19937 &rng);
  /// Collect valid, unvisited grid-neighbours of all observed configurations,
  /// up to `depth` discrete steps away (BFS). When `frontierOnly` is true,
  /// only nodes at exactly `depth` steps are added; otherwise all reachable
  /// nodes within `depth` steps are added.
  void fillNeighbors(std::unordered_set<size_t> &result, unsigned depth = 1,
                     bool frontierOnly = false);
};

} // namespace mlir::cinm
