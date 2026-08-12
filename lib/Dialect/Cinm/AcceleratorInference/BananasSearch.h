#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h>

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

/// Addressable candidate pool over a ConfigSpace's flat index range [0, N).
/// Configurations are not pre-stored; index i maps to the config at
/// ConfigSpace::at(i). Every index is a candidate: a space holds only the
/// configurations that satisfy its constraints, so the pool has nothing to
/// screen. Xo/yo are pre-allocated to evalBudget, not totalSize().
struct CandidatePool {
  using MaskType = std::unordered_set<size_t>;

  const ConfigSpace *space_;
  size_t N;         // = space_->totalSize(), cached
  MaskType visited; // flat indices of evaluated configs

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

  /// `evalBudget` sizes Xo/yo (not N).
  CandidatePool(const ConfigSpace &space, size_t evalBudget,
                bool exhaustive = false);
  ~CandidatePool();

  /// Number of configs in the pool.
  size_t size() const { return N; }
  /// Width of the surrogate's input vector (not the parameter count).
  size_t numFeatures() const;
  bool empty() const { return N == 0; }

  /// Return the configuration at flat pool index i (allocated by value).
  Configuration operator[](size_t i) const;

  void markVisited(size_t idx) { visited.insert(idx); }
  bool isVisited(size_t idx) const { return visited.count(idx); }
  /// Number of configs that have been evaluated (or marked visited).
  size_t numVisited() const { return visited.size(); }
  /// Flat index of the first unvisited config, or N if all visited.
  size_t firstUnvisited() const {
    for (size_t i = 0; i < N; ++i)
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

  /// Write a JSON sidecar at `path` summarising the search space: the
  /// Cartesian product of the declared domains, the number of configurations
  /// the space holds, and a per-parameter description (name, type, domain,
  /// cardinality). Delegates to dumpSpaceJSON with this pool's size.
  void dumpMetadataJSON(const ConfigSpace &space,
                        std::filesystem::path path) const;

private:
  /// Insert idx into result if it is unvisited and not already present.
  bool tryInsert(std::unordered_set<size_t> &result, size_t idx);
  /// Add up to `target` random unvisited indices to `result` (rejection
  /// sampling over [0, N), deduplicated via `result`). Used for BO candidate
  /// generation (nextCandidateIndices).
  void fillRandom(std::unordered_set<size_t> &result, size_t target,
                  std::mt19937 &rng);
  /// Collect unvisited grid-neighbours of all observed configurations,
  /// up to `depth` discrete steps away (BFS). When `frontierOnly` is true,
  /// only nodes at exactly `depth` steps are added; otherwise all reachable
  /// nodes within `depth` steps are added.
  void fillNeighbors(std::unordered_set<size_t> &result, unsigned depth = 1,
                     bool frontierOnly = false);
};

/// Write the space.json sidecar at `path` for `space`, whose enumerated
/// feasible set has `feasibleSize` configurations. Pool-free so it can serve
/// dump-space-only mode, where no search (and hence no CandidatePool) exists;
/// CandidatePool::dumpMetadataJSON delegates here. Besides the sizes and
/// per-parameter domains, the dump carries everything a human transcribing an
/// external schedule needs: per-parameter `doc` strings, the per-dimension
/// names eval-solution expects, and -- for permutation parameters -- the item
/// labels, the encoding rule, and the full orderings table with one
/// copy-pasteable eval-solution assignment per ordering.
void dumpSpaceJSON(const ConfigSpace &space, size_t feasibleSize,
                   std::filesystem::path path);

} // namespace mlir::cinm
