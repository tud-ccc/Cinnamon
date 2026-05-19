#pragma once

#include <concepts>
#include <cstddef>
#include <random>
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
using Configuration = std::vector<int64_t>;

/// Addressable candidate pool backed by the full ConfigSpace Cartesian product.
/// Configurations are not pre-stored; index i maps to the config at
/// ConfigSpace::at(i).  Invalid configs (constraint failures) are pre-marked
/// visited during construction so they are never selected.
/// encoded is a D×N arma::mat (one column per config) used for surrogate
/// predictions. Xo/yo are pre-allocated to evalBudget, not totalSize().
struct CandidatePool {
  const ConfigSpace *space_;
  size_t N; // = space_->totalSize(), cached
  llvm::BitVector visited;

  // Incrementally maintained observation matrices.
  // Preallocated to D×evalBudget / 1×evalBudget; first nObs columns are valid.
  arma::mat Xo;
  arma::mat yo;
  size_t nObs = 0;

  // Per-pool-index observed cost; NaN for unvisited or failed evaluations.
  arma::rowvec costByIdx;

  /// Encode the full Cartesian product and pre-mark constraint-violating
  /// configs as visited. evalBudget sizes Xo/yo (not N).
  CandidatePool(const ConfigSpace &space, size_t evalBudget);

  size_t size() const { return N; }
  size_t nDims() const;
  bool empty() const { return N == 0; }

  /// Return the configuration at index i (allocated by value).
  Configuration operator[](size_t i) const;

  void markVisited(size_t idx) { visited.set(idx); }
  bool isVisited(size_t idx) const { return visited.test(idx); }
  size_t numVisited() const { return static_cast<size_t>(visited.count()); }
  /// Index of the first unvisited entry, or size() if all have been visited.
  size_t firstUnvisited() const { return visited.find_first_unset(); }

  void recordObservation(size_t idx, double cost);

  /// Select n row-indices from the pool using Latin Hypercube Sampling.
  void sampleInitialSet(size_t n_samples, std::mt19937 &rng,
                        std::function<bool(size_t)> accept) const;

  /// Fit a BANANAS MLP ensemble on the observed subset (Xo/yo) and return
  /// the k unvisited pool indices with the lowest UCB acquisition score.
  /// Unvisited entries are derived from the visited bitvector; observations
  /// come from the incrementally maintained Xo/yo matrices (zero-copy view).
  bool nextCandidateIndices(const InferenceOptions &opts, std::mt19937 &rng,
                            std::function<bool(size_t)> accept);

  /// Dump the full candidate pool to a CSV file at `path`.
  /// Columns: one per search param, then "cost" (empty if not evaluated),
  /// then "mu", "sigma", "acq" from a final ensemble fit (columns omitted
  /// when fewer than 2 observations are available).
  void dumpToCSV(const ConfigSpace &space, const InferenceOptions &opts,
                 llvm::StringRef path) const;

private:
  /// Insert idx into result if it is unvisited, not already present, and valid.
  bool tryInsert(std::unordered_set<size_t> &result, size_t idx,
                 Configuration &conf);
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
