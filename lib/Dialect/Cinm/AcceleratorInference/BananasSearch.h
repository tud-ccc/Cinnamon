#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mlir/IR/Diagnostics.h>
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

/// One candidate the acquisition picked, and where it sat in the ranking.
struct SelectionRecord {
  size_t idx; // pool index
  double mu, sigma;
  // Position in the candidate ranking under each criterion: the acquisition
  // itself, the mean alone (pure exploitation), and the spread alone (pure
  // exploration, so 0 is the most uncertain candidate). Their spread is what
  // separates a point the surrogate believes is good from one it merely knows
  // nothing about.
  size_t rankAcq, rankMu, rankSigma;
  bool fromNeighbor; // came from an observed point's neighbourhood
  bool accepted;     // evaluation succeeded
};

/// Per-round record of what the surrogate-guided phase did: where the round's
/// time went, and which candidates it drew.
struct RoundRecord {
  int round;        // Phase-2 round, 0-based
  size_t nObs;      // observations available to the fit, before this round
  size_t nCand;     // candidates ranked this round
  size_t nNeighbor; // of which came from the observed points' neighbourhood
  // Where the round's wall clock went. One fit serves the whole batch, so its
  // ratio to the evaluation is what raising the batch size trades against.
  // `acceptMs` is the batch's elapsed time, not the sum of its evaluations:
  // with workers to spare they overlap.
  double fitMs, predictMs, acceptMs;
  std::vector<SelectionRecord> selections;
};

/// Measurement of one batch of candidates: either the batch a round actually
/// evaluated, or the top-q of the ranking at a size the round did not use.
/// Both are recorded at every q, so what an alternative batch size would have
/// covered is measurable from a run that never used it.
struct BatchRecord {
  int round;
  size_t q;
  /// Whether this is the batch the round evaluated (true) or the ranking's
  /// top-q measured for comparison (false).
  bool selected;
  /// q² over the summed RBF similarity of every ordered pair: how many
  /// mutually distinguishable points the batch really holds. q when the members
  /// sit further apart than the kernel width, 1 when they collapse onto one
  /// location. The kernel width is the candidate set's median pairwise
  /// distance, so the measure is comparable across rounds and problems.
  double qEff;
  /// The same measure on a uniformly drawn q-subset of the candidates. In a
  /// discrete space of this width pairwise distances concentrate, so qEff is
  /// well under q even for an unclustered batch; only the ratio to this
  /// reference says whether the acquisition did the concentrating.
  double qEffRef;
  double meanPdist;  // mean pairwise distance, normalised feature space
  double dispersion; // meanPdist / mean pairwise distance of a random q-subset
  double overlapMu;  // |top-q by acq ∩ top-q by mu| / q
  double overlapSigma; // |top-q by acq ∩ top-q by sigma| / q
  /// Per-dimension normalised entropy of the values the batch spans. Reveals
  /// which dimensions the surrogate has committed to and which it still
  /// spreads over, which the scalar distances cannot separate.
  std::vector<double> dimEntropy;
};

/// Diagnostics accumulated across the surrogate-guided phase of one seed.
struct SearchDiagnostics {
  std::vector<RoundRecord> rounds;
  std::vector<BatchRecord> batches;

  bool empty() const { return rounds.empty(); }
  void dumpRoundsCSV(std::filesystem::path path) const;
  void dumpBatchesCSV(const ConfigSpace &space,
                      std::filesystem::path path) const;
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

  /// Filled by nextCandidateIndices when opts.dumpDir is set.
  SearchDiagnostics diag;

  const InferenceOptions &opts;

  /// `evalBudget` sizes Xo/yo (not N).
  CandidatePool(const ConfigSpace &space, size_t evalBudget,
                const InferenceOptions &opts);

  ~CandidatePool();
  /// Number of configs in the pool.
  size_t size() const { return N; }
  /// Width of the surrogate's input vector (not the parameter count).
  size_t numFeatures() const { return space_->numFeatures(); }
  bool empty() const { return N == 0; }

  /// Return the configuration at flat pool index i (allocated by value).
  Configuration operator[](size_t i) const {
    Configuration conf;
    space_->at(i, conf);
    return conf;
  }

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

  /// Fit a BANANAS MLP ensemble on the observed subset (Xo/yo), draw
  /// `batchSize` unvisited candidates from the acquisition, and evaluate them
  /// through `accept`. Unvisited entries are derived from the visited
  /// bitvector; observations come from the incrementally maintained Xo/yo
  /// matrices (zero-copy view). Returns how many were accepted, 0 meaning the
  /// round made no progress and the search should stop.
  ///
  /// `accept` runs on a thread pool when `workers > 1`, so it must be
  /// thread-safe exactly as sampleInitialSet requires. A batch that accepts
  /// nothing falls back to walking the rest of the ranking for a single point,
  /// which is what keeps a rejected batch from ending the search.
  ///
  /// `round` counts Phase-2 rounds and paces the validation snapshots; `nObs`
  /// is the observation count the snapshots are stamped with, so the validation
  /// series stays indexed by evaluations spent even when a round spends more
  /// than one.
  size_t nextCandidateIndices(std::mt19937 &rng,
                              std::function<bool(size_t)> accept,
                              ValidationSet &validSet,
                              ValidationSet &trainingSet, int round,
                              size_t nObs, size_t batchSize = 1,
                              unsigned workers = 1);

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
  /// Append this round's RoundRecord (selection fields left for the caller to
  /// fill once a candidate is accepted) and one BatchRecord per configured
  /// batch size.
  void recordRoundDiagnostics(
      const InferenceOptions &opts, std::mt19937 &rng, int round, size_t nObs,
      llvm::ArrayRef<size_t> candIdx, const arma::mat &candEncoded,
      size_t nNeighborCands, const arma::uvec &order, const arma::uvec &orderMu,
      const arma::uvec &orderSigma, llvm::ArrayRef<arma::uword> selected);
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
