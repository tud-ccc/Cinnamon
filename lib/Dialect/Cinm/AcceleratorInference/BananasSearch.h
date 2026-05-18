#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallVector.h>

// Suppress mlpack's own info/warning streams — we only want LLVM diagnostics.
#ifndef MLPACK_NO_STD_COUT_PRINT
#define MLPACK_NO_STD_COUT_PRINT
#endif
#include <mlpack.hpp>

namespace mlir::cinm {

struct ConfigSpace;
struct InferenceOptions;
using Configuration = std::vector<int64_t>;

/// Pre-sampled candidate pool with configurations stored in both raw and
/// pre-encoded (normalized) form. encoded is a D×N arma::mat (one sample per
/// column, arma column-major) so callers get zero-copy submatrix views without
/// any float→double conversion at evaluation time.
struct CandidatePool {
  std::vector<Configuration> configs; // N configurations
  arma::mat encoded;                  // D × N, normalized to [0,1]
  llvm::BitVector visited;

  // Incrementally maintained observation matrices.
  // Preallocated to D×N / 1×N; first nObs columns are valid.
  arma::mat Xo; // D × N
  arma::mat yo; // 1 × N
  size_t nObs = 0;

  CandidatePool(std::vector<Configuration> configs, arma::mat encoded)
      : configs(std::move(configs)),
        encoded(std::move(encoded)),
        visited(static_cast<unsigned>(this->encoded.n_cols)),
        Xo(this->encoded.n_rows, this->encoded.n_cols),
        yo(1, this->encoded.n_cols) {}

  size_t size() const { return configs.size(); }
  size_t nDims() const { return encoded.n_rows; }
  bool empty() const { return configs.empty(); }

  const Configuration &operator[](size_t i) const { return configs[i]; }

  void markVisited(size_t idx) { visited.set(idx); }
  bool isVisited(size_t idx) const { return visited.test(idx); }
  size_t numVisited() const { return static_cast<size_t>(visited.count()); }
  /// Index of the first unvisited entry, or -1 if all have been visited.
  size_t firstUnvisited() const { return visited.find_first_unset(); }
  void recordObservation(size_t idx, double cost) {
    Xo.col(nObs) = encoded.col(idx);
    yo(0, nObs) = cost;
    ++nObs;
  }

  /// Rejection-sample up to maxPool distinct valid configurations and
  /// pre-encode them into column-major format in one pass.
  static CandidatePool sample(const ConfigSpace &space, size_t maxPool,
                              std::mt19937 &rng);

  /// Select n row-indices from the pool using Latin Hypercube Sampling.
  llvm::SmallVector<size_t> lhsIndices(size_t n, unsigned seed = 42) const;

  /// Fit a BANANAS MLP ensemble on the observed subset (Xo/yo) and return
  /// the k unvisited pool indices with the lowest UCB acquisition score.
  /// Unvisited entries are derived from the visited bitvector; observations
  /// come from the incrementally maintained Xo/yo matrices (zero-copy view).
  llvm::SmallVector<size_t>
  nextCandidateIndices(const InferenceOptions &opts, size_t k = 1) const;
};

} // namespace mlir::cinm
