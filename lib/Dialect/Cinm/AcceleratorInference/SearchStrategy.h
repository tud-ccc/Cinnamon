#pragma once

#include <armadillo>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>

namespace mlir::cinm {

struct CandidatePool;
struct ValidationSet;

/// Search policy for the post-init phase: each round picks unvisited
/// configurations from the pool and evaluates them through `accept`. The
/// driver (runSeedBO) owns everything strategy-independent -- the init
/// sample, budget accounting, observation bookkeeping, timing -- so that
/// strategies compared under one harness differ only in this policy.
class SearchStrategy {
public:
  virtual ~SearchStrategy();

  /// Run one round: choose up to `batchSize` unvisited pool indices and
  /// evaluate them through `accept` (thread-safe when `workers > 1`; up to
  /// `workers` calls run concurrently). Returns how many were accepted, 0
  /// meaning the round made no progress and the search should stop.
  ///
  /// `round` counts rounds; `nObsAtRound` is the observation count the round
  /// started with, so per-evaluation series stay indexed by evaluations spent
  /// even when a round spends more than one.
  virtual size_t step(std::mt19937 &rng,
                      const std::function<bool(size_t)> &accept, int round,
                      size_t nObsAtRound, size_t batchSize,
                      unsigned workers) = 0;

  /// Whether the strategy has a fitted model whose predictions are worth
  /// dumping (pool.csv's mu/sigma/acq columns).
  virtual bool hasModel() const { return false; }

  /// Predict mean/spread for the encoded configurations in `X` (one column
  /// per config). Only meaningful when hasModel(); the default predicts
  /// nothing and returns false.
  virtual bool predict(const arma::mat &X, arma::rowvec &mu,
                       arma::rowvec &sigma) const {
    return false;
  }

  /// Dump whatever per-round diagnostics the strategy collected into `dir`.
  /// Called once at the end of a seed, only when dumping is enabled.
  virtual void dumpDiagnostics(const std::filesystem::path &dir) const {}
};

/// Build the strategy selected by the pool's InferenceOptions. The validation
/// sets are only used by surrogate-based strategies (prediction snapshots);
/// they outlive the strategy in the driver.
std::unique_ptr<SearchStrategy> makeSearchStrategy(CandidatePool &pool,
                                                   ValidationSet &validSet,
                                                   ValidationSet &trainingSet);

} // namespace mlir::cinm
