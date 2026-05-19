#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <memory>
#include <numeric>
#include <random>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// CandidatePool construction
// ===----------------------------------------------------------------------===//

CandidatePool::CandidatePool(const ConfigSpace &space, size_t evalBudget)
    : space_(&space), N(space.totalSize()), visited(static_cast<unsigned>(N)),
      Xo(space.size(), evalBudget), yo(1, evalBudget),
      costByIdx(arma::rowvec(N).fill(arma::datum::nan)) {}

size_t CandidatePool::nDims() const { return space_->size(); }

Configuration CandidatePool::operator[](size_t i) const {
  Configuration conf;
  space_->at(i, conf);
  return conf;
}

void CandidatePool::recordObservation(size_t idx, double cost) {
  if (nObs >= Xo.n_cols) {
    const size_t newCols = Xo.n_cols + 32;
    Xo.resize(Xo.n_rows, newCols);
    yo.resize(1, newCols);
  }
  Configuration conf;
  space_->at(idx, conf);
  for (size_t d = 0; d < space_->size(); ++d)
    Xo(d, nObs) = (*space_)[d].featurize(conf[d]);
  yo(0, nObs) = cost;
  costByIdx(idx) = cost;
  ++nObs;
}

static arma::mat encodeSubset(const ConfigSpace &space,
                              const std::vector<size_t> &indices);

// ===----------------------------------------------------------------------===//
// Latin Hypercube Sampling
// ===----------------------------------------------------------------------===//

void CandidatePool::sampleInitialSet(size_t n, std::mt19937 &rng,
                                     std::function<bool(size_t)> accept) const {
  const size_t D = nDims();
  if (n == 0 || N == 0)
    return;

  // Collect valid, unvisited candidates for LHS distance computation.
  std::vector<size_t> unvIdx;
  unvIdx.reserve(N);
  Configuration tmpConf;
  for (size_t i = 0; i < N; ++i) {
    if (visited.test(i))
      continue;
    space_->at(i, tmpConf);
    if (space_->isValid(tmpConf))
      unvIdx.push_back(i);
  }

  const size_t M = unvIdx.size();
  if (M == 0)
    return;
  n = std::min(n, M);

  arma::mat enc = encodeSubset(*space_, unvIdx);

  // Per-dimension [0,1] normalisation.
  arma::mat normed(D, M);
  for (size_t d = 0; d < D; ++d) {
    double lo = enc.row(d).min();
    double hi = enc.row(d).max();
    double range = (hi > lo) ? (hi - lo) : 1.0;
    normed.row(d) = (enc.row(d) - lo) / range;
  }

  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<bool> used(M, false);
  size_t accepted = 0;

  // Keep generating LHS batches until n configurations pass accept().
  while (accepted < n) {
    size_t want = n - accepted;

    size_t nUnused = 0;
    for (size_t i = 0; i < M; ++i)
      nUnused += !used[i];
    if (nUnused == 0)
      break;
    want = std::min(want, nUnused);

    // LHS targets for this batch.
    std::vector<std::vector<double>> batchTargets(want, std::vector<double>(D));
    for (size_t d = 0; d < D; ++d) {
      std::vector<size_t> perm(want);
      std::iota(perm.begin(), perm.end(), 0);
      std::shuffle(perm.begin(), perm.end(), rng);
      for (size_t i = 0; i < want; ++i)
        batchTargets[i][d] = (static_cast<double>(perm[i]) + u01(rng)) /
                             static_cast<double>(want);
    }

    // Greedy nearest-neighbour: each target → closest unused candidate.
    for (size_t t = 0; t < want && accepted < n; ++t) {
      double bestDist = std::numeric_limits<double>::max();
      size_t bestPos = M; // position in unvIdx
      for (size_t i = 0; i < M; ++i) {
        if (used[i])
          continue;
        double dist = 0;
        for (size_t d = 0; d < D; ++d) {
          double diff = normed(d, i) - batchTargets[t][d];
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestPos = i;
        }
      }
      if (bestPos == M)
        break;
      used[bestPos] = true;
      if (accept(unvIdx[bestPos]))
        ++accepted;
    }
  }
}

// ===----------------------------------------------------------------------===//
// BANANAS MLP ensemble (mlpack FFN backend)
// ===----------------------------------------------------------------------===//

using MlpNet =
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>;

static std::unique_ptr<MlpNet> makeNet(int hidden, int depth) {
  auto net = std::make_unique<MlpNet>();
  net->Add<mlpack::Linear>(hidden);
  net->Add<mlpack::ReLU>();
  for (int l = 1; l < depth; ++l) {
    net->Add<mlpack::Linear>(hidden);
    net->Add<mlpack::ReLU>();
  }
  net->Add<mlpack::Linear>(1);
  return net;
}

struct BananasEnsemble {
  std::vector<std::unique_ptr<MlpNet>> models;
  double yMean = 0.0;
  double yStd = 1.0;

  BananasEnsemble(int n, int hidden, int depth) {
    models.reserve(n);
    for (int i = 0; i < n; ++i)
      models.push_back(makeNet(hidden, depth));
  }

  void fit(const arma::mat &X, const arma::mat &y, int epochs) {
    // Normalise in log10 space — compresses wide cost ranges into ~3 units.
    // IMPORTANT: anchor the scale to the best (minimum) observed cost rather
    // than the mean.  Using mean/std causes the worst-ever observation to shift
    // yMean upward on each bad eval, which paradoxically lowers the predicted
    // mu for the bad region after de-normalisation (the goalposts move).
    arma::mat yLog = arma::log10(y);
    yMean =
        arma::min(arma::vectorise(yLog)); // anchor = best observed log10 cost
    double s = arma::stddev(arma::vectorise(yLog));
    yStd = (s > 1e-8) ? s : 1.0;
    arma::mat yNorm = (yLog - yMean) / yStd;

    size_t n = X.n_cols;
    for (size_t mi = 0; mi < models.size(); ++mi) {
      arma::arma_rng::set_seed(
          static_cast<arma::arma_rng::seed_type>(mi * 1000003 + 7));
      // Augmented bootstrap: every observation is included once (mandatory),
      // then n additional samples are drawn with replacement for diversity.
      // Pure bootstrap omits any observation ~37% of the time; with small n
      // that means 2-3 members never see the worst-ever point and keep
      // predicting good cost there, holding mu down after the bad eval.
      arma::uvec mandatory = arma::regspace<arma::uvec>(0, n - 1);
      arma::uvec extra = arma::randi<arma::uvec>(
          n, arma::distr_param(0, static_cast<int>(n) - 1));
      arma::uvec idx = arma::join_cols(mandatory, extra);
      arma::mat Xb = X.cols(idx);
      arma::mat yb = yNorm.cols(idx);
      // ensmallen's maxIterations counts gradient updates, not epochs.
      // Compute steps-per-epoch so the training budget scales with the dataset
      // size, not with raw sample count (which caused ~100x overtraining
      // before). Cap batchSize at n to avoid undefined behaviour when n < 32.
      int batchSize = std::min<size_t>(32, n);
      size_t stepsPerEpoch = (2 * n + batchSize - 1) / batchSize;
      size_t maxIter = static_cast<size_t>(epochs) * stepsPerEpoch;
      ens::Adam opt(3e-3, batchSize, 0.9, 0.999, 1e-8, maxIter, 1e-7, true);
      models[mi]->Train(Xb, yb, opt);
    }
  }

  std::pair<arma::rowvec, arma::rowvec> predict(const arma::mat &Xp) const {
    size_t M = Xp.n_cols;
    arma::mat preds(models.size(), M);
    for (size_t mi = 0; mi < models.size(); ++mi) {
      arma::mat out;
      models[mi]->Predict(Xp, out);
      preds.row(mi) = out.row(0);
    }
    // De-standardise back to log10 space. We intentionally do NOT exponentiate
    // here: the acquisition function only needs correct ordering, which log10
    // preserves, and exponentiating a mildly-off log prediction blows up errors
    // by orders of magnitude in the original scale.
    arma::rowvec mu = arma::mean(preds, 0) * yStd + yMean;
    arma::rowvec sigma = arma::stddev(preds, 0, 0) * yStd;
    return {mu, sigma};
  }
};

// ===----------------------------------------------------------------------===//
// Acquisition function
// ===----------------------------------------------------------------------===//

// LCB acquisition with z-scored components.
//
// Raw LCB (mu - kappa*sigma) breaks when mu and sigma live at very different
// scales: if the sigma range (×kappa) exceeds the mu range, the acquisition
// degenerates to pure exploration and mu is ignored entirely.  This happens
// with MLP ensembles because sigma reflects cross-member disagreement, which
// can be as large as the full objective range in unvisited regions.
//
// Fixing by z-scoring each component separately:
//   acq = z(mu) - kappa * z(sigma)
// Now kappa means "one std of sigma exploration bonus is worth kappa std of
// mu exploitation gain" — a scale-independent, calibration-independent
// trade-off that remains valid regardless of ensemble quality.
static arma::rowvec computeAcq(const arma::rowvec &mu,
                               const arma::rowvec &sigma, double kappa) {
  auto zs = [](const arma::rowvec &v) -> arma::rowvec {
    double m = arma::mean(arma::vectorise(v));
    double s = arma::stddev(arma::vectorise(v));
    return (v - m) / ((s > 1e-8) ? s : 1.0);
  };
  return zs(mu) - kappa * zs(sigma);
}

// ===----------------------------------------------------------------------===//
// Next-candidate selection
// ===----------------------------------------------------------------------===//

// Build a temporary D×M encoded matrix for a set of candidate pool indices.
static arma::mat encodeSubset(const ConfigSpace &space,
                              const std::vector<size_t> &indices) {
  const size_t D = space.size();
  const size_t M = indices.size();
  arma::mat enc(D, M);
  Configuration conf;
  for (size_t j = 0; j < M; ++j) {
    space.at(indices[j], conf);
    for (size_t d = 0; d < D; ++d)
      enc(d, j) = space[d].featurize(conf[d]);
  }
  return enc;
}

bool CandidatePool::tryInsert(std::unordered_set<size_t> &result, size_t idx,
                              Configuration &conf) {
  if (visited.test(idx) || result.count(idx))
    return false;
  space_->at(idx, conf);
  if (!space_->isValid(conf)) {
    visited.set(idx);
    return false;
  }
  result.insert(idx);
  return true;
}

void CandidatePool::fillRandom(std::unordered_set<size_t> &result,
                               size_t target, std::mt19937 &rng) {
  if (result.size() >= target || N == 0)
    return;
  std::uniform_int_distribution<size_t> dist(0, N - 1);
  Configuration conf;
  for (size_t tries = 0; tries < target * 10 && result.size() < target;
       ++tries) {
    tryInsert(result, dist(rng), conf);
  }
}

void CandidatePool::fillNeighbors(std::unordered_set<size_t> &result,
                                   unsigned depth, bool frontierOnly) {
  // BFS outward from every observed point up to `depth` steps.
  std::unordered_set<size_t> frontier;
  Configuration conf;
  for (size_t i = 0; i < N; ++i) {
    if (!std::isnan(costByIdx(i)))
      frontier.insert(i);
  }

  llvm::SmallVector<size_t> nbrs;
  for (unsigned d = 0; d < depth && !frontier.empty(); ++d) {
    const bool isLastStep = (d + 1 == depth);
    std::unordered_set<size_t> nextFrontier;
    for (size_t src : frontier) {
      nbrs.clear();
      space_->neighborIndices(src, nbrs);
      for (size_t nb : nbrs) {
        // When frontierOnly, only insert at the last BFS level.
        if (!frontierOnly || isLastStep) {
          tryInsert(result, nb, conf);
        }
        // Always track the frontier for BFS expansion regardless.
        if (!visited.test(nb))
          nextFrontier.insert(nb);
      }
    }
    frontier = std::move(nextFrontier);
  }
}

bool CandidatePool::nextCandidateIndices(const InferenceOptions &opts,
                                         std::mt19937 &rng,
                                         std::function<bool(size_t)> accept) {
  const size_t D = nDims();

  // --- Build candidate set ---
  // Start with the grid-neighbours of every already-observed configuration.
  // Neighbours differ in exactly one dimension by one discrete step, so they
  // are the most likely region to contain a better point.
  std::unordered_set<size_t> candSet;
  fillNeighbors(candSet, opts.neighborDepth);
  // Add random candidates.
  fillRandom(candSet, opts.nCandidates, rng);

  if (candSet.empty())
    return false;
  assert(llvm::all_of(candSet, [&](auto idx) {
    Configuration conf;
    space_->at(idx, conf);
    return space_->isValid(conf);
  }));

  std::vector<size_t> candIdx(candSet.begin(), candSet.end());
  arma::mat candEncoded = encodeSubset(*space_, candIdx);

  // --- Fit surrogate and rank candidates ---
  arma::mat Xo_obs(const_cast<double *>(Xo.memptr()), D, nObs,
                   /*copy=*/false, /*strict=*/true);
  arma::mat yo_obs(const_cast<double *>(yo.memptr()), 1, nObs,
                   /*copy=*/false, /*strict=*/true);

  BananasEnsemble ensemble(opts.nEnsemble, opts.hidden, opts.depth);
  ensemble.fit(Xo_obs, yo_obs, opts.epochs);
  auto [mu, sigma] = ensemble.predict(candEncoded);

  arma::rowvec scores = computeAcq(mu, sigma, opts.kappa);
  arma::uvec order = arma::sort_index(scores, "ascend");

  for (size_t i = 0; i < order.n_elem; ++i) {
    if (accept(candIdx[order(i)]))
      return true;
  }
  return false;
}

// ===----------------------------------------------------------------------===//
// CSV dump
// ===----------------------------------------------------------------------===//

void CandidatePool::dumpToCSV(const ConfigSpace &space,
                              const InferenceOptions &opts,
                              llvm::StringRef path) const {
  std::filesystem::create_directories(
      std::filesystem::path(path.str()).parent_path());
  std::ofstream out(path.str());
  if (!out)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Finished inference\n"
                          << "- " << nObs << " / " << visited.count()
                          << " successful trials\n");

  // Collect all valid (constraint-passing) pool indices for surrogate
  // prediction. Invalid configs get no surrogate columns in the output.
  std::vector<size_t> validIdx;
  {
    Configuration conf;
    for (size_t i = 0; i < N; ++i) {
      space_->at(i, conf);
      if (space_->isValid(conf))
        validIdx.push_back(i);
    }
  }

  // Refit the ensemble on all observations to get per-candidate statistics.
  // Skipped when we have too few points to train on.
  const bool hasModel = nObs >= 2 && !validIdx.empty();
  // Per-valid-index predictions; indexed by position in validIdx.
  arma::rowvec mu_v, sigma_v, acq_v;
  if (hasModel) {
    arma::mat Xo_obs(const_cast<double *>(Xo.memptr()), nDims(), nObs,
                     /*copy=*/false, /*strict=*/true);
    arma::mat yo_obs(const_cast<double *>(yo.memptr()), 1, nObs,
                     /*copy=*/false, /*strict=*/true);
    BananasEnsemble ensemble(opts.nEnsemble, opts.hidden, opts.depth);
    ensemble.fit(Xo_obs, yo_obs, opts.epochs);
    arma::mat validEncoded = encodeSubset(*space_, validIdx);
    auto [m, s] = ensemble.predict(validEncoded);
    mu_v = m;
    sigma_v = s;
    acq_v = computeAcq(mu_v, sigma_v, opts.kappa);
  }

  // Build reverse map: pool index → position in validIdx.
  std::unordered_map<size_t, size_t> validPos;
  for (size_t j = 0; j < validIdx.size(); ++j)
    validPos[validIdx[j]] = j;

  // Header
  for (const auto &p : space.params)
    out << p.name << ",";
  out << "visited,cost";
  if (hasModel)
    out << ",mu,sigma,acq";
  out << "\n";

  // One row per pool member
  Configuration conf;
  for (size_t i = 0; i < N; ++i) {
    space_->at(i, conf);
    for (int64_t v : conf)
      out << v << ",";
    out << (visited.test(i) ? 1 : 0) << ",";
    double c = costByIdx(i);
    if (!std::isnan(c))
      out << c;
    if (hasModel) {
      auto it = validPos.find(i);
      if (it != validPos.end()) {
        size_t j = it->second;
        out << "," << mu_v(j) << "," << sigma_v(j) << "," << acq_v(j);
      } else {
        out << ",,,"; // invalid config — no surrogate prediction
      }
    }
    out << "\n";
  }
}

} // namespace mlir::cinm
