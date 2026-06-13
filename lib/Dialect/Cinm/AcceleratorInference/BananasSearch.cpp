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
#include <utility>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// CandidatePool construction
// ===----------------------------------------------------------------------===//

CandidatePool::CandidatePool(const ConfigSpace &space, size_t evalBudget)
    : space_(&space), N(space.totalSize()), visited(static_cast<unsigned>(N)),
      Xo(space.size(), evalBudget), yo(1, evalBudget),
      costByIdx(arma::rowvec(N).fill(arma::datum::nan)) {}

CandidatePool::~CandidatePool() = default;

size_t CandidatePool::nDims() const { return space_->size(); }

Configuration CandidatePool::operator[](size_t i) const {
  Configuration conf;
  space_->at(i, conf);
  return conf;
}

void CandidatePool::recordObservation(size_t idx, double cost, size_t iter) {
  assert(!std::isnan(cost));
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
  iterByIdx[idx] = iter;
  ++nObs;
}

void CandidatePool::recordFailedEvaluation(size_t idx, size_t iter) {
  // For failed evaluations we still record the iteration number so
  // that we can plot at what iteration we failed.
  iterByIdx[idx] = iter;
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

static arma::mat applyScale(const arma::mat &y, const std::string &scale) {
  if (scale == "linear")
    return y;
  if (scale == "log2")
    return arma::log2(y);
  if (scale == "ln")
    return arma::log(y);
  if (scale == "sqrt")
    return arma::sqrt(y);
  if (scale == "cbrt")
    return arma::pow(y, 1.0 / 3.0);
  return arma::log10(y); // "log10" and default
}

struct BananasEnsemble {
  std::vector<std::unique_ptr<MlpNet>> models;
  int step = 0; // used to randomize seed
  std::string scale_ = "log10";

  BananasEnsemble(int n, int hidden, int depth) {
    models.reserve(n);
    for (int i = 0; i < n; ++i)
      models.push_back(makeNet(hidden, depth));
  }

  void fit(const arma::mat &X, const arma::mat &y, int epochs) {
    // Train directly on the scale-transformed costs (no z-standardisation).
    // z-standardisation shifts targets each iteration as the observed range
    // grows, which destabilises warm-started weights and compresses the
    // contrast between good and bad configs.  The scale transform already
    // handles range compression.
    arma::mat yScaled = applyScale(y, scale_);

    size_t n = X.n_cols;
    int step = this->step++;
    for (size_t mi = 0; mi < models.size(); ++mi) {
      arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(
          mi * 1000003 + 7 + 399 * step));
      // Shuffle training-set order per member so mini-batches differ across
      // the ensemble, producing divergent gradient paths and diverse solutions.
      arma::uvec perm = arma::shuffle(arma::regspace<arma::uvec>(0, n - 1));
      arma::mat Xs = X.cols(perm);
      arma::mat ys = yScaled.cols(perm);
      int batchSize = std::min<size_t>(32, n);
      size_t stepsPerEpoch = (n + batchSize - 1) / batchSize;
      // For the first few fits, use more epochs to get a better initial fit
      size_t thisEpochs = step < 5 ? epochs * 5 : epochs;
      size_t maxIter = thisEpochs * stepsPerEpoch;
      ens::Adam opt(3e-3, batchSize, 0.9, 0.999, 1e-8, maxIter, 1e-7, true);
      models[mi]->Train(Xs, ys, opt);
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
    // mu/sigma are in scaled space. We intentionally do NOT invert the
    // transform: the acquisition function only needs correct ordering, which
    // any monotone transform preserves.
    arma::rowvec mu = arma::mean(preds, 0);
    arma::rowvec sigma = arma::stddev(preds, 0, 0);
    return {mu, sigma};
  }
};

// ===----------------------------------------------------------------------===//
// Acquisition function
// ===----------------------------------------------------------------------===//

// LCB acquisition
static arma::rowvec computeAcq(const arma::rowvec &mu,
                               const arma::rowvec &sigma, double kappa) {
  return mu - kappa * sigma;
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

  std::unordered_set<size_t> nextFrontier;
  llvm::SmallVector<size_t> nbrs;
  for (unsigned d = 0; d < depth && !frontier.empty(); ++d) {
    const bool isLastStep = (d + 1 == depth);
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
    std::swap(frontier, nextFrontier);
    nextFrontier.clear();
  }
}

void recordValidationData(ValidationSet &validSet, BananasEnsemble *ensemble_,
                          int iter) {
  if (!validSet.empty()) {
    auto [vmu, vsigma] = ensemble_->predict(validSet.encoded);
    validSet.recordSnapshot(iter, std::move(vmu), std::move(vsigma));
  }
}

bool CandidatePool::nextCandidateIndices(const InferenceOptions &opts,
                                         std::mt19937 &rng,
                                         std::function<bool(size_t)> accept,
                                         ValidationSet &validSet,
                                         ValidationSet &trainingValidSet,
                                         int iter) {
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

  // Warm-start: reuse weights from the previous iteration. Reinitialise only
  // when the ensemble doesn't exist yet or its configuration has changed.
  if (!ensemble_ ||
      ensemble_->models.size() != static_cast<size_t>(opts.nEnsemble))
    ensemble_ = std::make_unique<BananasEnsemble>(opts.nEnsemble, opts.hidden,
                                                  opts.depth);
  ensemble_->scale_ = opts.objectiveScale;
  ensemble_->fit(Xo_obs, yo_obs, opts.epochs);
  auto [mu, sigma] = ensemble_->predict(candEncoded);

  // validation logging
  recordValidationData(trainingValidSet, ensemble_.get(), iter);

  if (opts.validationInterval > 0 && iter % opts.validationInterval == 0) {
    recordValidationData(validSet, ensemble_.get(), iter);
  }

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
                              std::filesystem::path path) const {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
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

  // Use the warm-started ensemble to get per-candidate statistics.
  const bool hasModel = ensemble_ && nObs >= 2 && !validIdx.empty();
  // Per-valid-index predictions; indexed by position in validIdx.
  arma::rowvec mu_v, sigma_v, acq_v;
  if (hasModel) {
    arma::mat validEncoded = encodeSubset(*space_, validIdx);
    auto [m, s] = ensemble_->predict(validEncoded);
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
  out << "visited,valid,cost,eval_iter";
  if (hasModel)
    out << ",mu,sigma,acq";
  out << "\n";

  // One row per pool member
  Configuration conf;
  for (size_t i = 0; i < N; ++i) {
    space_->at(i, conf);
    for (int64_t v : conf)
      out << v << ",";
    out << (visited.test(i) ? 1 : 0) << "," << (validPos.count(i) ? 1 : 0)
        << ",";
    double c = costByIdx(i);
    if (!std::isnan(c))
      out << c;
    out << ",";
    auto it = iterByIdx.find(i);
    if (it != iterByIdx.end())
      out << it->second;
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

// ===----------------------------------------------------------------------===//
// ValidationSet
// ===----------------------------------------------------------------------===//

void ValidationSet::record(size_t idx, double cost) {
  indices.push_back(idx);
  trueCosts.push_back(cost);
  const size_t D = space_->size();
  Configuration conf;
  space_->at(idx, conf);
  if (encoded.is_empty())
    encoded.set_size(D, 0);
  encoded.insert_cols(encoded.n_cols, 1);
  for (size_t d = 0; d < D; ++d)
    encoded(d, encoded.n_cols - 1) = (*space_)[d].featurize(conf[d]);
}

void ValidationSet::recordSnapshot(int iter, arma::rowvec mu,
                                   arma::rowvec sigma) {
  snapshots.push_back({iter, std::move(mu), std::move(sigma)});
}

void ValidationSet::dumpToCSV(std::filesystem::path path) const {
  if (empty() || snapshots.empty())
    return;
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out)
    return;

  for (const auto &p : space_->params)
    out << p.name << ",";
  out << "cost,iter,mu,sigma\n";

  Configuration conf;
  for (const auto &snap : snapshots) {
    for (size_t j = 0; j < snap.mu.n_elem; ++j) {
      space_->at(indices[j], conf);
      for (int64_t v : conf)
        out << v << ",";
      out << trueCosts[j] << "," << snap.iter << "," << snap.mu(j) << ","
          << snap.sigma(j) << "\n";
    }
  }
}

} // namespace mlir::cinm
