#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <llvm/Support/Debug.h>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>
#include <variant>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// CandidatePool construction
// ===----------------------------------------------------------------------===//

struct ConfigHash {
  size_t operator()(const Configuration &c) const {
    size_t h = c.size();
    for (int64_t v : c)
      h ^= static_cast<size_t>(v) + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
  }
};

CandidatePool CandidatePool::sample(const ConfigSpace &space, size_t maxPool,
                                    std::mt19937 &rng) {
  const size_t D = space.size();
  std::vector<Configuration> configs;
  std::unordered_set<Configuration, ConfigHash> seen;
  configs.reserve(maxPool);

  for (size_t tries = 0, limit = maxPool * 50;
       tries < limit && configs.size() < maxPool; ++tries) {
    Configuration config(D);
    for (size_t i = 0; i < D; ++i) {
      std::uniform_real_distribution<double> dist(space[i].dlo(),
                                                  space[i].dhi());
      config[i] = space[i].discretize(dist(rng));
    }
    if (space.isValid(config) && seen.insert(config).second)
      configs.push_back(config);
  }

  // Encode into D×N arma::mat (column-major: sample i is column i).
  const size_t N = configs.size();
  arma::mat encoded(D, N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t d = 0; d < D; ++d) {
      auto &dim = space[d];
      encoded(d, i) = dim.featurize(configs[i][d]);
    }
  }

  return CandidatePool(std::move(configs), std::move(encoded));
}

// ===----------------------------------------------------------------------===//
// Latin Hypercube Sampling
// ===----------------------------------------------------------------------===//

llvm::SmallVector<size_t> CandidatePool::lhsIndices(size_t n,
                                                    unsigned seed) const {
  const size_t N = size();
  const size_t D = nDims();
  std::mt19937 rng(seed);
  n = std::min(n, N);
  if (n == 0)
    return {};

  // Per-dimension [0,1] normalisation of the already-encoded pool.
  arma::mat normed(D, N);
  for (size_t d = 0; d < D; ++d) {
    double lo = encoded.row(d).min();
    double hi = encoded.row(d).max();
    double range = (hi > lo) ? (hi - lo) : 1.0;
    normed.row(d) = (encoded.row(d) - lo) / range;
  }

  // Generate n LHS target points — one stratum per dimension.
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<std::vector<double>> targets(n, std::vector<double>(D));
  for (size_t d = 0; d < D; ++d) {
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (size_t i = 0; i < n; ++i)
      targets[i][d] =
          (static_cast<double>(perm[i]) + u01(rng)) / static_cast<double>(n);
  }

  // Greedy nearest-neighbour: each target → closest unused candidate.
  std::vector<bool> used(N, false);
  llvm::SmallVector<size_t> selected;
  selected.reserve(n);
  for (size_t t = 0; t < n; ++t) {
    double bestDist = std::numeric_limits<double>::max();
    size_t bestIdx = 0;
    for (size_t i = 0; i < N; ++i) {
      if (used[i])
        continue;
      double dist = 0;
      for (size_t d = 0; d < D; ++d) {
        double diff = normed(d, i) - targets[t][d];
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    used[bestIdx] = true;
    selected.push_back(bestIdx);
  }
  return selected;
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
    // Train in log10 space: compresses wide cost ranges (e.g. 1e2–1e5) into
    // ~3 units, making the landscape far smoother for the MLP to learn.
    // log10 is monotone so the argmin is preserved — the surrogate still
    // selects the lowest-cost candidate.
    arma::mat yLog = arma::log10(y);
    yMean = arma::mean(arma::vectorise(yLog));
    double s = arma::stddev(arma::vectorise(yLog));
    yStd = (s > 1e-8) ? s : 1.0;
    arma::mat yNorm = (yLog - yMean) / yStd;

    size_t n = X.n_cols;
    for (size_t mi = 0; mi < models.size(); ++mi) {
      arma::arma_rng::set_seed(
          static_cast<arma::arma_rng::seed_type>(mi * 1000003 + 7));
      arma::uvec idx = arma::randi<arma::uvec>(
          n, arma::distr_param(0, static_cast<int>(n) - 1));
      arma::mat Xb = X.cols(idx);
      arma::mat yb = yNorm.cols(idx);
      // ensmallen's maxIterations counts gradient updates, not epochs.
      // Compute steps-per-epoch so the training budget scales with the dataset
      // size, not with raw sample count (which caused ~100x overtraining
      // before). Cap batchSize at n to avoid undefined behaviour when n < 32.
      int batchSize = std::min<size_t>(32, n);
      size_t stepsPerEpoch = (n + batchSize - 1) / batchSize;
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
// Next-candidate selection
// ===----------------------------------------------------------------------===//

llvm::SmallVector<size_t>
CandidatePool::nextCandidateIndices(const InferenceOptions &opts,
                                    size_t k) const {
  const double kappa = opts.kappa;
  const int epochs = opts.epochs;
  const int nEnsemble = opts.nEnsemble;
  const int hidden = opts.hidden;
  const int depth = opts.depth;

  // Zero-copy views of the preallocated observation matrices.
  arma::mat Xo_obs(const_cast<double *>(Xo.memptr()), nDims(), nObs,
                   /*copy=*/false, /*strict=*/true);
  arma::mat yo_obs(const_cast<double *>(yo.memptr()), 1, nObs,
                   /*copy=*/false, /*strict=*/true);

  BananasEnsemble ensemble(nEnsemble, hidden, depth);
  ensemble.fit(Xo_obs, yo_obs, epochs);
  auto [mu, sigma] = ensemble.predict(encoded);

  arma::rowvec scores = mu - kappa * sigma;
  arma::uvec order = arma::sort_index(scores, "ascend");

  // Walk in score order and collect the first k unvisited pool indices.
  llvm::SmallVector<size_t> result;
  result.reserve(k);
  for (size_t i = 0; i < order.n_elem && result.size() < k; ++i) {
    size_t idx = order(i);
    if (!visited.test(idx))
      result.push_back(idx);
  }
  return result;
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

  // Refit the ensemble on all observations to get per-candidate statistics.
  // Skipped when we have too few points to train on.
  const bool hasModel = nObs >= 2;
  arma::rowvec mu, sigma, acq;
  if (hasModel) {
    arma::mat Xo_obs(const_cast<double *>(Xo.memptr()), nDims(), nObs,
                     /*copy=*/false, /*strict=*/true);
    arma::mat yo_obs(const_cast<double *>(yo.memptr()), 1, nObs,
                     /*copy=*/false, /*strict=*/true);
    BananasEnsemble ensemble(opts.nEnsemble, opts.hidden, opts.depth);
    ensemble.fit(Xo_obs, yo_obs, opts.epochs);
    auto [m, s] = ensemble.predict(encoded);
    mu = m;
    sigma = s;
    acq = mu - opts.kappa * sigma;
  }

  // Header
  for (const auto &p : space.params)
    out << p.name << ",";
  out << "visited,cost";
  if (hasModel)
    out << ",mu,sigma,acq";
  out << "\n";

  // One row per pool member
  for (size_t i = 0; i < configs.size(); ++i) {
    for (int64_t v : configs[i])
      out << v << ",";
    out << (visited.test(i) ? 1 : 0) << ",";
    double c = costByIdx(i);
    if (!std::isnan(c))
      out << c;
    if (hasModel)
      out << "," << mu(i) << "," << sigma(i) << "," << acq(i);
    out << "\n";
  }
}

} // namespace mlir::cinm
