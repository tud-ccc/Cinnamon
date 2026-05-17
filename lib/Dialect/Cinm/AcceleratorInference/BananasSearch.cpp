#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>

// Suppress mlpack's own info/warning streams — we only want LLVM diagnostics.
#ifndef MLPACK_NO_STD_COUT_PRINT
#define MLPACK_NO_STD_COUT_PRINT
#endif

#include <mlpack.hpp>

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Encoding
// ===----------------------------------------------------------------------===//

std::vector<float> encodeConfig(const ConfigSpace &space,
                                const Configuration &config) {
  std::vector<float> features;
  features.reserve(space.size());
  for (size_t i = 0; i < space.size(); ++i) {
    double lo = space[i].dlo(), hi = space[i].dhi();
    double val = static_cast<double>(config[i]);
    float norm = (hi > lo) ? static_cast<float>((val - lo) / (hi - lo)) : 0.0f;
    features.push_back(norm);
  }
  return features;
}

// ===----------------------------------------------------------------------===//
// LHS sampling
// ===----------------------------------------------------------------------===//

llvm::SmallVector<size_t> lhsIndices(const std::vector<float> &encodedFlat,
                                     size_t N, size_t D, size_t n,
                                     unsigned seed) {
  std::mt19937 rng(seed);
  n = std::min(n, N);
  if (n == 0)
    return {};

  // Per-dimension [0,1] normalisation of the candidate matrix.
  std::vector<float> normed(N * D);
  for (size_t d = 0; d < D; ++d) {
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < N; ++i) {
      lo = std::min(lo, encodedFlat[i * D + d]);
      hi = std::max(hi, encodedFlat[i * D + d]);
    }
    float range = (hi > lo) ? (hi - lo) : 1.0f;
    for (size_t i = 0; i < N; ++i)
      normed[i * D + d] = (encodedFlat[i * D + d] - lo) / range;
  }

  // Generate n LHS target points — one stratum per dimension.
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  std::vector<std::vector<float>> targets(n, std::vector<float>(D));
  for (size_t d = 0; d < D; ++d) {
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (size_t i = 0; i < n; ++i)
      targets[i][d] = (static_cast<float>(perm[i]) + u01(rng)) / static_cast<float>(n);
  }

  // Greedy nearest-neighbour assignment: each LHS target → closest unused candidate.
  std::vector<bool> used(N, false);
  llvm::SmallVector<size_t> selected;
  selected.reserve(n);
  for (size_t t = 0; t < n; ++t) {
    float bestDist = std::numeric_limits<float>::max();
    size_t bestIdx = 0;
    for (size_t i = 0; i < N; ++i) {
      if (used[i])
        continue;
      float dist = 0;
      for (size_t d = 0; d < D; ++d) {
        float diff = normed[i * D + d] - targets[t][d];
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

using MlpNet = mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>;

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
    // Normalise targets for stable training.
    yMean = arma::mean(arma::vectorise(y));
    double s = arma::stddev(arma::vectorise(y));
    yStd = (s > 1e-8) ? s : 1.0;
    arma::mat yNorm = (y - yMean) / yStd;

    size_t n = X.n_cols;
    for (size_t mi = 0; mi < models.size(); ++mi) {
      // Different seed per model → different initialisation + bootstrap resample.
      arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(mi * 1000003 + 7));

      arma::uvec idx = arma::randi<arma::uvec>(n, arma::distr_param(0, static_cast<int>(n) - 1));
      arma::mat Xb = X.cols(idx);
      arma::mat yb = yNorm.cols(idx);

      size_t maxIter = static_cast<size_t>(epochs) * n;
      ens::Adam opt(3e-3, 32, 0.9, 0.999, 1e-8, maxIter, 1e-7, true);
      models[mi]->Train(Xb, yb, opt);
    }
  }

  // Returns (mu, sigma), both row-vectors of length nPool.
  std::pair<arma::rowvec, arma::rowvec> predict(arma::mat &Xp) const {
    size_t M = Xp.n_cols;
    arma::mat preds(models.size(), M);
    for (size_t mi = 0; mi < models.size(); ++mi) {
      arma::mat out;
      models[mi]->Predict(Xp, out); // out: (1, M)
      preds.row(mi) = out.row(0);
    }
    // Denormalise: each row is normalised predictions from one model.
    arma::rowvec mu = arma::mean(preds, 0) * yStd + yMean;
    arma::rowvec sigma = arma::stddev(preds, 0, 0) * yStd;
    return {mu, sigma};
  }
};

// ===----------------------------------------------------------------------===//
// Next-candidate selection
// ===----------------------------------------------------------------------===//

/// Convert a flat row-major float buffer (nSamples × D) to an Armadillo matrix
/// in column-major format (D × nSamples) as expected by mlpack.
static arma::mat toArma(const std::vector<float> &flat, size_t nSamples,
                        size_t D) {
  arma::mat M(D, nSamples);
  for (size_t i = 0; i < nSamples; ++i)
    for (size_t d = 0; d < D; ++d)
      M(d, i) = static_cast<double>(flat[i * D + d]);
  return M;
}

llvm::SmallVector<size_t>
nextCandidateIndices(const std::vector<float> &X_obs, size_t nObs,
                     const std::vector<float> &y_obs,
                     const std::vector<float> &X_pool, size_t nPool, size_t D,
                     int k, float kappa, int epochs, int nEnsemble, int hidden,
                     int depth) {
  arma::mat Xo = toArma(X_obs, nObs, D);
  arma::mat yo(1, nObs);
  for (size_t i = 0; i < nObs; ++i)
    yo(0, i) = static_cast<double>(y_obs[i]);
  arma::mat Xp = toArma(X_pool, nPool, D);

  BananasEnsemble ensemble(nEnsemble, hidden, depth);
  ensemble.fit(Xo, yo, epochs);
  auto [mu, sigma] = ensemble.predict(Xp);

  // UCB for minimisation: lower score = more promising.
  arma::rowvec scores = mu - static_cast<double>(kappa) * sigma;

  arma::uvec order = arma::sort_index(scores, "ascend");
  size_t kActual = std::min(static_cast<size_t>(k), nPool);
  llvm::SmallVector<size_t> result;
  result.reserve(kActual);
  for (size_t i = 0; i < kActual; ++i)
    result.push_back(static_cast<size_t>(order(i)));
  return result;
}

} // namespace mlir::cinm
