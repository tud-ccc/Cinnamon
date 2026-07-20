#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <algorithm>
#include <armadillo>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/Threading.h>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <unordered_set>
#include <utility>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// CandidatePool construction
// ===----------------------------------------------------------------------===//

void CandidatePool::computeValidMask(const ConfigSpace &space,
                                     llvm::BitVector &validMask,
                                     std::vector<size_t> &validIndices) {
  validMask.clear();
  validMask.resize(space.totalSize());
  space.forEach([&](const Configuration &conf, size_t i) {
    if (space.isValid(conf)) {
      validMask.set(static_cast<unsigned>(i));
      validIndices.push_back(i);
    }
    return true;
  });
}

CandidatePool CandidatePool::build(const ConfigSpace &space, size_t evalBudget,
                                   bool exhaustive) {

  auto validMask = std::make_shared<llvm::BitVector>();
  auto validIndices = std::make_shared<std::vector<size_t>>();

  computeValidMask(space, *validMask, *validIndices);

  return CandidatePool(space, evalBudget, std::move(validMask),
                       std::move(validIndices), exhaustive);
}

CandidatePool::CandidatePool(const ConfigSpace &space, size_t evalBudget,
                             std::shared_ptr<llvm::BitVector> validMask,

                             std::shared_ptr<std::vector<size_t>> validIndices,
                             bool exhaustive)
    : space_(&space), N(space.totalSize()), validMask_(std::move(validMask)),
      validIndices_(std::move(validIndices)), Xo(space.size(), evalBudget),
      yo(1, evalBudget), exhaustive(exhaustive) {}

CandidatePool::~CandidatePool() = default;

size_t CandidatePool::nDims() const { return space_->size(); }

Configuration CandidatePool::operator[](size_t i) const {
  Configuration conf;
  space_->at(i, conf);
  return conf;
}

void CandidatePool::recordObservation(size_t idx, double cost, size_t iter,
                                      std::chrono::milliseconds evalTime,
                                      uint64_t cpuTimeMs) {
  // assert(!std::isnan(cost));
  if (nObs >= Xo.n_cols) {
    const size_t newCols = Xo.n_cols + 32;
    Xo.resize(Xo.n_rows, newCols);
    yo.resize(1, newCols);
  }
  if (!exhaustive) {
    Configuration conf;
    space_->at(idx, conf);
    for (size_t d = 0; d < space_->size(); ++d)
      Xo(d, nObs) = (*space_)[d].featurize(conf[d]);
    yo(0, nObs) = cost;
  }
  costByIdx[idx] = cost;
  iterByIdx[idx] = iter;
  if (evalTime.count() > 0)
    evalTimeByIdx[idx] = static_cast<uint64_t>(evalTime.count());
  if (cpuTimeMs > 0)
    cpuTimeByIdx[idx] = cpuTimeMs;
  ++nObs;
}

void CandidatePool::recordFailedEvaluation(size_t idx, size_t iter) {
  // For failed evaluations we still record the iteration number so
  // that we can plot at what iteration we failed.
  iterByIdx[idx] = iter;
}

// Build a temporary D×M encoded matrix for a set of candidate pool indices.
template <class Collection>
static arma::mat encodeSubset(const ConfigSpace &space,
                              const Collection &indices) {
  const size_t D = space.size();
  const size_t M = indices.size();
  arma::mat enc(D, M);
  Configuration conf;
  size_t j = 0;
  for (auto ix : indices) {
    space.at(ix, conf);
    for (size_t d = 0; d < D; ++d)
      enc(d, j) = space[d].featurize(conf[d]);
    j++;
  }
  return enc;
}
static arma::mat encodeValidSpace(const ConfigSpace &space,
                                  const CandidatePool &pool) {
  const size_t D = space.size();
  // Only valid configs get a column; sizing to pool.N (the full Cartesian
  // product) would waste — and can fail to allocate — many GB for large spaces.
  arma::mat enc(D, pool.size());

  size_t ix = 0;
  space.forEach([&](const Configuration &conf, size_t i) {
    if (!pool.isValid(i))
      return true;

    for (size_t d = 0; d < D; ++d)
      enc(d, ix) = space[d].featurize(conf[d]);
    ix++;
    return true;
  });
  return enc;
}
// ===----------------------------------------------------------------------===//
// Latin Hypercube Sampling
// ===----------------------------------------------------------------------===//

void CandidatePool::sampleInitialSet(size_t n, std::mt19937 &rng,
                                     std::function<bool(size_t)> accept,
                                     unsigned workers) {
  // size_t nAccepted = 0;
  // while (nAccepted < n) {
  //   std::unordered_set<size_t> result;
  //   size_t want = n - nAccepted;
  //   fillRandom(result, want, rng);
  //   for (auto i : result) {
  //     if (accept(i))
  //       nAccepted++;
  //   }
  //   if (result.size() < want || nAccepted >= n)
  //     return;
  //   result.clear();
  // }
  // return;

  const size_t D = nDims();
  const size_t M = size();
  if (n == 0 || M == 0)
    return;

  // DxM matrix
  arma::mat enc = encodeValidSpace(*space_, *this);

  // Per-dimension [0,1] normalisation.
  for (size_t d = 0; d < D; ++d) {
    double lo = enc.row(d).min();
    double hi = enc.row(d).max();
    double range = (hi > lo) ? (hi - lo) : 1.0;
    enc.row(d) = (enc.row(d) - lo) / range;
  }

  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::unordered_set<size_t> used;
  std::atomic<size_t> accepted{0};

  // accept() may run a simulator that takes seconds, so the accepted calls are
  // dispatched to a thread pool. The (cheap) LHS target generation and greedy
  // nearest-neighbour candidate selection stay on this thread; only accept()
  // runs concurrently. `workers` sizes the pool (same knob as exhaustive
  // search).
  llvm::DefaultThreadPool threadPool(
      llvm::hardware_concurrency(std::max(1u, workers)));

  // Keep generating LHS batches until n configurations pass accept().
  while (accepted.load(std::memory_order_relaxed) < n) {
    size_t want = n - accepted.load(std::memory_order_relaxed);

    size_t nUnused = M - used.size();
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
    // Dispatch each candidate to the thread pool immediately after selection
    // so that NN selection and evaluation overlap (selection is O(want×M)
    // and would otherwise block pool threads for the entire batch).
    size_t nDispatched = 0;
    for (size_t t = 0; t < want; ++t) {
      double bestDist = std::numeric_limits<double>::max();
      size_t bestPos = M; // position in candidates
      for (size_t i = 0; i < M; ++i) {
        if (used.count(i))
          continue;
        double dist = 0;
        for (size_t d = 0; d < D; ++d) {
          double diff = enc(d, i) - batchTargets[t][d];
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestPos = i;
        }
      }
      if (bestPos == M)
        break;
      used.insert(bestPos);
      ++nDispatched;
      size_t idx = (*validIndices_)[bestPos];
      threadPool.async([&accept, &accepted, idx, n]() {
        if (accepted.load(std::memory_order_relaxed) >= n)
          return;
        if (accept(idx))
          accepted.fetch_add(1, std::memory_order_relaxed);
      });
    }
    if (nDispatched == 0)
      break;
    // Barrier: the next batch's `want` depends on how many were accepted.
    threadPool.wait();
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
      size_t thisEpochs = step < 2 ? epochs * 3 : epochs;
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

bool CandidatePool::tryInsert(std::unordered_set<size_t> &result, size_t idx) {
  if (isVisited(idx) || !isValid(idx))
    return false;
  auto res = result.insert(idx);
  return res.second;
}

void CandidatePool::fillRandom(std::unordered_set<size_t> &result,
                               size_t target, std::mt19937 &rng) {
  if (result.size() >= target || empty())
    return;
  size_t numValid = size();
  size_t numVisited = this->numVisited();
  auto dist = std::uniform_int_distribution<size_t>(0, numValid - 1);

  size_t numAttempts = 0;
  while (result.size() < std::min(target, numValid - numVisited) &&
         numAttempts++ <= target * 5) {
    tryInsert(result, (*validIndices_)[dist(rng)]);
  }
}

void CandidatePool::fillNeighbors(std::unordered_set<size_t> &result,
                                  unsigned depth, bool frontierOnly) {
  // BFS outward from every observed point up to `depth` steps.
  std::unordered_set<size_t> frontier;

  for (auto [k, _] : costByIdx)
    frontier.insert(k);

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
          tryInsert(result, nb);
        }
        // Always track the frontier for BFS expansion regardless.
        if (!isVisited(nb))
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
  // Build compact training matrices excluding INF-cost observations.
  // Training the MLP on INF targets causes gradient explosion → NaN weights
  // → NaN predictions → arma::sort_index abort.
  arma::uvec finiteCols(nObs);
  arma::uword nFinite = 0;
  for (arma::uword i = 0; i < static_cast<arma::uword>(nObs); ++i)
    if (std::isfinite(yo(0, i)))
      finiteCols(nFinite++) = i;
  finiteCols.resize(nFinite);

  if (nFinite < 2) {
    // Not enough finite observations to train; pick the first accepted
    // candidate.
    for (size_t idx : candIdx)
      if (accept(idx))
        return true;
    return false;
  }

  arma::mat Xo_obs = Xo.cols(finiteCols);
  arma::mat yo_obs = yo.cols(finiteCols);

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
  // Guard: replace any NaN/Inf from MLP instability with +inf so they
  // land at the end of the sorted order and are never preferred.
  scores.for_each([](double &x) {
    if (!std::isfinite(x))
      x = arma::datum::inf;
  });
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

  // Use the warm-started ensemble to get per-candidate statistics.
  const bool hasModel = ensemble_ && nObs >= 2 && !empty();
  // Per-valid-index predictions; indexed by position in validIdx.
  arma::rowvec mu_v, sigma_v, acq_v;
  if (hasModel) {
    const size_t D = space_->size();
    const size_t confsToEncode =
        opts.dumpFullPool ? this->size() : visited.size();
    arma::mat encoded(D, confsToEncode);
    size_t ix = 0;
    space_->forEach([&](const Configuration &conf, size_t i) {
      if (!isValid(i) || !isVisited(i))
        return true;
      for (size_t d = 0; d < D; ++d)
        encoded(d, ix) = (*space_)[d].featurize(conf[d]);
      ix++;
      return true;
    });
    auto [m, s] = ensemble_->predict(encoded);
    mu_v = m;
    sigma_v = s;
    acq_v = computeAcq(mu_v, sigma_v, opts.kappa);
  }

  // Header
  for (const auto &p : space.params)
    out << p.name << ",";
  out << "visited,valid,cost,eval_iter,eval_time_ms,cpu_time_ms";
  if (hasModel)
    out << ",mu,sigma,acq";
  out << "\n";

  // One row per valid pool member, in flat-index order.
  size_t j = 0;
  space.forEach([&](auto &conf, size_t i) -> bool {
    if (!isValid(i) || (!opts.dumpFullPool && !isVisited(i)))
      return true;

    for (int64_t v : conf)
      out << v << ",";
    out << (visited.count(i) ? 1 : 0) << ",1,";
    auto cit = costByIdx.find(i);
    double c = (cit != costByIdx.end()) ? cit->second : arma::datum::nan;
    if (!std::isnan(c))
      out << c;
    out << ",";
    auto iit = iterByIdx.find(i);
    if (iit != iterByIdx.end())
      out << iit->second;
    out << ",";
    auto tit = evalTimeByIdx.find(i);
    if (tit != evalTimeByIdx.end())
      out << tit->second;
    out << ",";
    auto cit2 = cpuTimeByIdx.find(i);
    if (cit2 != cpuTimeByIdx.end())
      out << cit2->second;
    if (hasModel)
      out << "," << mu_v(j) << "," << sigma_v(j) << "," << acq_v(j);
    out << "\n";
    j++;
    return true;
  });
}

void CandidatePool::dumpMetadataJSON(const ConfigSpace &space,
                                     std::filesystem::path path) const {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out)
    return;

  auto jsonStr = [&](const std::string &s) {
    out << '"';
    for (char c : s) {
      if (c == '"' || c == '\\')
        out << '\\';
      out << c;
    }
    out << '"';
  };

  out << "{\n";
  out << "  \"total_size\": " << N << ",\n";
  out << "  \"n_valid\": " << size() << ",\n";
  out << "  \"params\": [\n";
  for (size_t i = 0; i < space.params.size(); ++i) {
    const auto &p = space.params[i];
    out << "    {";
    out << "\"name\": ";
    jsonStr(p.name);
    out << ", ";
    out << "\"cardinality\": " << p.cardinality() << ", ";
    if (auto *r = std::get_if<IntRange>(&p.domain)) {
      out << "\"type\": \"range\", ";
      out << "\"lo\": " << r->lo << ", ";
      out << "\"hi\": " << r->hi << ", ";
      out << "\"step\": " << r->step;
    } else if (auto *v = std::get_if<ValueList>(&p.domain)) {
      out << "\"type\": \"values\", \"values\": [";
      for (size_t j = 0; j < v->values.size(); ++j) {
        if (j)
          out << ", ";
        out << v->values[j];
      }
      out << "]";
    }
    out << "}";
    if (i + 1 < space.params.size())
      out << ",";
    out << "\n";
  }
  out << "  ]\n}\n";
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
