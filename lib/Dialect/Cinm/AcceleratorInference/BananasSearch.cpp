#include "BananasSearch.h"
#include "SearchStrategy.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <algorithm>
#include <armadillo>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Parallel.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/Threading.h>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>
#include <utility>

#define DEBUG_TYPE "cinm-inference"

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// CandidatePool construction
// ===----------------------------------------------------------------------===//
CandidatePool::CandidatePool(const ConfigSpace &space, size_t evalBudget,
                             const InferenceOptions &opts)
    : space_(&space), N(space.totalSize()),
      // Exhaustive search never reads/writes Xo/yo (see recordObservation);
      // its evalBudget is the full totalSize(), which would otherwise try to
      // allocate a dense D×N matrix for a matrix that's never used.
      Xo(space.numFeatures(), opts.exhaustiveSearch ? 0 : evalBudget),
      yo(1, opts.exhaustiveSearch ? 0 : evalBudget), opts(opts) {}

void CandidatePool::recordObservation(size_t idx, double cost, size_t iter,
                                      std::chrono::milliseconds evalTime,
                                      uint64_t cpuTimeMs) {
  // assert(!std::isnan(cost));
  if (!opts.exhaustiveSearch) {
    if (nObs >= Xo.n_cols) {
      const size_t newCols = Xo.n_cols + 32;
      Xo.resize(Xo.n_rows, newCols);
      yo.resize(1, newCols);
    }
    Configuration conf;
    space_->at(idx, conf);
    llvm::SmallVector<double, 16> features;
    space_->encode(conf, features);
    for (size_t d = 0; d < features.size(); ++d)
      Xo(d, nObs) = features[d];
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

// Build a temporary D×M encoded matrix for a set of candidate pool indices,
// where D is the feature count rather than the parameter count.
template <class Collection>
static arma::mat encodeSubset(const ConfigSpace &space,
                              const Collection &indices) {
  const size_t D = space.numFeatures();
  const size_t M = indices.size();
  arma::mat enc(D, M);
  Configuration conf;
  llvm::SmallVector<double, 16> features;
  size_t j = 0;
  for (auto ix : indices) {
    space.at(ix, conf);
    features.clear();
    space.encode(conf, features);
    for (size_t d = 0; d < D; ++d)
      enc(d, j) = features[d];
    j++;
  }
  return enc;
}
/// The flat indices of the whole pool, [0, N), as encodeSubset wants them.
static std::vector<size_t> allIndices(size_t n) {
  std::vector<size_t> out(n);
  std::iota(out.begin(), out.end(), size_t{0});
  return out;
}

// ===----------------------------------------------------------------------===//
// Initial-set sampling: Latin Hypercube or uniform
// ===----------------------------------------------------------------------===//

void CandidatePool::sampleInitialSet(size_t n, std::mt19937 &rng,
                                     std::function<bool(size_t)> accept,
                                     unsigned workers,
                                     InferenceOptions::SamplingMode mode) {
  const size_t D = numFeatures();
  const size_t M = size();
  if (n == 0 || M == 0)
    return;

  const bool lhs = mode == InferenceOptions::SamplingMode::LHS;

  // LHS only: the normalised encoding the targets are snapped against.
  arma::mat enc;
  if (lhs) {
    // DxM matrix, one column per config.
    enc = encodeSubset(*space_, allIndices(M));

    // Per-dimension [0,1] normalisation.
    for (size_t d = 0; d < D; ++d) {
      double lo = enc.row(d).min();
      double hi = enc.row(d).max();
      double range = (hi > lo) ? (hi - lo) : 1.0;
      enc.row(d) = (enc.row(d) - lo) / range;
    }
  }

  // Uniform only: a lazy Fisher-Yates over every index. Drawing the prefix one
  // swap at a time makes each remaining index equally likely at every step,
  // which is the property the mode exists for; rejection sampling against
  // `used` would only approximate it, and degrades as the draw approaches M.
  std::vector<size_t> shuffled;
  size_t nDrawn = 0;
  if (!lhs) {
    shuffled.resize(M);
    std::iota(shuffled.begin(), shuffled.end(), 0);
  }

  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::unordered_set<size_t> used;
  std::atomic<size_t> accepted{0};

  // accept() may run a simulator that takes seconds, so the accepted calls are
  // dispatched to a thread pool. Candidate selection -- LHS target generation
  // and greedy nearest-neighbour, or the Fisher-Yates swap -- stays on this
  // thread; only accept() runs concurrently. `workers` sizes the pool (same
  // knob as exhaustive search).
  llvm::DefaultThreadPool threadPool(
      llvm::hardware_concurrency(std::max(1u, workers)));

  // Selection happens inline rather than into a batch vector so that it
  // overlaps evaluation: LHS selection is O(want×M) and would otherwise block
  // pool threads for the whole batch.
  size_t nDispatched = 0;
  auto dispatch = [&](size_t idx) {
    used.insert(idx);
    ++nDispatched;
    threadPool.async([&accept, &accepted, idx, n]() {
      if (accepted.load(std::memory_order_relaxed) >= n)
        return;
      if (accept(idx))
        accepted.fetch_add(1, std::memory_order_relaxed);
    });
  };

  // Keep generating batches until n configurations pass accept(). Both modes
  // draw without replacement, so an index accept() rejects is never offered
  // again and the accepted set stays a draw from what accept() would take.
  while (accepted.load(std::memory_order_relaxed) < n) {
    size_t want = n - accepted.load(std::memory_order_relaxed);

    size_t nUnused = M - used.size();
    if (nUnused == 0)
      break;
    want = std::min(want, nUnused);

    nDispatched = 0;
    if (lhs) {
      // LHS targets for this batch.
      std::vector<std::vector<double>> batchTargets(want,
                                                    std::vector<double>(D));
      for (size_t d = 0; d < D; ++d) {
        std::vector<size_t> perm(want);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (size_t i = 0; i < want; ++i)
          batchTargets[i][d] = (static_cast<double>(perm[i]) + u01(rng)) /
                               static_cast<double>(want);
      }

      // Greedy nearest-neighbour: each target → closest unused candidate.
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
        dispatch(bestPos);
      }
    } else {
      for (size_t t = 0; t < want && nDrawn < M; ++t) {
        std::uniform_int_distribution<size_t> pick(nDrawn, M - 1);
        std::swap(shuffled[nDrawn], shuffled[pick(rng)]);
        dispatch(shuffled[nDrawn++]);
      }
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
// BANANAS strategy
// ===----------------------------------------------------------------------===//

/// BANANAS-style BO policy: each round fits the MLP ensemble on the pool's
/// observations and draws a batch from an acquisition function over a
/// neighbour+random candidate set. Owns the warm-started ensemble, the
/// per-round diagnostics, and the validation-snapshot pacing.
class BananasStrategy final : public SearchStrategy {
public:
  BananasStrategy(CandidatePool &pool, ValidationSet &validSet,
                  ValidationSet &trainingValidSet)
      : pool(pool), space_(pool.space_), opts(pool.opts), validSet(validSet),
        trainingValidSet(trainingValidSet) {}

  size_t step(std::mt19937 &rng, const std::function<bool(size_t)> &accept,
              int round, size_t nObsAtRound, size_t batchSize,
              unsigned workers) override;

  bool hasModel() const override { return ensemble_ != nullptr; }

  bool predict(const arma::mat &X, arma::rowvec &mu,
               arma::rowvec &sigma) const override {
    if (!ensemble_)
      return false;
    std::tie(mu, sigma) = ensemble_->predict(X);
    return true;
  }

  void dumpDiagnostics(const std::filesystem::path &dir) const override {
    diag.dumpRoundsCSV(dir / "rounds.csv");
    diag.dumpBatchesCSV(*space_, dir / "batchdiag.csv");
  }

private:
  /// Append this round's RoundRecord (selection fields left for step to fill
  /// once a candidate is accepted) and one BatchRecord per configured batch
  /// size.
  void recordRoundDiagnostics(std::mt19937 &rng, int round, size_t nObs,
                              llvm::ArrayRef<size_t> candIdx,
                              const arma::mat &candEncoded,
                              size_t nNeighborCands, const arma::uvec &order,
                              const arma::uvec &orderMu,
                              const arma::uvec &orderSigma,
                              llvm::ArrayRef<arma::uword> selected);

  CandidatePool &pool;
  const ConfigSpace *space_;
  const InferenceOptions &opts;
  ValidationSet &validSet;
  ValidationSet &trainingValidSet;
  /// Warm-start ensemble: persisted across rounds so each step fine-tunes
  /// from the previous fit rather than reinitialising from random weights.
  std::unique_ptr<BananasEnsemble> ensemble_;
  /// Filled per round when opts.dumpDir is set.
  SearchDiagnostics diag;
};

// ===----------------------------------------------------------------------===//
// Acquisition function
// ===----------------------------------------------------------------------===//

// LCB acquisition
static arma::rowvec computeAcq(const arma::rowvec &mu,
                               const arma::rowvec &sigma, double kappa) {
  return mu - kappa * sigma;
}

/// Pick `q` distinct candidate positions to evaluate this round.
///
/// LCB takes the ranking's head, which for q > 1 is q neighbours of one
/// optimum. Thompson draws an independent posterior sample per slot and takes
/// each draw's minimum: two slots agree wherever the ensemble is confident and
/// diverge wherever it is not, so the batch's spread is the surrogate's own
/// uncertainty rather than a separately tuned quantity.
static llvm::SmallVector<arma::uword> selectBatch(const InferenceOptions &opts,
                                                  const arma::rowvec &mu,
                                                  const arma::rowvec &sigma,
                                                  const arma::uvec &acqOrder,
                                                  size_t q, std::mt19937 &rng) {
  const size_t M = acqOrder.n_elem;
  q = std::min(q, M);
  llvm::SmallVector<arma::uword> batch;
  batch.reserve(q);

  if (opts.acquisition == InferenceOptions::Acquisition::LCB) {
    for (size_t i = 0; i < q; ++i)
      batch.push_back(acqOrder(i));
    return batch;
  }

  std::unordered_set<arma::uword> taken;
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (size_t slot = 0; slot < q; ++slot) {
    double best = std::numeric_limits<double>::infinity();
    arma::uword bestPos = M;
    for (arma::uword i = 0; i < static_cast<arma::uword>(mu.n_elem); ++i) {
      if (taken.count(i))
        continue;
      const double s = std::isfinite(sigma(i)) ? sigma(i) : 0.0;
      const double m = std::isfinite(mu(i)) ? mu(i) : arma::datum::inf;
      const double draw = m + s * gauss(rng);
      if (draw < best) {
        best = draw;
        bestPos = i;
      }
    }
    // Every candidate drew a non-finite score, so this slot has nothing to
    // add; the acquisition ranking backfills the rest.
    if (bestPos == M)
      break;
    taken.insert(bestPos);
    batch.push_back(bestPos);
  }
  for (size_t i = 0; batch.size() < q && i < M; ++i)
    if (taken.insert(acqOrder(i)).second)
      batch.push_back(acqOrder(i));
  return batch;
}

/// Run `accept` over `batch`, concurrently when `workers > 1`. Returns the
/// number accepted and records the outcome against each selection.
static size_t evaluateBatch(llvm::ArrayRef<arma::uword> batch,
                            llvm::ArrayRef<size_t> candIdx,
                            const std::function<bool(size_t)> &accept,
                            unsigned workers,
                            llvm::MutableArrayRef<SelectionRecord> outcomes) {
  std::atomic<size_t> accepted{0};
  auto runOne = [&](size_t slot) {
    const bool ok = accept(candIdx[batch[slot]]);
    if (ok)
      accepted.fetch_add(1, std::memory_order_relaxed);
    if (slot < outcomes.size())
      outcomes[slot].accepted = ok;
  };

  if (workers <= 1 || batch.size() <= 1) {
    for (size_t i = 0; i < batch.size(); ++i)
      runOne(i);
    return accepted.load();
  }

  // Never wider than the batch: a round dispatches one task per selection, so
  // threads past that would be spawned only to find nothing to run. This pool
  // is rebuilt every round, which a search spends microseconds on against
  // evaluations that take seconds.
  llvm::DefaultThreadPool threadPool(
      llvm::hardware_concurrency(std::min<unsigned>(workers, batch.size())));
  for (size_t i = 0; i < batch.size(); ++i)
    threadPool.async([&runOne, i]() { runOne(i); });
  threadPool.wait();
  return accepted.load();
}

/// Run `accept` over `indices`, concurrently when `workers > 1`. Returns the
/// number accepted. The model-free strategies' evaluation loop; evaluateBatch
/// above additionally records per-selection outcomes for the BANANAS
/// diagnostics.
static size_t evaluateIndices(llvm::ArrayRef<size_t> indices,
                              const std::function<bool(size_t)> &accept,
                              unsigned workers) {
  std::atomic<size_t> accepted{0};
  if (workers <= 1 || indices.size() <= 1) {
    for (size_t idx : indices)
      if (accept(idx))
        ++accepted;
    return accepted.load();
  }
  llvm::DefaultThreadPool threadPool(
      llvm::hardware_concurrency(std::min<unsigned>(workers, indices.size())));
  for (size_t idx : indices)
    threadPool.async([&accept, &accepted, idx]() {
      if (accept(idx))
        accepted.fetch_add(1, std::memory_order_relaxed);
    });
  threadPool.wait();
  return accepted.load();
}

// ===----------------------------------------------------------------------===//
// Search diagnostics
// ===----------------------------------------------------------------------===//

/// Rescale every feature to [0,1] across the candidate set, so that distances
/// weigh each dimension by the spread the candidates actually cover rather than
/// by the units the encoder happened to emit.
static arma::mat normaliseFeatures(const arma::mat &enc) {
  arma::mat out = enc;
  for (arma::uword d = 0; d < out.n_rows; ++d) {
    const double lo = out.row(d).min(), hi = out.row(d).max();
    out.row(d) = (out.row(d) - lo) / ((hi > lo) ? (hi - lo) : 1.0);
  }
  return out;
}

static double meanPairwiseDistance(const arma::mat &enc,
                                   llvm::ArrayRef<arma::uword> cols) {
  if (cols.size() < 2)
    return 0.0;
  double sum = 0;
  size_t n = 0;
  for (size_t i = 0; i + 1 < cols.size(); ++i)
    for (size_t j = i + 1; j < cols.size(); ++j, ++n)
      sum += arma::norm(enc.col(cols[i]) - enc.col(cols[j]), 2);
  return n ? sum / n : 0.0;
}

/// Median distance between randomly drawn candidate pairs. Serves as the RBF
/// width: at this scale a pair of typical candidates is neither identical nor
/// unrelated, which is what makes the effective batch size read as a fraction
/// of the candidate set's own spread instead of an absolute length.
static double medianPairDistance(const arma::mat &enc, std::mt19937 &rng) {
  const arma::uword M = enc.n_cols;
  if (M < 2)
    return 1.0;
  constexpr size_t kPairs = 256;
  std::uniform_int_distribution<arma::uword> pick(0, M - 1);
  std::vector<double> dists;
  dists.reserve(kPairs);
  for (size_t k = 0; k < kPairs; ++k) {
    arma::uword a = pick(rng), b = pick(rng);
    if (a != b)
      dists.push_back(arma::norm(enc.col(a) - enc.col(b), 2));
  }
  if (dists.empty())
    return 1.0;
  auto mid = dists.begin() + dists.size() / 2;
  std::nth_element(dists.begin(), mid, dists.end());
  return *mid > 0 ? *mid : 1.0;
}

/// Fraction of `batch` that appears in `b`'s first `batch.size()` entries.
static double topQOverlap(llvm::ArrayRef<arma::uword> batch,
                          const arma::uvec &b) {
  const size_t q = batch.size();
  std::unordered_set<arma::uword> top(b.begin(), b.begin() + q);
  size_t hits = 0;
  for (arma::uword x : batch)
    hits += top.count(x);
  return q ? static_cast<double>(hits) / static_cast<double>(q) : 0.0;
}

/// Measure `batch` for internal spread and for how much of it the mean and the
/// spread each account for on their own. Serves both the batch a round actually
/// evaluated and the hypothetical top-q of the ranking, which is what lets the
/// two be compared at the same q.
static BatchRecord measureBatch(int round, llvm::ArrayRef<arma::uword> batch,
                                const arma::mat &normEnc, double bandwidth,
                                const arma::uvec &orderMu,
                                const arma::uvec &orderSigma,
                                llvm::ArrayRef<ParmValue> candDimVals,
                                llvm::ArrayRef<size_t> dimDistinct,
                                std::mt19937 &rng, bool selected) {
  const size_t q = batch.size();
  BatchRecord rec{};
  rec.round = round;
  rec.q = q;
  rec.selected = selected;

  // Effective batch size: q² / ΣΣ k, which is q for mutually distant members
  // and 1 for a batch that has collapsed onto a single location.
  auto effectiveSize = [&](llvm::ArrayRef<arma::uword> cols) {
    if (cols.empty())
      return 1.0;
    double kSum = 0;
    for (arma::uword a : cols)
      for (arma::uword b : cols) {
        double d = arma::norm(normEnc.col(a) - normEnc.col(b), 2);
        kSum += std::exp(-(d * d) / (2 * bandwidth * bandwidth));
      }
    const double n = static_cast<double>(cols.size());
    return kSum > 0 ? (n * n) / kSum : 1.0;
  };
  rec.qEff = effectiveSize(batch);
  rec.meanPdist = meanPairwiseDistance(normEnc, batch);

  // Reference batch: same size, drawn uniformly from the candidates. The
  // ratios against it are what say whether the acquisition concentrated the
  // batch or merely inherited whatever spread the candidate set had.
  llvm::SmallVector<arma::uword> randBatch;
  std::uniform_int_distribution<arma::uword> pick(0, normEnc.n_cols - 1);
  std::unordered_set<arma::uword> seen;
  for (size_t guard = 0; randBatch.size() < q && guard < q * 10; ++guard) {
    arma::uword c = pick(rng);
    if (seen.insert(c).second)
      randBatch.push_back(c);
  }
  rec.qEffRef = effectiveSize(randBatch);
  double refPdist = meanPairwiseDistance(normEnc, randBatch);
  rec.dispersion = refPdist > 0 ? rec.meanPdist / refPdist : 0.0;

  rec.overlapMu = topQOverlap(batch, orderMu);
  rec.overlapSigma = topQOverlap(batch, orderSigma);

  // Per-dimension entropy of the values the batch spans, normalised so that 1
  // is the most spread a batch of this size could be over the values the
  // candidate set offers on that dimension.
  const size_t nDims = dimDistinct.size();
  rec.dimEntropy.assign(nDims, 0.0);
  for (size_t d = 0; d < nDims; ++d) {
    std::unordered_map<ParmValue, size_t> counts;
    for (arma::uword c : batch)
      counts[candDimVals[c * nDims + d]]++;
    double h = 0;
    for (auto [_, n] : counts) {
      double p = static_cast<double>(n) / static_cast<double>(q);
      h -= p * std::log(p);
    }
    double maxH = std::log(static_cast<double>(std::min(q, dimDistinct[d])));
    rec.dimEntropy[d] = maxH > 0 ? h / maxH : 0.0;
  }
  return rec;
}

/// Position of candidate `pos` within a ranking.
static size_t rankOf(const arma::uvec &order, arma::uword pos) {
  for (size_t i = 0; i < order.n_elem; ++i)
    if (order(i) == pos)
      return i;
  return order.n_elem;
}

void BananasStrategy::recordRoundDiagnostics(
    std::mt19937 &rng, int round, size_t nObs, llvm::ArrayRef<size_t> candIdx,
    const arma::mat &candEncoded, size_t nNeighborCands,
    const arma::uvec &order, const arma::uvec &orderMu,
    const arma::uvec &orderSigma, llvm::ArrayRef<arma::uword> selected) {
  RoundRecord rec{};
  rec.round = round;
  rec.nObs = nObs;
  rec.nCand = candIdx.size();
  rec.nNeighbor = nNeighborCands;
  // Timings and selections are filled in once the round has run.
  diag.rounds.push_back(std::move(rec));

  // Decode every candidate once: the entropies below are per-dimension over
  // configuration values, which the encoded features do not expose (one
  // parameter may span several features, and one-hot columns are not values).
  const size_t nDims = space_->numDims();
  std::vector<ParmValue> candDimVals(candIdx.size() * nDims);
  std::vector<std::unordered_set<ParmValue>> distinct(nDims);
  Configuration conf;
  for (size_t c = 0; c < candIdx.size(); ++c) {
    space_->at(candIdx[c], conf);
    size_t d = 0;
    for (ParmValue v : conf) {
      if (d >= nDims)
        break;
      candDimVals[c * nDims + d] = v;
      distinct[d].insert(v);
      ++d;
    }
  }
  llvm::SmallVector<size_t> dimDistinct(nDims);
  for (size_t d = 0; d < nDims; ++d)
    dimDistinct[d] = std::max<size_t>(distinct[d].size(), 1);

  const arma::mat normEnc = normaliseFeatures(candEncoded);
  const double bandwidth = medianPairDistance(normEnc, rng);

  // The batch the round evaluated. Measuring it is the only way to see whether
  // the acquisition in use actually spreads a batch; the top-q rows below
  // describe the ranking, which is the same object whatever draws from it.
  if (selected.size() >= 2)
    diag.batches.push_back(measureBatch(round, selected, normEnc, bandwidth,
                                        orderMu, orderSigma, candDimVals,
                                        dimDistinct, rng, /*selected=*/true));

  for (size_t q : opts.diagBatchSizes) {
    if (q < 2 || q > order.n_elem)
      continue;
    llvm::ArrayRef<arma::uword> topQ(order.memptr(), q);
    diag.batches.push_back(measureBatch(round, topQ, normEnc, bandwidth,
                                        orderMu, orderSigma, candDimVals,
                                        dimDistinct, rng, /*selected=*/false));
  }
}

void SearchDiagnostics::dumpRoundsCSV(std::filesystem::path path) const {
  if (rounds.empty())
    return;
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out)
    return;
  // One row per selected candidate. The round-level columns repeat across a
  // batch's rows, so anything summed over rounds (the fit time, the batch's
  // elapsed evaluation) must be taken per distinct round, not per row.
  out << "round,n_obs,n_cand,n_neighbor,batch_size,batch_pos,"
         "sel_idx,sel_mu,sel_sigma,rank_acq,rank_mu,rank_sigma,"
         "from_neighbor,accepted,fit_ms,predict_ms,accept_ms\n";
  for (const auto &r : rounds) {
    auto roundCols = [&]() {
      out << r.round << "," << r.nObs << "," << r.nCand << "," << r.nNeighbor
          << "," << r.selections.size() << ",";
    };
    // A round where every candidate was rejected has no selection to report,
    // but its fit time still counts towards the search's wall clock.
    if (r.selections.empty()) {
      roundCols();
      out << ",,,,,,,," << r.fitMs << "," << r.predictMs << "," << r.acceptMs
          << "\n";
      continue;
    }
    for (size_t i = 0; i < r.selections.size(); ++i) {
      const SelectionRecord &s = r.selections[i];
      roundCols();
      out << i << "," << s.idx << "," << s.mu << "," << s.sigma << ","
          << s.rankAcq << "," << s.rankMu << "," << s.rankSigma << ","
          << (s.fromNeighbor ? 1 : 0) << "," << (s.accepted ? 1 : 0) << ","
          << r.fitMs << "," << r.predictMs << "," << r.acceptMs << "\n";
    }
  }
}

void SearchDiagnostics::dumpBatchesCSV(const ConfigSpace &space,
                                       std::filesystem::path path) const {
  if (batches.empty())
    return;
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out)
    return;
  out << "round,q,selected,q_eff,q_eff_ref,mean_pdist,dispersion,overlap_mu,"
         "overlap_sigma";
  for (size_t d = 0; d < space.numDims(); ++d)
    out << ",ent_" << space.dimName(d);
  out << "\n";
  for (const auto &b : batches) {
    out << b.round << "," << b.q << "," << (b.selected ? 1 : 0) << "," << b.qEff
        << "," << b.qEffRef << "," << b.meanPdist << "," << b.dispersion << ","
        << b.overlapMu << "," << b.overlapSigma;
    for (double e : b.dimEntropy)
      out << "," << e;
    out << "\n";
  }
}

// ===----------------------------------------------------------------------===//
// Next-candidate selection
// ===----------------------------------------------------------------------===//

bool CandidatePool::tryInsert(std::unordered_set<size_t> &result, size_t idx) {
  if (isVisited(idx))
    return false;
  auto res = result.insert(idx);
  return res.second;
}

void CandidatePool::fillRandom(std::unordered_set<size_t> &result,
                               size_t target, std::mt19937 &rng) {
  if (result.size() >= target || empty())
    return;
  size_t numVisited = this->numVisited();
  auto dist = std::uniform_int_distribution<size_t>(0, N - 1);

  size_t numAttempts = 0;
  while (result.size() < std::min(target, N - numVisited) &&
         numAttempts++ <= target * 5) {
    tryInsert(result, dist(rng));
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

size_t BananasStrategy::step(std::mt19937 &rng,
                             const std::function<bool(size_t)> &accept,
                             int round, size_t nObsAtRound, size_t batchSize,
                             unsigned workers) {
  using Clock = std::chrono::steady_clock;
  auto elapsedMs = [](Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since)
        .count();
  };
  const bool wantDiag = !opts.dumpDir.empty();

  // --- Build candidate set ---
  // Start with the grid-neighbours of every already-observed configuration.
  // Neighbours differ in exactly one dimension by one discrete step, so they
  // are the most likely region to contain a better point.
  std::unordered_set<size_t> candSet;
  pool.fillNeighbors(candSet, opts.neighborDepth);
  // Everything in the set at this point came from the neighbourhood, and
  // fillRandom dedups against the same set, so a snapshot here is what
  // separates the two provenances afterwards.
  std::unordered_set<size_t> neighborCands;
  if (wantDiag)
    neighborCands = candSet;
  const size_t nNeighborCands = candSet.size();
  // Add random candidates: on top of the neighbours when nRandCandidates
  // says so, otherwise only up to the nCandidates total -- which adds none
  // once the neighbour set alone exceeds it.
  pool.fillRandom(candSet,
                  opts.nRandCandidates > 0
                      ? candSet.size() + opts.nRandCandidates
                      : opts.nCandidates,
                  rng);

  if (candSet.empty())
    return 0;

  std::vector<size_t> candIdx(candSet.begin(), candSet.end());
  arma::mat candEncoded = encodeSubset(*space_, candIdx);

  // --- Fit surrogate and rank candidates ---
  // Build compact training matrices excluding INF-cost observations.
  // Training the MLP on INF targets causes gradient explosion → NaN weights
  // → NaN predictions → arma::sort_index abort.
  arma::uvec finiteCols(pool.nObs);
  arma::uword nFinite = 0;
  for (arma::uword i = 0; i < static_cast<arma::uword>(pool.nObs); ++i)
    if (std::isfinite(pool.yo(0, i)))
      finiteCols(nFinite++) = i;
  finiteCols.resize(nFinite);

  if (nFinite < 2) {
    // Not enough finite observations to train; pick the first accepted
    // candidate.
    for (size_t idx : candIdx)
      if (accept(idx))
        return 1;
    return 0;
  }

  arma::mat Xo_obs = pool.Xo.cols(finiteCols);
  arma::mat yo_obs = pool.yo.cols(finiteCols);

  // Warm-start: reuse weights from the previous iteration. Reinitialise only
  // when the ensemble doesn't exist yet or its configuration has changed.
  if (!ensemble_ ||
      ensemble_->models.size() != static_cast<size_t>(opts.nEnsemble))
    ensemble_ = std::make_unique<BananasEnsemble>(opts.nEnsemble, opts.hidden,
                                                  opts.depth);
  ensemble_->scale_ = opts.objectiveScale;
  auto tFit = Clock::now();
  ensemble_->fit(Xo_obs, yo_obs, opts.epochs);
  const double fitMs = elapsedMs(tFit);

  auto tPredict = Clock::now();
  auto [mu, sigma] = ensemble_->predict(candEncoded);
  const double predictMs = elapsedMs(tPredict);

  // validation logging
  recordValidationData(trainingValidSet, ensemble_.get(), nObsAtRound);

  if (opts.validationInterval > 0 && round % opts.validationInterval == 0) {
    recordValidationData(validSet, ensemble_.get(), nObsAtRound);
  }

  arma::rowvec scores = computeAcq(mu, sigma, opts.kappa);
  // Guard: replace any NaN/Inf from MLP instability with +inf so they
  // land at the end of the sorted order and are never preferred.
  scores.for_each([](double &x) {
    if (!std::isfinite(x))
      x = arma::datum::inf;
  });
  arma::uvec order = arma::sort_index(scores, "ascend");

  // The two degenerate rankings the acquisition interpolates between: the mean
  // alone is pure exploitation, the spread alone pure exploration. Where a
  // selection sits between them is what the diagnostics report.
  arma::rowvec muGuard = mu, sigmaGuard = sigma;
  muGuard.for_each([](double &x) {
    if (!std::isfinite(x))
      x = arma::datum::inf;
  });
  sigmaGuard.for_each([](double &x) {
    if (!std::isfinite(x))
      x = 0.0;
  });
  arma::uvec orderMu = arma::sort_index(muGuard, "ascend");
  arma::uvec orderSigma = arma::sort_index(sigmaGuard, "descend");

  llvm::SmallVector<arma::uword> batch =
      selectBatch(opts, mu, sigma, order, std::max<size_t>(batchSize, 1), rng);

  if (wantDiag)
    recordRoundDiagnostics(rng, round, nObsAtRound, candIdx, candEncoded,
                           nNeighborCands, order, orderMu, orderSigma, batch);

  // Describe the batch before evaluating it: `accepted` is filled in by
  // evaluateBatch, the rest is what the acquisition saw when it chose.
  std::vector<SelectionRecord> selections;
  if (wantDiag) {
    selections.reserve(batch.size());
    for (arma::uword pos : batch)
      selections.push_back({candIdx[pos], mu(pos), sigma(pos),
                            rankOf(order, pos), rankOf(orderMu, pos),
                            rankOf(orderSigma, pos),
                            neighborCands.count(candIdx[pos]) > 0, false});
  }

  auto tAccept = Clock::now();
  size_t accepted = evaluateBatch(batch, candIdx, accept, workers, selections);

  // The whole batch was rejected -- every candidate failed to lower. Walking
  // the rest of the ranking for one usable point keeps that from ending the
  // search, and is the entire behaviour when batchSize is 1.
  if (accepted == 0) {
    std::unordered_set<arma::uword> tried(batch.begin(), batch.end());
    for (size_t i = 0; i < order.n_elem && accepted == 0; ++i) {
      const arma::uword pos = order(i);
      if (tried.count(pos) || !accept(candIdx[pos]))
        continue;
      accepted = 1;
      if (wantDiag)
        selections.push_back({candIdx[pos], mu(pos), sigma(pos), i,
                              rankOf(orderMu, pos), rankOf(orderSigma, pos),
                              neighborCands.count(candIdx[pos]) > 0, true});
    }
  }

  if (wantDiag) {
    RoundRecord &rec = diag.rounds.back();
    rec.fitMs = fitMs;
    rec.predictMs = predictMs;
    rec.acceptMs = elapsedMs(tAccept);
    rec.selections = std::move(selections);
  }
  return accepted;
}

// ===----------------------------------------------------------------------===//
// CSV dump
// ===----------------------------------------------------------------------===//

void CandidatePool::dumpToCSV(const ConfigSpace &space,
                              const InferenceOptions &opts,
                              std::filesystem::path path,
                              const SearchStrategy *strategy) const {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out)
    return;

  // Use the strategy's warm-started model to get per-candidate statistics.
  const bool hasModel =
      strategy && strategy->hasModel() && nObs >= 2 && !empty();
  // Per-index predictions; indexed by position in the dumped rows.
  arma::rowvec mu_v, sigma_v, acq_v;
  if (hasModel) {
    const size_t D = space_->numFeatures();
    const size_t confsToEncode =
        opts.dumpFullPool ? this->size() : visited.size();
    arma::mat encoded(D, confsToEncode);
    size_t ix = 0;
    Configuration conf;
    llvm::SmallVector<double, 16> features;
    for (size_t i = 0; i < N; ++i) {
      if (!isVisited(i))
        continue;
      space_->at(i, conf);
      features.clear();
      space_->encode(conf, features);
      for (size_t d = 0; d < D; ++d)
        encoded(d, ix) = features[d];
      ix++;
    }
    strategy->predict(encoded, mu_v, sigma_v);
    acq_v = computeAcq(mu_v, sigma_v, opts.kappa);
  }

  // Header. One column per *dimension*, since that is what a row holds: a
  // parameter spanning several contributes one column each, named for the
  // entry rather than for the parameter.
  for (size_t d = 0; d < space.numDims(); ++d)
    out << space.dimName(d) << ",";
  out << "visited,valid,cost,eval_iter,eval_time_ms,cpu_time_ms";
  if (hasModel)
    out << ",mu,sigma,acq";
  out << "\n";

  // One row per pool member, in flat-index order. The `valid` column is always
  // 1: a space holds nothing else. It is kept because the analysis scripts
  // select on it.
  size_t j = 0;
  Configuration conf;
  for (size_t i = 0; i < N; ++i) {
    if (!opts.dumpFullPool && !isVisited(i))
      continue;
    space.at(i, conf);

    for (ParmValue v : conf)
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
  }
}

void CandidatePool::dumpMetadataJSON(const ConfigSpace &space,
                                     std::filesystem::path path) const {
  dumpSpaceJSON(space, N, path);
}

/// Past this many items a permutation's orderings table is omitted (n! rows);
/// the per-dimension encoding documentation still tells a reader how to write
/// one by hand. Ops here have 2-4 iteration dims, so the cap is generous.
static constexpr unsigned kMaxListedPermutationItems = 5;

void dumpSpaceJSON(const ConfigSpace &space, size_t feasibleSize,
                   std::filesystem::path path) {
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

  // Two sizes, because "total" is ambiguous: the Cartesian product of the
  // declared domains, and what the constraints leave of it. Their ratio is the
  // density.
  double cartesian = 1;
  for (const SearchParam &param : space.params)
    cartesian *= param.numValues();

  out << "{\n";
  out << "  \"cartesian_size\": " << cartesian << ",\n";
  out << "  \"feasible_size\": " << feasibleSize << ",\n";
  out << "  \"space_feasible_density\": "
      << (static_cast<double>(feasibleSize) / cartesian) << ",\n";
  // The offline setup cost of the search: declaring, solving and
  // enumerating the space. Reported here because no other artifact of a run
  // records it, and walltime comparisons must not credit it for free.
  out << "  \"space_build_seconds\": " << space.buildWallSeconds << ",\n";
  // Whatever the builder recorded about how it planned the space -- which
  // parameters it enumerated jointly, and where each constraint ended up. The
  // sizes above are the outcome of those decisions and do not explain them.
  if (space.metadata)
    space.metadata->printJSONMembers(out);

  out << "  \"params\": [\n";
  for (size_t i = 0; i < space.params.size(); ++i) {
    const auto &p = space.params[i];
    out << "    {";
    out << "\"name\": ";
    jsonStr(p.name);
    out << ", ";
    if (!p.doc.empty()) {
      out << "\"doc\": ";
      jsonStr(p.doc);
      out << ", ";
    }
    // Both, because they differ for anything of arity > 1 and the difference
    // is the point: `cardinality` is one dimension's domain, `num_values` is
    // how many values the parameter has (n vs n! for an ordering of n items).
    out << "\"cardinality\": " << p.cardinality() << ", ";
    out << "\"num_values\": " << p.numValues() << ", ";
    out << "\"arity\": " << p.arity() << ", ";
    out << "\"kind\": ";
    jsonStr(paramKindName(p.kind()).str());
    out << ", ";
    if (p.kind() == ParamKind::Integer) {
      // How the surrogate reads a distance between two of these values, which
      // is a declared property and not one recoverable from the domain below:
      // a "range" spelling can be multiplicative and a "values" one linear.
      out << "\"spacing\": ";
      jsonStr(p.spacing == Spacing::Multiplicative ? "multiplicative"
                                                   : "linear");
      out << ", ";
    }
    // The dimension names as pool.csv columns and eval-solution spell them.
    // Redundant for arity 1 (it is the parameter's own name) but load-bearing
    // for permutations, whose n dimensions are the parameter's contract with
    // eval-solution and are otherwise documented nowhere outside the C++.
    if (p.arity() > 1) {
      out << "\"dims\": [";
      for (size_t k = 0; k < p.arity(); ++k) {
        if (k)
          out << ", ";
        jsonStr(p.dimName(k));
      }
      out << "], ";
    }
    if (p.kind() == ParamKind::Permutation) {
      // Everything a transcriber needs to write an ordering by hand, so the
      // encoding (ParmKind<Permutation>: dimension k holds the 1-BASED place
      // of item k) never has to be reverse-engineered from ConfigSpace.cpp.
      out << "\"items\": [";
      for (unsigned k = 0; k < p.permutationSize; ++k) {
        if (k)
          out << ", ";
        jsonStr(k < p.itemLabels.size() ? p.itemLabels[k]
                                        : "item" + std::to_string(k));
      }
      out << "], ";
      out << "\"encoding\": \"dimension k holds the 1-based place of item k; "
             "place 1 is the outermost workgroup axis (slowest-varying "
             "across leaves, cnm.workgroup_dim_order position 0). Items "
             "declared conditionally active take the low places when active; "
             "inactive items are forced to the remaining high places in "
             "item-index order.\", ";
      if (p.permutationSize <= kMaxListedPermutationItems) {
        // All n! orderings, each with its exact eval-solution assignment --
        // copy-pasteable, so transcription cannot mis-encode an order.
        out << "\"orderings\": [\n";
        llvm::SmallVector<unsigned> byPlace(p.permutationSize);
        for (unsigned k = 0; k < p.permutationSize; ++k)
          byPlace[k] = k; // byPlace[place] = item
        auto label = [&](unsigned item) {
          return item < p.itemLabels.size() ? p.itemLabels[item]
                                            : "item" + std::to_string(item);
        };
        bool firstRow = true;
        do {
          out << (firstRow ? "" : ",\n") << "      {\"order\": ";
          firstRow = false;
          std::string pretty;
          for (unsigned place = 0; place < p.permutationSize; ++place) {
            if (place)
              pretty += ">";
            pretty += label(byPlace[place]);
          }
          jsonStr(pretty);
          out << ", \"assignment\": {";
          // Invert: dimension (item) k gets place-of-k + 1.
          for (unsigned item = 0; item < p.permutationSize; ++item) {
            unsigned place = 0;
            while (byPlace[place] != item)
              ++place;
            if (item)
              out << ", ";
            jsonStr(p.dimName(item));
            out << ": " << (place + 1);
          }
          out << "}}";
        } while (std::next_permutation(byPlace.begin(), byPlace.end()));
        out << "\n    ], ";
      }
    }
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
  const size_t D = space_->numFeatures();
  Configuration conf;
  space_->at(idx, conf);
  llvm::SmallVector<double, 16> features;
  space_->encode(conf, features);
  if (encoded.is_empty())
    encoded.set_size(D, 0);
  encoded.insert_cols(encoded.n_cols, 1);
  for (size_t d = 0; d < D; ++d)
    encoded(d, encoded.n_cols - 1) = features[d];
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
      for (ParmValue v : conf)
        out << v << ",";
      out << trueCosts[j] << "," << snap.iter << "," << snap.mu(j) << ","
          << snap.sigma(j) << "\n";
    }
  }
}

// ===----------------------------------------------------------------------===//
// Random strategy
// ===----------------------------------------------------------------------===//

/// Uniform random search: each round draws `batchSize` unvisited configs
/// uniformly and evaluates them. No model, no candidate set -- the whole
/// budget goes into evaluations. The floor every learned strategy has to
/// beat at equal budget (see docs/SearchStrategyPlan.md).
class RandomStrategy final : public SearchStrategy {
public:
  explicit RandomStrategy(CandidatePool &pool) : pool(pool) {}

  size_t step(std::mt19937 &rng, const std::function<bool(size_t)> &accept,
              int round, size_t nObsAtRound, size_t batchSize,
              unsigned workers) override {
    // Keep drawing until something is accepted: a returned 0 ends the whole
    // search (the driver's stop condition), which a failed draw does not
    // justify here. Termination: every drawn config is marked visited whether
    // its evaluation succeeds or not, so each pass shrinks the unvisited set.
    while (pool.numVisited() < pool.size()) {
      std::unordered_set<size_t> draw;
      pool.fillRandom(draw, batchSize, rng);
      if (draw.empty()) {
        // Rejection sampling gave up (near-exhausted pool); fall back to a
        // scan so the search visits the stragglers rather than stopping.
        size_t idx = pool.firstUnvisited();
        if (idx == pool.N)
          break;
        draw.insert(idx);
      }
      llvm::SmallVector<size_t> candIdx(draw.begin(), draw.end());
      size_t accepted = evaluateIndices(candIdx, accept, workers);
      if (accepted > 0)
        return accepted;
    }
    return 0;
  }

private:
  CandidatePool &pool;
};

// ===----------------------------------------------------------------------===//
// Descent strategy
// ===----------------------------------------------------------------------===//

/// Random-restart neighbourhood descent. An incumbent's unvisited grid
/// neighbours are evaluated batch by batch; the climb moves to the best
/// improving neighbour, and restarts from a random unvisited config once the
/// neighbourhood is exhausted without improvement. The first climb starts
/// from the best init observation, so the shared init sample seeds the
/// descent exactly as it seeds the surrogate strategies.
class DescentStrategy final : public SearchStrategy {
public:
  explicit DescentStrategy(CandidatePool &pool)
      : pool(pool), space_(pool.space_) {}

  size_t step(std::mt19937 &rng, const std::function<bool(size_t)> &accept,
              int round, size_t nObsAtRound, size_t batchSize,
              unsigned workers) override {
    while (pool.numVisited() < pool.size()) {
      if (cur == kNone && !startClimb(rng, accept))
        continue; // restart evaluation failed; its config is now visited
      if (restartAccepted) {
        // The restart's own evaluation is this round's progress; the climb
        // out of it begins next round.
        restartAccepted = false;
        return 1;
      }
      if (frontier.empty()) {
        // Neighbourhood exhausted without improvement: local optimum.
        cur = kNone;
        continue;
      }
      llvm::SmallVector<size_t> batch;
      while (!frontier.empty() && batch.size() < batchSize) {
        size_t idx = frontier.back();
        frontier.pop_back();
        // The frontier is not re-checked on refill, and an index can be a
        // neighbour of several incumbents.
        if (!pool.isVisited(idx))
          batch.push_back(idx);
      }
      if (batch.empty())
        continue;
      size_t accepted = evaluateIndices(batch, accept, workers);
      // Steepest of the batch: move only if something improved.
      size_t bestIdx = kNone;
      double bestCost = curCost;
      for (size_t idx : batch) {
        auto it = pool.costByIdx.find(idx);
        if (it != pool.costByIdx.end() && it->second < bestCost) {
          bestIdx = it->first;
          bestCost = it->second;
        }
      }
      if (bestIdx != kNone) {
        cur = bestIdx;
        curCost = bestCost;
        refillFrontier(rng);
      }
      if (accepted > 0)
        return accepted;
      // Every evaluation failed; failures cost no budget, keep climbing.
    }
    return 0;
  }

private:
  static constexpr size_t kNone = std::numeric_limits<size_t>::max();

  /// Set `cur` to a climb start: the best observation so far on the first
  /// climb (free -- it is already evaluated), a random unvisited config
  /// afterwards (evaluated here; `restartAccepted` tells step the round spent
  /// budget on it). Returns false when the chosen start failed to evaluate or
  /// the pool is exhausted.
  bool startClimb(std::mt19937 &rng,
                  const std::function<bool(size_t)> &accept) {
    if (firstClimb) {
      firstClimb = false;
      for (auto [idx, cost] : pool.costByIdx)
        if (cost < curCost) {
          cur = idx;
          curCost = cost;
        }
      if (cur != kNone) {
        refillFrontier(rng);
        return true;
      }
    }
    std::unordered_set<size_t> draw;
    pool.fillRandom(draw, 1, rng);
    size_t idx = draw.empty() ? pool.firstUnvisited() : *draw.begin();
    if (idx >= pool.N)
      return false;
    if (!accept(idx))
      return false;
    cur = idx;
    curCost = pool.costByIdx.at(idx);
    restartAccepted = true;
    refillFrontier(rng);
    return true;
  }

  void refillFrontier(std::mt19937 &rng) {
    frontier.clear();
    llvm::SmallVector<size_t> nbrs;
    space_->neighborIndices(cur, nbrs);
    for (size_t nb : nbrs)
      if (!pool.isVisited(nb))
        frontier.push_back(nb);
    // Shuffled so that a batch that stops mid-neighbourhood is not biased
    // towards whichever dimension neighborIndices enumerates first.
    std::shuffle(frontier.begin(), frontier.end(), rng);
  }

  CandidatePool &pool;
  const ConfigSpace *space_;
  size_t cur = kNone;
  double curCost = std::numeric_limits<double>::infinity();
  /// Unvisited neighbours of `cur` not yet offered, shuffled.
  std::vector<size_t> frontier;
  bool firstClimb = true;
  bool restartAccepted = false;
};

// ===----------------------------------------------------------------------===//
// GA strategy
// ===----------------------------------------------------------------------===//

/// Steady-state genetic algorithm: tournament-select two parents from a
/// population of the best observed configs, cross them per *parameter* (so a
/// permutation parameter's dimensions never mix between parents), mutate by
/// grid-neighbour steps, and let evaluated children compete into the
/// population by cost. The population is seeded from the shared init sample.
class GaStrategy final : public SearchStrategy {
public:
  explicit GaStrategy(CandidatePool &pool)
      : pool(pool), space_(pool.space_),
        popSize(std::max<size_t>(16, static_cast<size_t>(pool.opts.nInit))) {}

  size_t step(std::mt19937 &rng, const std::function<bool(size_t)> &accept,
              int round, size_t nObsAtRound, size_t batchSize,
              unsigned workers) override {
    if (population.empty())
      seedPopulation();
    while (pool.numVisited() < pool.size()) {
      llvm::SmallVector<size_t> batch;
      std::unordered_set<size_t> proposed;
      for (size_t guard = 0; batch.size() < batchSize && guard < batchSize * 20;
           ++guard) {
        size_t child = proposeChild(rng);
        if (child != kNone && !pool.isVisited(child) &&
            proposed.insert(child).second)
          batch.push_back(child);
      }
      if (batch.empty()) {
        // Converged onto visited ground: random immigrants keep the search
        // alive and reintroduce diversity.
        std::unordered_set<size_t> draw;
        pool.fillRandom(draw, batchSize, rng);
        if (draw.empty())
          return 0;
        batch.assign(draw.begin(), draw.end());
      }
      size_t accepted = evaluateIndices(batch, accept, workers);
      for (size_t idx : batch)
        offerToPopulation(idx);
      if (accepted > 0)
        return accepted;
    }
    return 0;
  }

private:
  static constexpr size_t kNone = std::numeric_limits<size_t>::max();

  double costOf(size_t idx) const {
    auto it = pool.costByIdx.find(idx);
    return it == pool.costByIdx.end() ? std::numeric_limits<double>::infinity()
                                      : it->second;
  }

  void seedPopulation() {
    for (auto [idx, cost] : pool.costByIdx)
      if (std::isfinite(cost))
        population.push_back(idx);
    llvm::sort(population,
               [&](size_t a, size_t b) { return costOf(a) < costOf(b); });
    if (population.size() > popSize)
      population.resize(popSize);
  }

  /// Binary tournament over population slots; kNone when the population is
  /// too small to breed.
  size_t tournament(std::mt19937 &rng) const {
    if (population.size() < 2)
      return kNone;
    std::uniform_int_distribution<size_t> pick(0, population.size() - 1);
    size_t a = population[pick(rng)], b = population[pick(rng)];
    return costOf(a) <= costOf(b) ? a : b;
  }

  size_t proposeChild(std::mt19937 &rng) {
    size_t pa = tournament(rng), pb = tournament(rng);
    if (pa == kNone || pb == kNone)
      return kNone;
    Configuration a, b;
    space_->at(pa, a);
    space_->at(pb, b);
    // Uniform per-parameter crossover. Copying whole parameters keeps
    // multi-dimension parameters internally consistent; the result can still
    // violate cross-parameter constraints, in which case the child falls back
    // to a parent and mutation must move it.
    std::bernoulli_distribution coin(0.5);
    Configuration child = a;
    for (size_t p = 0; p < space_->numParams(); ++p)
      if (coin(rng)) {
        auto vals = space_->paramValues(b, p);
        std::copy(vals.begin(), vals.end(),
                  child.begin() + space_->dimOffset(p));
      }
    const bool feasible = space_->isEncodable(child);
    size_t idx = feasible ? space_->indexOf(child) : pa;
    // Mutation by neighbour steps -- neighbours never leave the constrained
    // space. Forced when the child is a fallback parent (already visited) so
    // the proposal is never a wasted duplicate.
    unsigned steps = (!feasible || pool.isVisited(idx)) ? 1 : 0;
    if (coin(rng))
      ++steps;
    llvm::SmallVector<size_t> nbrs;
    for (unsigned m = 0; m < steps; ++m) {
      nbrs.clear();
      space_->neighborIndices(idx, nbrs);
      if (nbrs.empty())
        break;
      std::uniform_int_distribution<size_t> pick(0, nbrs.size() - 1);
      idx = nbrs[pick(rng)];
    }
    return idx;
  }

  /// Insert an evaluated config into the population if it beats the worst
  /// member (or the population is not full yet). Failed evaluations have no
  /// cost and are never inserted.
  void offerToPopulation(size_t idx) {
    double cost = costOf(idx);
    if (!std::isfinite(cost))
      return;
    if (population.size() < popSize) {
      population.push_back(idx);
      return;
    }
    size_t worstSlot = 0;
    for (size_t s = 1; s < population.size(); ++s)
      if (costOf(population[s]) > costOf(population[worstSlot]))
        worstSlot = s;
    if (cost < costOf(population[worstSlot]))
      population[worstSlot] = idx;
  }

  CandidatePool &pool;
  const ConfigSpace *space_;
  size_t popSize;
  std::vector<size_t> population; // pool indices of evaluated configs
};

// ===----------------------------------------------------------------------===//
// SearchStrategy factory
// ===----------------------------------------------------------------------===//

SearchStrategy::~SearchStrategy() = default;

std::unique_ptr<SearchStrategy> makeSearchStrategy(CandidatePool &pool,
                                                   ValidationSet &validSet,
                                                   ValidationSet &trainingSet) {
  switch (pool.opts.searchStrategy) {
  case InferenceOptions::SearchStrategyKind::Bananas:
    return std::make_unique<BananasStrategy>(pool, validSet, trainingSet);
  case InferenceOptions::SearchStrategyKind::Random:
    return std::make_unique<RandomStrategy>(pool);
  case InferenceOptions::SearchStrategyKind::Descent:
    return std::make_unique<DescentStrategy>(pool);
  case InferenceOptions::SearchStrategyKind::Ga:
    return std::make_unique<GaStrategy>(pool);
  }
  llvm_unreachable("unknown search strategy");
}

} // namespace mlir::cinm
