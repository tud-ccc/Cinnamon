#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <llvm/ADT/SmallVector.h>

namespace mlir::cinm {

struct ConfigSpace;
using Configuration = llvm::SmallVector<int64_t>;

/// Encode a configuration as a float feature vector.
/// Each parameter value is normalised to [0, 1] using the parameter's domain
/// bounds so the surrogate model sees a uniform input scale.
std::vector<float> encodeConfig(const ConfigSpace &space,
                                const Configuration &config);

/// Select *n* indices from [0, N) into the candidate matrix (row-major,
/// N rows × D cols) using Latin Hypercube Sampling.  Returns up to n indices.
llvm::SmallVector<size_t> lhsIndices(const std::vector<float> &encodedFlat,
                                     size_t N, size_t D, size_t n,
                                     int seed = 42);

/// Fit a BANANAS MLP ensemble on (X_obs, y_obs) and return the k indices into
/// X_pool (row-major, nPool × D) with the lowest UCB acquisition score.
llvm::SmallVector<size_t>
nextCandidateIndices(const std::vector<float> &X_obs, size_t nObs,
                     const std::vector<float> &y_obs,
                     const std::vector<float> &X_pool, size_t nPool, size_t D,
                     int k = 1, float kappa = 2.0f, int epochs = 200,
                     int nEnsemble = 5, int hidden = 64, int depth = 2);

} // namespace mlir::cinm
