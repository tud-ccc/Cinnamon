#pragma once

#include <cstddef>
#include <vector>

#include <llvm/ADT/SmallVector.h>

namespace mlir::cinm {

struct ConfigSpace;
using Configuration = llvm::SmallVector<int64_t>;

/// Encode a configuration as a float feature vector.
/// Each parameter is normalised to [0, 1] using the parameter's domain bounds.
std::vector<float> encodeConfig(const ConfigSpace &space,
                                const Configuration &config);

/// Select *n* row-indices from the candidate matrix (row-major, N×D floats)
/// using Latin Hypercube Sampling in the encoded feature space.
llvm::SmallVector<size_t> lhsIndices(const std::vector<float> &encodedFlat,
                                     size_t N, size_t D, size_t n,
                                     unsigned seed = 42);

/// Fit a BANANAS MLP ensemble on (X_obs [nObs×D], y_obs [nObs]) and return
/// the k row-indices into X_pool [nPool×D] with the lowest UCB score.
llvm::SmallVector<size_t>
nextCandidateIndices(const std::vector<float> &X_obs, size_t nObs,
                     const std::vector<float> &y_obs,
                     const std::vector<float> &X_pool, size_t nPool, size_t D,
                     int k = 1, float kappa = 2.0f, int epochs = 200,
                     int nEnsemble = 5, int hidden = 64, int depth = 2);

} // namespace mlir::cinm
