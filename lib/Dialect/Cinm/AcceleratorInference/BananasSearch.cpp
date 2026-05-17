#include "BananasSearch.h"
#include "cinm-mlir/Dialect/Cinm/AcceleratorInference/AcceleratorInference.h"

#include <mutex>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mlir::cinm {

// ===----------------------------------------------------------------------===//
// Python interpreter lifecycle
// ===----------------------------------------------------------------------===//

// Initialise Python exactly once per process, and prepend BANANAS_BRIDGE_DIR
// (injected by CMake) to sys.path so `import bo_bridge` resolves.
static void ensurePython() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    if (!Py_IsInitialized())
      py::initialize_interpreter();
    py::module_::import("sys")
        .attr("path")
        .attr("insert")(0, BANANAS_BRIDGE_DIR);
  });
}

static py::module_ getBoBridge() {
  ensurePython();
  return py::module_::import("bo_bridge");
}

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
                                     size_t N, size_t D, size_t n, int seed) {
  py::gil_scoped_acquire gil;
  auto bridge = getBoBridge();

  // Wrap the flat buffer as a 2-D numpy array — zero-copy read-only view.
  py::array_t<float> arr(
      {static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(D)},
      encodedFlat.data());

  auto result = bridge.attr("lhs_indices")(arr, static_cast<int>(n), seed)
                    .cast<std::vector<size_t>>();
  return {result.begin(), result.end()};
}

// ===----------------------------------------------------------------------===//
// Surrogate-guided next batch
// ===----------------------------------------------------------------------===//

llvm::SmallVector<size_t>
nextCandidateIndices(const std::vector<float> &X_obs, size_t nObs,
                     const std::vector<float> &y_obs,
                     const std::vector<float> &X_pool, size_t nPool, size_t D,
                     int k, float kappa, int epochs, int nEnsemble, int hidden,
                     int depth) {
  py::gil_scoped_acquire gil;
  auto bridge = getBoBridge();

  py::array_t<float> xObsArr(
      {static_cast<py::ssize_t>(nObs), static_cast<py::ssize_t>(D)},
      X_obs.data());
  py::array_t<float> yObsArr(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(nObs)}, y_obs.data());
  py::array_t<float> xPoolArr(
      {static_cast<py::ssize_t>(nPool), static_cast<py::ssize_t>(D)},
      X_pool.data());

  auto result = bridge.attr("next_candidate_indices")(xObsArr, yObsArr,
                                                      xPoolArr, k, kappa,
                                                      epochs, nEnsemble,
                                                      hidden, depth)
                    .cast<std::vector<size_t>>();
  return {result.begin(), result.end()};
}

} // namespace mlir::cinm
