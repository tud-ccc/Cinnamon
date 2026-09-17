// Llama-2 110M (the stories110M configuration), W8A8, one decode step: the
// dimensions of llama2_110M_w8a8.mlir; the driver itself is
// llama2_w8a8_driver.hpp.

#include <cstddef>

namespace {
constexpr size_t H = 768, F = 2048, L = 6, A = 6, V = 32000, N = 1024;
// Classifier rows as stored: the vocabulary padded to a power of two.
constexpr size_t VPAD = 32768;
} // namespace

#include "llama2_w8a8_driver.hpp"
