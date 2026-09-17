// OpenLLaMA 3B v2, W8A8, one decode step: the dimensions of
// llama_3B_w8a8.mlir; the driver itself is llama2_w8a8_driver.hpp. The
// operands come to 3.6 GB.

#include <cstddef>

namespace {
constexpr size_t H = 3200, F = 8640, L = 26, A = 32, V = 32000, N = 1024;
// Classifier rows as stored: the vocabulary padded to a power of two.
constexpr size_t VPAD = 32768;
} // namespace

#include "llama2_w8a8_driver.hpp"
