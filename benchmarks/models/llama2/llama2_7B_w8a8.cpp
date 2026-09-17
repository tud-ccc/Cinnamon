// Llama-2 7B, W8A8, one decode step: the dimensions of llama2_7B_w8a8.mlir;
// the driver itself is llama2_w8a8_driver.hpp. The operands come to 7 GB.

#include <cstddef>

namespace {
constexpr size_t H = 4096, F = 11008, L = 32, A = 32, V = 32000, N = 1024;
// Classifier rows as stored: the vocabulary padded to a power of two.
constexpr size_t VPAD = 32768;
} // namespace

#include "llama2_w8a8_driver.hpp"
