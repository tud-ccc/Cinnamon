#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>

namespace memristor_runtime {

// NOTE: This struct is also used for ipc to systemc-rram-wrapper
// If you change it, make sure to update the corresponding struct in the wrapper
struct matrix_descriptor {
  uint8_t bits;
  uint16_t rows;
  uint16_t cols;

  size_t byte_count() const { return rows * cols * storage_bitwidth() / 8; }

  uint8_t storage_bitwidth() const {
    uint8_t storage_bits = bits - 1;
    storage_bits |= storage_bits >> 1;
    storage_bits |= storage_bits >> 2;
    storage_bits |= storage_bits >> 4;
    storage_bits |= storage_bits >> 8;
    storage_bits++;
    return storage_bits;
  }
};
} // namespace memristor_runtime