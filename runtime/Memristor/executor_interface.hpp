#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace memristor_runtime {
template <size_t Dimension> struct memref_descriptor {
  void *base;
  void *data;
  int64_t offset;
  std::array<int64_t, Dimension> sizes;
  std::array<int64_t, Dimension> strides;

  template <size_t D>
    requires(D > Dimension)
  operator memref_descriptor<D>() {
    memref_descriptor<D> result;
    result.base = base;
    result.data = data;
    result.offset = offset;

    size_t dimension_padding = D - Dimension;
    for (size_t i = 0; i < D; i++) {
      result.sizes[i] =
          i < dimension_padding ? 1 : sizes[i - dimension_padding];
      result.strides[i] =
          i < dimension_padding ? 1 : strides[i - dimension_padding];
    }
    return result;
  }

  template <typename T> void print() {
    std::cout << "memref [base: " << base << " size: ";
    for (auto size : sizes)
      std::cout << size << " ";
    std::cout << "] {" << std::endl;

    print_dimension<T>(0, 0);

    std::cout << "}" << std::endl;
  }

  template <typename T> void print_dimension(size_t dimension, size_t start) {
    auto *typed_data = (T *)data;

    if (dimension == Dimension - 1) {
      for (int i = 0; i < sizes[dimension]; i++) {
        std::cout << typed_data[start + i * strides[dimension]] << "\t ";
      }
      std::cout << std::endl;
    } else {
      for (int i = 0; i < sizes[dimension]; i++) {
        print_dimension<T>(dimension + 1, offset + i * strides[dimension]);
      }
    }
  }
};

struct crossbar {
  memref_descriptor<2> rhs;
  memref_descriptor<2> lhs;
  memref_descriptor<2> result;
};

extern crossbar crossbars[16];

template <typename T>
void memristor_write_to_crossbar(int32_t crossbar_id, memref_descriptor<2>);

template <typename T>
void memristor_gemm(int32_t crossbar_id, memref_descriptor<2>,
                    memref_descriptor<2>);

template <typename T>
void memristor_gemv(int32_t crossbar_id, memref_descriptor<1>,
                    memref_descriptor<1>);

void memristor_barrier(int32_t crossbar_id);
} // namespace memristor_runtime