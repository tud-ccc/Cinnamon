#ifndef ALPINE_MEMREF_ABI_H
#define ALPINE_MEMREF_ABI_H

#include <stdint.h>

template <typename T, int Rank> struct StridedMemRefType {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

template <typename T, int Rank>
static inline T &memrefAt(StridedMemRefType<T, Rank> *m,
                          const int64_t idxs[Rank]) {
  int64_t lin = m->offset;
  for (int i = 0; i < Rank; ++i)
    lin += idxs[i] * m->strides[i];
  return m->aligned[lin];
}

template <typename T, int Rank, typename F>
static inline void foreachIndex(StridedMemRefType<T, Rank> *m, F &&f) {
  int64_t idxs[Rank] = {};
  while (true) {
    f(idxs);
    int d = Rank - 1;
    while (d >= 0) {
      ++idxs[d];
      if (idxs[d] < m->sizes[d])
        break;
      idxs[d] = 0;
      --d;
    }
    if (d < 0)
      break;
  }
}

#endif
