/*
 * Copyright EPFL 2021
 * Joshua Klein
 *
 * This file contains functions for dequeueing larger data structures.
 *
 * TODO: Add in activation functions for optimized dequeue utilization.
 *
 */

#ifndef __AIMC_DEQUEUE_HH__
#define __AIMC_DEQUEUE_HH__

#include <stdint.h>

// int8_t vector dequeueing into array.
inline void dequeueVector(int size, int8_t *v, int tid = 0) {
#if defined(LOOSELY_COUPLED_MMIO)
  for (int i = 0; i < size; i++) {
    v[i] = aimcDequeue();
  }
#else  // (LOOSELY_COUPLED_MMIO)
  uint32_t tmp;
  uint32_t *ptmp;

  for (int i = 0; i < size; i++) {
    if (i % 4 == 0) {
      tmp = aimcDequeue(tid);
      ptmp = &tmp;
    }

    v[i] = *ptmp;
    *ptmp >>= 8;
  }
#endif // (LOOSELY_COUPLED_MMIO)

  return;
}

// Vector dequeueing for higher precision, scaled according to max.
template <typename T>
inline void dequeueVector(int size, T max, T *v, int tid = 0) {
#if defined(LOOSELY_COUPLED_MMIO)
  for (int i = 0; i < size; i++) {
    v[i] = (aimcDequeue()) / 127 * max;
  }
#else  // (LOOSELY_COUPLED_MMIO)
  uint32_t tmp;
  uint32_t *ptmp;

  for (int i = 0; i < size; i++) {
    if (i % 4 == 0) {
      tmp = aimcDequeue(tid);
      ptmp = &tmp;
    }

    v[i] = (T)(*ptmp) / 127 * max;
    *ptmp >>= 8;
  }
#endif // (LOOSELY_COUPLED_MMIO)

  return;
}

#endif // __AIMC_DEQUEUE_HH__
