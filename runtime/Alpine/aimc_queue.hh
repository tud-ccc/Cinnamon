/* 
 * Copyright EPFL 2021
 * Joshua Klein
 * 
 * This file contains functions for queueing larger data structures.
 *
 */

#ifndef __AIMC_QUEUE_HH__
#define __AIMC_QUEUE_HH__

#include <stdint.h>

// int8_t vector queueing.
inline void
queueVector(int size, int8_t * v, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
    for (int i = 0; i < size; i++) {
        aimcQueue(v[i]);
    }
#else // (LOOSELY_COUPLED_MMIO)
    uint32_t tmp = 0;

    // Pack 4 int8 values into tmp, and queue into AIMC input memory.
    for (int i = 0; i < size; i++) {
        if ((i % 4 == 0) && i > 0) {
            aimcQueue(tmp, tid);
            tmp = 0;
        }

        tmp |= (v[i] & 0xff) << (8 * (i % 4));
    }

    // Queue the final pack. If size==0, skip; if size%4==0, this queues the last full pack.
    if (size > 0) {
        aimcQueue(tmp, tid);
    }
#endif // (LOOSELY_COUPLED_MMIO)

    return;
}

// Vector queueing for higher precision, scaled according to max.
template <typename T>
inline void
queueVector(int size, T max, T * v, int tid = 0)
{
#if defined (LOOSELY_COUPLED_MMIO)
    for (int i = 0; i < size; i++) {
        aimcQueue(int8_t(v[i] / max * 127));
    }
#else // (LOOSELY_COUPLED_MMIO)
    uint32_t tmp = 0;

    // Pack 4 int8 values into tmp, and queue into AIMC input memory.
    for (int i = 0; i < size; i++) {
        if ((i % 4 == 0) && i > 0) {
            aimcQueue(tmp, tid);
            tmp = 0;
        }

        tmp |= ((int8_t)(v[i] / max * 127) & 0xff) << (8 * (i % 4));
    }

    // Queue the leftover values, in case |v| % 4 != 0.
    aimcQueue(tmp, tid);
#endif // (LOOSELY_COUPLED_MMIO)

    return;
}

#endif // __AIMC_QUEUE_HH__
