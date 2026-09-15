/*
 * Copyright EPFL 2021
 * Joshua Klein
 *
 * This file contains all of the includes for which to build the AIMC library.
 *
 */

#ifndef __AIMC_HH__
#define __AIMC_HH__

#if defined(USE_CHECKER) // Are we simulating the gem5 code w/ ISA extension?
#include "aimc_check.hh"
#else // (USE_CHECKER)
#include "aimc_intrinsics.hh"
#endif // (USE_CHECKER)

#include "aimc_dequeue.hh"
#include "aimc_queue.hh"
#include "aimc_tile.hh"
#include "alpine_runtime.h"
#include "memref_rt_min.h"

#endif // __AIMC_HH__
