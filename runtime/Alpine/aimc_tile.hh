/*
 * Copyright EPFL 2021
 * Joshua Klein
 *
 * This file contains functions for mapping and tiling matrices to the AIMC
 * core using parameter write intrinsics.
 *
 */

#ifndef __AIMC_TILE_HH__
#define __AIMC_TILE_HH__

#include <stdint.h>

/* Map int8_t matrix to AIMC core, where the top left corner (0, 0) of the
 * matrix starts at (aimc_x, aimc_y) within the core.
 *
 * TODO/NOTE: There are no bounds check!
 */
inline void mapMatrix(int aimc_x, int aimc_y, int height, int width,
                      int8_t **m) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      aimcParamWrite(i + aimc_x, j + aimc_y, m[j][i]);
    }
  }

  return;
}

/* Map T matrix to AIMC core, where the top left corner (0, 0) of the
 * matrix starts at (aimc_x, aimc_y) within the core.  The values are scaled
 * to int8_t according to the maximum.
 *
 * TODO/NOTE: There are no bounds check!
 */
template <typename T>
inline void mapMatrix(int aimc_x, int aimc_y, int height, int width, T max,
                      T **m) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      aimcParamWrite(i + aimc_x, j + aimc_y, (int8_t)(m[j][i] / max * 127));
    }
  }

  return;
}

/* Map int8_t flattened matrix to AIMC core, where the top left corner (0, 0)
 * of the matrix starts at (aimc_x, aimc_y) within the core.
 *
 * TODO/NOTE: There are no bounds check!
 */
inline void mapMatrix(int aimc_x, int aimc_y, int height, int width,
                      int8_t *m) {
  // m is a flattened row-major [height][width] buffer.
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      aimcParamWrite(i + aimc_x, j + aimc_y, m[j * width + i]);
    }
  }

  return;
}

/* Map T flattened matrix to AIMC core, where the top left corner (0, 0) of the
 * matrix starts at (aimc_x, aimc_y) within the core.  The values are scaled
 * to int8_t according to the maximum.
 *
 * TODO/NOTE: There are no bounds check!
 */
template <typename T>
inline void mapMatrix(int aimc_x, int aimc_y, int height, int width, T max,
                      T *m) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      aimcParamWrite(i + aimc_x, j + aimc_y,
                     (int8_t)(m[i * width + j] / max * 127));
    }
  }

  return;
}

#endif // __AIMC_TILE_HH__
