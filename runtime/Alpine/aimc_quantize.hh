/* 
 * AIMC Quantize/Dequantize helpers for ALPINE/MLIR integration.
 *
 * These are lightweight, header-only utilities to perform per-tensor
 * quantization and dequantization between float and int8_t with a given
 * scale and zero-point. The signatures are kept simple and consistent with
 * the style of other helpers in this folder (inline C-style functions).
 *
 * Quantize:   q = clamp(round(x / scale) + zero_point, [-128, 127])
 * Dequantize: x = (q - zero_point) * scale
 *
 * Notes:
 * - No dependency on the C++ standard library; rounding is implemented
 *   manually to avoid pulling in libm when used in freestanding builds.
 * - Provide scalar and vector forms; vector forms operate in-place style.
 */

#ifndef __AIMC_QUANTIZE_HH__
#define __AIMC_QUANTIZE_HH__

#include <stdint.h>

// Helper: clamp a 32-bit integer to int8_t range
static inline int8_t aimc_clamp_int8(int32_t v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Helper: round-to-nearest (ties away from zero) without libm
static inline int32_t aimc_round_to_int(float v)
{
    return (int32_t)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
}

// Scalar quantize: float -> int8
static inline int8_t
aimc_quantize_scalar(float x, float scale, int8_t zero_point)
{
    // Guard against scale==0
    if (scale == 0.0f) {
        return zero_point;
    }
    float r = x / scale;
    int32_t q = aimc_round_to_int(r) + (int32_t)zero_point;
    return aimc_clamp_int8(q);
}

// Scalar dequantize: int8 -> float
static inline float
aimc_dequantize_scalar(int8_t q, float scale, int8_t zero_point)
{
    return ((int32_t)q - (int32_t)zero_point) * scale;
}

// Vector quantize: float[N] -> int8[N]
static inline void
aimc_quantize_vector(int size, const float* x, int8_t* q, float scale, int8_t zero_point)
{
    // Guard against scale==0
    if (scale == 0.0f) {
        for (int i = 0; i < size; i++) q[i] = zero_point;
        return;
    }
    for (int i = 0; i < size; i++) {
        float r = x[i] / scale;
        int32_t qi = aimc_round_to_int(r) + (int32_t)zero_point;
        q[i] = aimc_clamp_int8(qi);
    }
}

// Vector dequantize: int8[N] -> float[N]
static inline void
aimc_dequantize_vector(int size, const int8_t* q, float* x, float scale, int8_t zero_point)
{
    for (int i = 0; i < size; i++) {
        x[i] = ((int32_t)q[i] - (int32_t)zero_point) * scale;
    }
}

#endif // __AIMC_QUANTIZE_HH__

