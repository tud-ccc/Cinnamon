// replacement for math.h, which is not available in the upmem DPU runtime.
#include "string.h"
#include <stdint.h>

#define PI 3.141592654f

/* Value returned on overflow.  With IEEE 754 floating point, this is
   +Infinity, otherwise the largest representable positive value.  */
#define HUGE_VAL (__builtin_huge_val())

/* IEEE positive infinity.  */
#define INFINITY (__builtin_inff())

/* IEEE Not A Number.  */
#define NAN (__builtin_nanf(""))

static float fmaf(float a, float b, float c) { return a * b + c; }

static float fabsf(float a) { return a >= 0.0f ? a : -a; }

static float int_as_float(int a) {
  float r = 0;
  memcpy(&r, &a, sizeof(float));
  return r;
}

// epxf implementation taken from:
// https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528
/*
  Copyright (c) 2015-2021, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
static float expf(float a) {
  if (a == 0.0f || a == -0.0f) {
    return 1.0f;
  } else if (a == -INFINITY) {
    return 0.0f;
  } else if (a == INFINITY) {
    return INFINITY;
  } else if (a == NAN) {
    return NAN;
  }

  float f, r, j, s, t;
  int i, ia;

  // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
  j = fmaf(1.442695f, a, 12582912.f) - 12582912.f; // 0x1.715476p0, 0x1.8p23
  f = fmaf(j, -6.93145752e-1f, a); // -0x1.62e400p-1  // log_2_hi
  f = fmaf(j, -1.42860677e-6f, f); // -0x1.7f7d1cp-20 // log_2_lo
  i = (int)j;
  // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
  r = 1.37805939e-3f;             // 0x1.694000p-10
  r = fmaf(r, f, 8.37312452e-3f); // 0x1.125edcp-7
  r = fmaf(r, f, 4.16695364e-2f); // 0x1.555b5ap-5
  r = fmaf(r, f, 1.66664720e-1f); // 0x1.555450p-3
  r = fmaf(r, f, 4.99999851e-1f); // 0x1.fffff6p-2
  r = fmaf(r, f, 1.00000000e+0f); // 0x1.000000p+0
  r = fmaf(r, f, 1.00000000e+0f); // 0x1.000000p+0
  // exp(a) = 2**i * r
  ia = (i > 0) ? 0 : 0x83000000;
  s = int_as_float(0x7f000000 + ia);
  t = int_as_float((i << 23) - ia);
  r = r * s;
  r = r * t;
  // handle special cases: severe overflow / underflow
  if (fabsf(a) >= 104.0f)
    r = s * s;
  return r;
}

static float logf(float x) {
  if (x == 0.0f || x == -0.0f) {
    return -INFINITY;
  } else if (x == 1.0f) {
    return 0.0f;
  } else if (x < 0.0f) {
    return NAN;
  } else if (x == INFINITY) {
    return INFINITY;
  } else if (x == NAN) {
    return NAN;
  } else if (x > 0.0f && x <= 10.0f) {
    // aproximation of log, works in range (0.0, 89.0]
    // https://stackoverflow.com/a/63773160

    // using the first term of the taylor series as initial-value
    float yn = x - 1.0f;
    float yn1 = yn;

    do {
      yn = yn1;
      yn1 = yn + 2 * (x - expf(yn)) / (x + expf(yn));
    } while (fabsf(yn - yn1) > 1e-6);

    return yn1;
  } else {
    // approximate ln(x) = ln(2ⁿ y) = ln(2ⁿ) + ln(y) = ln(2) n + ln(y)
    // https://stackoverflow.com/a/44232045
    int log2 = 31 - __builtin_clz((int32_t)x);
    float y = x / (float)(1 << log2); // normalized value between [1.0, 2.0]
    return logf(y) + ((float)log2) * 0.69314718; // ln(2) = 0.69314718
  }
}

static float powf(float a, float b) {
  if (b == 0.0f) {
    return 1.0f;
  } else if (a == 0.0f) {
    return 0.0f;
  } else if (a == NAN || b == NAN) {
    return NAN;
  } else {
    return expf(b * logf(a));
  }
}

// fast inverse square root
// https://en.wikipedia.org/wiki/Fast_inverse_square_root
static float isqrtf(float number) {
  if (number < 0.0f) {
    return -NAN;
  } else if (number == 0.0f) {
    return INFINITY;
  }

  const float x2 = number * 0.5F;
  float y = number;
  int32_t i = *(int32_t *)&y;
  i = 0x5f3759df - (i >> 1);
  y = *(float *)&i;

  // we do 3 iterations for better accuracy
  for (size_t i = 0; i < 3; i++) {
    y = y * (1.5F - (x2 * y * y));
  }

  return y;
}

static float sqrtf(float number) { return 1.0f / isqrtf(number); }
