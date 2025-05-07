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

#include <string.h>

float fmaf(float a, float b, float c) { return a * b + c; }

float fabsf(float a) { return a < 0 ? -a : a; }

float int_as_float(int a) {
  float r = 0;
  memcpy(&r, &a, sizeof(float));
  return r;
}

float expf(float a) {
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
