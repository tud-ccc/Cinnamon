/* Copyright EPFL 2022
 * Joshua Klein
 *
 * Example dense perceptron layer using AIMClib sized to fit a 256x256 tile.
 * The GCC/G++ build command line determines if the AIMClib checker (gem5
 * model emulation) is used or not.
 */

#include "aimc_quantize.hh"
#include "alpine_runtime.h"

#include <memory>

namespace {
constexpr int kTileDim = 256;
constexpr int kInputDim = 32;
constexpr int kInferences = 1;

inline int row_major(int row, int col) { return row * kTileDim + col; }

inline void sys_write(int fd, const char *buf, unsigned long len) {
  register long x0 asm("x0") = fd;
  register const char *x1 asm("x1") = buf;
  register unsigned long x2 asm("x2") = len;
  register long x8 asm("x8") = 64; // SYS_write
  asm volatile("svc #0" : : "r"(x0), "r"(x1), "r"(x2), "r"(x8) : "memory");
}

inline int write_int(long v, char *out) {
  char tmp[32];
  int idx = 0;
  bool neg = v < 0;
  unsigned long x = neg ? static_cast<unsigned long>(-v) : static_cast<unsigned long>(v);
  if (x == 0)
    tmp[idx++] = '0';
  while (x) {
    tmp[idx++] = static_cast<char>('0' + (x % 10));
    x /= 10;
  }
  int o = 0;
  if (neg)
    out[o++] = '-';
  while (idx--)
    out[o++] = tmp[idx];
  return o;
}

inline int write_float_two_dec(float v, char *out) {
  long sign = (v < 0.0f) ? -1 : 1;
  float abs_v = (v < 0.0f) ? -v : v;
  long ip = static_cast<long>(abs_v);
  long fp = static_cast<long>((abs_v - static_cast<float>(ip)) * 100.0f + 0.5f);
  char tmp[32];
  int idx = 0;
  unsigned long x = static_cast<unsigned long>(ip);
  if (x == 0)
    tmp[idx++] = '0';
  while (x) {
    tmp[idx++] = static_cast<char>('0' + (x % 10));
    x /= 10;
  }
  int o = 0;
  if (sign < 0)
    out[o++] = '-';
  while (idx--)
    out[o++] = tmp[idx];
  out[o++] = '.';
  out[o++] = static_cast<char>('0' + ((fp / 10) % 10));
  out[o++] = static_cast<char>('0' + (fp % 10));
  return o;
}

} // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  // Quantization parameters (per-tensor)
  constexpr float scale_x = 1.0f;
  constexpr float scale_w = 1.0f;
  constexpr int8_t zp_x = 0;
  constexpr int8_t zp_w = 0;
  constexpr float scale_y = scale_x * scale_w;
  constexpr int8_t zp_y = 0;

  auto input_f = std::unique_ptr<float[]>(new float[kInputDim]);
  auto weights_f = std::unique_ptr<float[]>(new float[kTileDim * kTileDim]);

  for (int r = 0; r < kTileDim; ++r)
    for (int c = 0; c < kTileDim; ++c)
      weights_f[row_major(r, c)] = 0.0f;

  for (int r = 0; r < kInputDim; ++r) {
    input_f[r] = static_cast<float>(r + 1);
    weights_f[row_major(r, r)] = 1.0f;
  }

  auto input_q = std::unique_ptr<int8_t[]>(new int8_t[kInputDim]);
  auto weights_q = std::unique_ptr<int8_t[]>(new int8_t[kTileDim * kTileDim]);
  auto output_q = std::unique_ptr<int8_t[]>(new int8_t[kInputDim]);
  auto output_f = std::unique_ptr<float[]>(new float[kInputDim]);

  aimc_quantize_vector(kInputDim, input_f.get(), input_q.get(), scale_x, zp_x);
  for (int r = 0; r < kTileDim; ++r)
    for (int c = 0; c < kTileDim; ++c)
      weights_q[row_major(r, c)] =
          aimc_quantize_scalar(weights_f[row_major(r, c)], scale_w, zp_w);

  // Inform the runtime about the tile dimensions (updates gem5 checker too).
  alpine_alloc_tile(kTileDim, kTileDim);

  // Write the entire tile worth of weights (row-major).
  alpine_write_weights(/*tile=*/0,
                       /*alloc=*/weights_q.get(),
                       /*aligned=*/weights_q.get(),
                       /*offset=*/0,
                       /*rows=*/kTileDim,
                       /*cols=*/kTileDim,
                       /*stride0=*/kTileDim,
                       /*stride1=*/1,
                       /*reserved=*/0);

  for (int iter = 0; iter < kInferences; ++iter) {
    alpine_enqueue_vec(/*tile=*/0,
                       /*alloc=*/input_q.get(),
                       /*unused=*/input_q.get(),
                       /*offset=*/0,
                       /*size=*/kInputDim,
                       /*stride=*/1,
                       /*reserved=*/0);

    alpine_process(/*tile=*/0);

    alpine_dequeue_vec(/*tile=*/0,
                       /*alloc=*/output_q.get(),
                       /*unused=*/output_q.get(),
                       /*offset=*/0,
                       /*size=*/kInputDim,
                       /*stride=*/1,
                       /*reserved=*/0);
  }

  aimc_dequantize_vector(kInputDim, output_q.get(), output_f.get(), scale_y, zp_y);

  for (int j = 0; j < kInputDim; ++j) {
    char buf[32];
    int n = write_int(static_cast<long>(output_q[j]), buf);
    sys_write(1, buf, static_cast<unsigned long>(n));
    if (j + 1 < kInputDim)
      sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);

  for (int j = 0; j < kInputDim; ++j) {
    char buf[32];
    int n = write_float_two_dec(output_f[j], buf);
    sys_write(1, buf, static_cast<unsigned long>(n));
    if (j + 1 < kInputDim)
      sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);

  register long x0 asm("x0") = 0;  // status code
  register long x8 asm("x8") = 93; // SYS_exit
  asm volatile("svc #0" : : "r"(x0), "r"(x8) : "memory");
  __builtin_unreachable();
}
