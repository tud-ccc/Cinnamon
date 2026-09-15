#include <stdint.h>

// Minimal memref descriptors matching the MLIR C interface for the lowered
// function signature. Only the fields we use are populated.
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1DF32;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2DF32;

extern void alpine_example(MemRef1DF32 *input, MemRef2DF32 *weights,
                           MemRef1DF32 *output);

enum { kInputDim = 32, kTileDim = 256 };

static inline void sys_write(int fd, const char *buf, unsigned long len) {
  register long x0 asm("x0") = fd;
  register const char *x1 asm("x1") = buf;
  register unsigned long x2 asm("x2") = len;
  register long x8 asm("x8") = 64; // SYS_write
  asm volatile("svc #0" : : "r"(x0), "r"(x1), "r"(x2), "r"(x8) : "memory");
}

static int write_int(long value, char *out) {
  char tmp[32];
  int idx = 0;
  unsigned long v;
  if (value < 0) {
    v = (unsigned long)(-value);
  } else {
    v = (unsigned long)value;
  }
  if (v == 0)
    tmp[idx++] = '0';
  while (v) {
    tmp[idx++] = (char)('0' + (v % 10));
    v /= 10;
  }
  int o = 0;
  if (value < 0)
    out[o++] = '-';
  while (idx--)
    out[o++] = tmp[idx];
  return o;
}

static int write_float_two_dec(float value, char *out) {
  int o = 0;
  if (value < 0.0f) {
    out[o++] = '-';
    value = -value;
  }
  long ip = (long)value;
  float frac = value - (float)ip;
  long fp = (long)(frac * 100.0f + 0.5f);

  char tmp[32];
  int idx = 0;
  unsigned long v = (unsigned long)ip;
  if (v == 0)
    tmp[idx++] = '0';
  while (v) {
    tmp[idx++] = (char)('0' + (v % 10));
    v /= 10;
  }
  while (idx--)
    out[o++] = tmp[idx];
  out[o++] = '.';
  out[o++] = (char)('0' + ((fp / 10) % 10));
  out[o++] = (char)('0' + (fp % 10));
  return o;
}

static int8_t quantize_scalar(float value) {
  float rounded = value >= 0.0f ? value + 0.5f : value - 0.5f;
  long q = (long)rounded;
  if (q > 127)
    q = 127;
  if (q < -128)
    q = -128;
  return (int8_t)q;
}

static void print_vector_int(const int8_t *data, int64_t len) {
  for (int64_t i = 0; i < len; ++i) {
    char buf[32];
    int n = write_int((long)data[i], buf);
    sys_write(1, buf, (unsigned long)n);
    if (i + 1 < len)
      sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);
}

static void print_vector_float(const float *data, int64_t len) {
  for (int64_t i = 0; i < len; ++i) {
    char buf[32];
    int n = write_float_two_dec(data[i], buf);
    sys_write(1, buf, (unsigned long)n);
    if (i + 1 < len)
      sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);
}

int main(void) {
  static float weights[kTileDim * kTileDim];
  static float input[kInputDim];
  static float output[kInputDim];
  static int8_t output_q[kInputDim];

  for (int64_t r = 0; r < kTileDim; ++r) {
    for (int64_t c = 0; c < kTileDim; ++c)
      weights[r * kTileDim + c] = 0.0f;
  }

  for (int64_t i = 0; i < kInputDim; ++i) {
    input[i] = (float)(i + 1);
    output[i] = 0.0f;
    weights[i * kTileDim + i] = 1.0f;
  }

  MemRef1DF32 input_desc = {
      .allocated = input,
      .aligned = input,
      .offset = 0,
      .sizes = {kInputDim},
      .strides = {1},
  };

  MemRef1DF32 output_desc = {
      .allocated = output,
      .aligned = output,
      .offset = 0,
      .sizes = {kInputDim},
      .strides = {1},
  };

  MemRef2DF32 weights_desc = {
      .allocated = weights,
      .aligned = weights,
      .offset = 0,
      .sizes = {kTileDim, kTileDim},
      .strides = {kTileDim, 1},
  };

  print_vector_float(input, kInputDim);

  alpine_example(&input_desc, &weights_desc, &output_desc);

  for (int64_t i = 0; i < kInputDim; ++i)
    output_q[i] = quantize_scalar(output[i]);

  print_vector_int(output_q, kInputDim);
  print_vector_float(output, kInputDim);

  // Exit via syscall to avoid relying on libc.
  register long x0 asm("x0") = 0;
  register long x8 asm("x8") = 93; // SYS_exit
  asm volatile("svc #0" : : "r"(x0), "r"(x8) : "memory");
  __builtin_unreachable();
}
