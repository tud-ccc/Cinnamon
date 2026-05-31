#include <stdint.h>

// Rank-4 memref descriptor for f32 buffers.
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} MemRef4Df32;

// Flattened LLVM-ABI: kernel takes ONE rank-4 memref (input) and RETURNS a rank-4 memref (output).
extern MemRef4Df32 kernel(
  // input memref<1x3x16x16xf32> flattened as:
  float *in_alloc, float *in_aligned, int64_t in_offset,
  int64_t in_size0, int64_t in_size1, int64_t in_size2, int64_t in_size3,
  int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3
);

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
  if (value < 0) v = (unsigned long)(-value); else v = (unsigned long)value;
  if (v == 0) tmp[idx++] = '0';
  while (v) { tmp[idx++] = (char)('0' + (v % 10)); v /= 10; }
  int o = 0;
  if (value < 0) out[o++] = '-';
  while (idx--) out[o++] = tmp[idx];
  return o;
}

static int write_float_two_dec(float value, char *out) {
  int o = 0;
  if (value < 0.0f) { out[o++] = '-'; value = -value; }
  long ip = (long)value;
  float frac = value - (float)ip;
  long fp = (long)(frac * 100.0f + 0.5f);

  char tmp[32]; int idx = 0; unsigned long v = (unsigned long)ip;
  if (v == 0) tmp[idx++] = '0';
  while (v) { tmp[idx++] = (char)('0' + (v % 10)); v /= 10; }
  while (idx--) out[o++] = tmp[idx];
  out[o++] = '.';
  out[o++] = (char)('0' + ((fp / 10) % 10));
  out[o++] = (char)('0' + (fp % 10));
  return o;
}

static void print_vector(const float *data, int64_t len) {
  int64_t to_print = len < 32 ? len : 32;
  for (int64_t i = 0; i < to_print; ++i) {
    char buf[32];
    int n = write_float_two_dec(data[i], buf);
    sys_write(1, buf, (unsigned long)n);
    if (i + 1 < to_print) sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);
}

int main(void) {
  enum { N = 1, C = 3, H = 16, W = 16 };
  enum { OUT_C = 8, OUT_H = 16, OUT_W = 16 };

  static float input[N * C * H * W];
  static float output[N * OUT_C * OUT_H * OUT_W]; // keep this buffer if you want to copy into it

  for (int64_t i = 0; i < (int64_t)(N * C * H * W); ++i)
    input[i] = (float)(i % 257) * 0.01f;
  for (int64_t i = 0; i < (int64_t)(N * OUT_C * OUT_H * OUT_W); ++i)
    output[i] = 0.0f;

  // Input strides (row-major): (C*H*W, H*W, W, 1)
  const int64_t in_s0 = (int64_t)(C * H * W);
  const int64_t in_s1 = (int64_t)(H * W);
  const int64_t in_s2 = (int64_t)W;
  const int64_t in_s3 = 1;

  // Call flattened function with the **input** only; it returns the output memref.
  MemRef4Df32 out_desc = kernel(
    /* in_alloc   */ input,
    /* in_aligned */ input,
    /* in_offset  */ 0,
    /* sizes      */ N, C, H, W,
    /* strides    */ in_s0, in_s1, in_s2, in_s3
  );

  int64_t total = (int64_t)(N * OUT_C * OUT_H * OUT_W);
  for (int64_t i = 0; i < total; ++i)
    output[i] = out_desc.aligned[i];

  print_vector(output, total);
  return 0;
}
