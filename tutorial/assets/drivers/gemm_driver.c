
#include <stdint.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2Df32;

extern MemRef2Df32 kernel(
    float *a_alloc, float *a_aligned, int64_t a_offset,
    int64_t a_size0, int64_t a_size1, int64_t a_stride0, int64_t a_stride1,
    float *b_alloc, float *b_aligned, int64_t b_offset,
    int64_t b_size0, int64_t b_size1, int64_t b_stride0, int64_t b_stride1);

static inline void sys_write(int fd, const char *buf, unsigned long len) {
  register long x0 asm("x0") = fd;
  register const char *x1 asm("x1") = buf;
  register unsigned long x2 asm("x2") = len;
  register long x8 asm("x8") = 64;
  asm volatile("svc #0" : : "r"(x0), "r"(x1), "r"(x2), "r"(x8) : "memory");
}

static int write_int(long value, char *out) {
  char tmp[32];
  int idx = 0;
  unsigned long v;
  if (value < 0)
    v = (unsigned long)(-value);
  else
    v = (unsigned long)value;
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

static void print_vector(const float *data, int64_t len) {
  int64_t to_print = len < 32 ? len : 32;
  for (int64_t i = 0; i < to_print; ++i) {
    char buf[32];
    int n = write_float_two_dec(data[i], buf);
    sys_write(1, buf, (unsigned long)n);
    if (i + 1 < to_print)
      sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);
}

int main(void) {
  enum { M = 1024, K = 2048, N = 512 };

  static float a[M * K];
  static float b[K * N];
  static float c[M * N];

  for (int64_t i = 0; i < (int64_t)(M * K); ++i)
    a[i] = (float)((i % 113) - 56) * 0.01f;
  for (int64_t i = 0; i < (int64_t)(K * N); ++i)
    b[i] = (float)((i % 97) - 48) * 0.02f;
  for (int64_t i = 0; i < (int64_t)(M * N); ++i)
    c[i] = 0.0f;

  const int64_t a_s0 = K;
  const int64_t a_s1 = 1;
  const int64_t b_s0 = N;
  const int64_t b_s1 = 1;

  MemRef2Df32 out = kernel(a, a, 0, M, K, a_s0, a_s1,
                           b, b, 0, K, N, b_s0, b_s1);

  int64_t total = (int64_t)(M * N);
  for (int64_t i = 0; i < total; ++i)
    c[i] = out.aligned[i];

  print_vector(c, total);
  sys_write(1, "rows=", 5);
  char buf[32];
  int n = write_int(M, buf);
  sys_write(1, buf, (unsigned long)n);
  sys_write(1, " cols=", 6);
  n = write_int(N, buf);
  sys_write(1, buf, (unsigned long)n);
  sys_write(1, "\n", 1);
  return 0;
}

