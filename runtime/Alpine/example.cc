/* Copyright EPFL 2022
 * Joshua Klein
 *
 * Example dense perceptron layer using AIMClib with modest dimensions so it
 * runs quickly under gem5. The GCC/G++ build command line determines if the
 * AIMClib checker (gem5 model emulation) is used or not.
 *
 */

#include "aimc.hh"
#include "aimc_quantize.hh"

int main(int argc, char *argv[]) {
  // Test bench parameters.
  int n_x = 16; // MLP input/output dimensions kept small for fast runs.
  int T_x = 1;   // Number of inferences.

  // Set up and initialize vectors/matrices.
  // Quantization parameters (per-tensor)
  float scale_x = 1.0f, scale_w = 1.0f;
  int8_t zp_x = 0, zp_w = 0;
  float scale_y = scale_x * scale_w;
  int8_t zp_y = 0;

  // Allocate float-domain tensors
  float *input_f = new float[n_x];
  float **Wf = new float *[n_x];
  for (int j = 0; j < n_x; j++) input_f[j] = (float)(j + 1);
  for (int r = 0; r < n_x; r++) {
    Wf[r] = new float[n_x];
    for (int c = 0; c < n_x; c++) Wf[r][c] = (r == c) ? 1.0f : 0.0f;
  }

  // Allocate int8-domain tensors
  int8_t *input_q = new int8_t[n_x];
  int8_t **W = new int8_t *[n_x];
  int8_t **output = new int8_t *[T_x];
  float *output_f = new float[n_x];

  for (int i = 0; i < T_x; i++) output[i] = new int8_t[n_x];
  for (int r = 0; r < n_x; r++) W[r] = new int8_t[n_x];

  // Quantize input vector and weight matrix
  aimc_quantize_vector(n_x, input_f, input_q, scale_x, zp_x);
  for (int r = 0; r < n_x; r++)
    for (int c = 0; c < n_x; c++)
      W[r][c] = aimc_quantize_scalar(Wf[r][c], scale_w, zp_w);

  // Map weights to AIMC tile.
  mapMatrix(0, 0, n_x, n_x, W);

  // Do inference.
  for (int i = 0; i < T_x; i++) {
    // Queue quantized input for next inference in first layer.
    queueVector(n_x, input_q);

    // Do MVM.
    aimcProcess();

    // Dequeue output from AIMC tile MVM.
    dequeueVector(n_x, output[i]);
  }

  aimc_dequantize_vector(n_x, output[0], output_f, scale_y, zp_y);

  // Minimal syscall writer (no libc)
  auto sys_write = [](int fd, const char *buf, unsigned long len) {
    register long x0 asm("x0") = fd;
    register const char *x1 asm("x1") = buf;
    register unsigned long x2 asm("x2") = len;
    register long x8 asm("x8") = 64; // SYS_write
    asm volatile("svc #0" : : "r"(x0), "r"(x1), "r"(x2), "r"(x8) : "memory");
  };

  // Minimal itoa for signed int8 values
  auto itoa = [](long v, char *out) {
    char tmp[32];
    int idx = 0;
    bool neg = v < 0;
    unsigned long x = neg ? (unsigned long)(-v) : (unsigned long)v;
    if (x == 0)
      tmp[idx++] = '0';
    while (x) {
      tmp[idx++] = char('0' + (x % 10));
      x /= 10;
    }
    int o = 0;
    if (neg)
      out[o++] = '-';
    while (idx--)
      out[o++] = tmp[idx];
    return o;
  };


  // Print the quantized matvec output (first inference) as space-separated ints
  for (int j = 0; j < n_x; j++) {
    char num[32];
    int n = itoa((long)output[0][j], num);
    sys_write(1, num, (unsigned long)n);
    if (j + 1 < n_x)
      sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);


  auto ftoa2 = [](float v, char *out) {
    long sign = (v < 0.0f) ? -1 : 1;
    float av = (v < 0.0f) ? -v : v;
    long ip = (long)av;
    long fp = (long)((av - (float)ip) * 100.0f + 0.5f);
    char tmp[32]; int idx = 0;
    unsigned long x = (unsigned long)ip;
    if (x == 0) tmp[idx++] = '0';
    while (x) { tmp[idx++] = (char)('0' + (x % 10)); x /= 10; }
    int o = 0; if (sign < 0) out[o++] = '-';
    while (idx--) out[o++] = tmp[idx];
    out[o++] = '.';
    out[o++] = (char)('0' + ((fp / 10) % 10));
    out[o++] = (char)('0' + (fp % 10));
    return o;
  };

  for (int j = 0; j < n_x; j++) {
    char buf[32];
    int n = ftoa2(output_f[j], buf);
    sys_write(1, buf, (unsigned long)n);
    if (j + 1 < n_x) sys_write(1, " ", 1);
  }
  sys_write(1, "\n", 1);

  // Cleanup and return.
  for (int i = 0; i < T_x; i++) {
    delete[] output[i];
  }
  delete[] output_f;
  delete[] output;
  for (int r = 0; r < n_x; r++) delete[] W[r];
  delete[] W;
  delete[] input_q;
  for (int r = 0; r < n_x; r++) delete[] Wf[r];
  delete[] Wf;
  delete[] input_f;

  // Exit cleanly via Linux AArch64 syscall to avoid relying on libc.
  register long x0 asm("x0") = 0;  // status code
  register long x8 asm("x8") = 93; // SYS_exit
  asm volatile("svc #0" : : "r"(x0), "r"(x8) : "memory");
  __builtin_unreachable();
}
