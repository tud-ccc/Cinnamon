#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct { float   *allocated, *aligned; int64_t offset; } MemRefF32R0;
typedef struct { int64_t *allocated, *aligned; int64_t offset; } MemRefI64R0;

typedef struct {
    float  *allocated, *aligned;
    int64_t offset, sizes[1], strides[1];
} MemRefF32R1;

typedef struct {
    float  *allocated, *aligned;
    int64_t offset, sizes[3], strides[3];
} MemRefF32R3;

/* Two memref<10x10x4xf32> results: [timestep][row][neuron] */
typedef struct {
    MemRefF32R3 spk2;   /* layer-2 spikes * scale2 (per step)     */
    MemRefF32R3 mem2;   /* layer-2 membrane after reset (per step) */
} KernelResults;


extern void _mlir_ciface_kernel(
    KernelResults *result,
    MemRefF32R0 *thr1,     /* %1  : layer-1 threshold               */
    MemRefF32R0 *scale1,   /* %2  : layer-1 spike scale             */
    MemRefI64R0 *unused0,  /* %3  : unused rank-0                   */
    MemRefF32R0 *beta1,    /* %4  : layer-1 beta (clamped 0..1)     */
    MemRefF32R1 *unused1,  /* %5  : unused rank-1                   */
    MemRefF32R1 *unused2,  /* %6  : unused rank-1                   */
    MemRefF32R0 *thr2,     /* %7  : layer-2 threshold               */
    MemRefF32R0 *scale2,   /* %8  : layer-2 spike scale             */
    MemRefI64R0 *unused3,  /* %9  : unused rank-0                   */
    MemRefF32R0 *beta2,    /* %10 : layer-2 beta (clamped 0..1)     */
    MemRefF32R1 *unused4,  /* %11 : unused rank-1                   */
    MemRefF32R3 *input);   /* %12 : input memref<10x10x128xf32>     */


static inline long sys_write(int fd, const char *buf, unsigned long len) {
    register long           x0 asm("x0") = fd;
    register const char    *x1 asm("x1") = buf;
    register unsigned long  x2 asm("x2") = len;
    register long           x8 asm("x8") = 64;   /* __NR_write */
    asm volatile("svc #0" : "+r"(x0) : "r"(x1), "r"(x2), "r"(x8) : "memory");
    return x0;
}

/* ------------------------------------------------------------------ */
/* Printing helpers                                                    */
/* ------------------------------------------------------------------ */
static int write_uint(long v, char *out) {
    char tmp[32]; int idx = 0, o = 0;
    do { tmp[idx++] = (char)('0' + (v % 10)); v /= 10; } while (v);
    while (idx--) out[o++] = tmp[idx];
    return o;
}

static int write_float_two_dec(float value, char *out) {
    int o = 0;
    if (value < 0.0f) { out[o++] = '-'; value = -value; }
    long scaled = (long)(value * 100.0f + 0.5f);   /* round to hundredths */
    o += write_uint(scaled / 100, out + o);
    long fp = scaled % 100;
    out[o++] = '.';
    out[o++] = (char)('0' + (fp / 10));
    out[o++] = (char)('0' + (fp % 10));
    return o;
}

static void print_memref3(const MemRefF32R3 *m) {
    if (!m->aligned) {
        sys_write(1, "(null)\n", 7);
        return;
    }
    const float *p = m->aligned + m->offset;
    int64_t n = m->sizes[0] * m->sizes[1] * m->sizes[2];   /* = 10 */
    char buf[32];
    for (int64_t i = 0; i < n; ++i) {
        int k = write_float_two_dec(p[i * m->strides[2]], buf);
        sys_write(1, buf, (unsigned long)k);
        sys_write(1, (i + 1 < n) ? " " : "\n", 1);
    }
}


enum { T = 10, B = 10, IN = 128 };

int main(void) {
    /* scalar parameters */
    float   thr1_val   = 1.0f;
    float   scale1_val = 1.0f;
    float   beta1_val  = 0.85f;
    float   thr2_val   = 1.0f;
    float   scale2_val = 1.0f;
    float   beta2_val  = 0.85f;
    int64_t i64_val    = 10;
    float   dummy[1]   = { 0.0f };

    MemRefF32R0 a_thr1   = { &thr1_val,   &thr1_val,   0 };
    MemRefF32R0 a_scale1 = { &scale1_val, &scale1_val, 0 };
    MemRefI64R0 a_u0     = { &i64_val,    &i64_val,    0 };
    MemRefF32R0 a_beta1  = { &beta1_val,  &beta1_val,  0 };
    MemRefF32R1 a_u1     = { dummy, dummy, 0, {1}, {1} };
    MemRefF32R1 a_u2     = { dummy, dummy, 0, {1}, {1} };
    MemRefF32R0 a_thr2   = { &thr2_val,   &thr2_val,   0 };
    MemRefF32R0 a_scale2 = { &scale2_val, &scale2_val, 0 };
    MemRefI64R0 a_u3     = { &i64_val,    &i64_val,    0 };
    MemRefF32R0 a_beta2  = { &beta2_val,  &beta2_val,  0 };
    MemRefF32R1 a_u4     = { dummy, dummy, 0, {1}, {1} };

    /* input: 10 timesteps x 10 rows x 128 features, contiguous.
       Quantizer scale is 1/32 (int8 -> about +/-4), so keep values < 4. */
    static float in_data[T * B * IN];
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < B; ++b)
            for (int i = 0; i < IN; ++i)
                in_data[(t * B + b) * IN + i] = (float)((i + b + t) % IN) / 32.0f;

    MemRefF32R3 a_input = {
        in_data, in_data,
        /*offset=*/0,
        /*sizes=*/{T, B, IN},
        /*strides=*/{B * IN, IN, 1},
    };

    KernelResults results;
    memset(&results, 0, sizeof(results));

    _mlir_ciface_kernel(&results,
                        &a_thr1, &a_scale1, &a_u0, &a_beta1, &a_u1, &a_u2,
                        &a_thr2, &a_scale2, &a_u3, &a_beta2, &a_u4,
                        &a_input);

    print_memref3(&results.spk2);
    print_memref3(&results.mem2);

    /* 'allocated' is the raw malloc() pointer returned by the kernel */
    if (results.spk2.allocated) free(results.spk2.allocated);
    if (results.mem2.allocated) free(results.mem2.allocated);
    return 0;
}