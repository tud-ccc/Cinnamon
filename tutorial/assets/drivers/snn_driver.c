#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* MLIR memref descriptors                                             */
/* ------------------------------------------------------------------ */

/* rank-0: memref<f32> / memref<i64> */
typedef struct {
    float   *allocated;
    float   *aligned;
    int64_t  offset;
} MemRefF32R0;

typedef struct {
    int64_t *allocated;
    int64_t *aligned;
    int64_t  offset;
} MemRefI64R0;

/* rank-1 */
typedef struct {
    float   *allocated;
    float   *aligned;
    int64_t  offset;
    int64_t  sizes[1];
    int64_t  strides[1];
} MemRefF32R1;

/* rank-2: memref<1x128xf32> (input) */
typedef struct {
    float   *allocated;
    float   *aligned;
    int64_t  offset;
    int64_t  sizes[2];
    int64_t  strides[2];
} MemRefF32R2;

/* rank-3: memref<1x1x10xf32> (results) */
typedef struct {
    float   *allocated;
    float   *aligned;
    int64_t  offset;
    int64_t  sizes[3];
    int64_t  strides[3];
} MemRefF32R3;

/* Packed result struct: two memref<1x1x10xf32> results */
typedef struct {
    MemRefF32R3 result0;   /* layer-2 spikes * scale2          */
    MemRefF32R3 result1;   /* layer-2 membrane after reset     */
} KernelResults;

_Static_assert(sizeof(MemRefF32R3)   == 72,  "rank-3 descriptor layout");
_Static_assert(sizeof(KernelResults) == 144, "result struct layout");

/* ------------------------------------------------------------------ */
/* Kernel entry point: result pointer + 10 memref descriptors          */
/* ------------------------------------------------------------------ */
extern void _mlir_ciface_kernel(
    KernelResults *result,
    MemRefF32R0   *arg0,   /* %1  : layer-2 threshold            */
    MemRefF32R0   *arg1,   /* %2  : layer-2 output spike scale   */
    MemRefI64R0   *arg2,   /* %3  : unused                       */
    MemRefF32R1   *arg3,   /* %4  : unused (rank-1)              */
    MemRefF32R0   *arg4,   /* %5  : layer-1 threshold            */
    MemRefF32R0   *arg5,   /* %6  : layer-1 spike scale          */
    MemRefI64R0   *arg6,   /* %7  : unused                       */
    MemRefF32R0   *arg7,   /* %8  : layer-1 beta (clamped 0..1)  */
    MemRefF32R1   *arg8,   /* %9  : unused (rank-1)              */
    MemRefF32R2   *arg9);  /* %10 : input memref<1x128xf32>      */

/* ------------------------------------------------------------------ */
/* Raw AArch64 Linux write syscall                                     */
/* ------------------------------------------------------------------ */
static inline long sys_write(int fd, const char *buf, unsigned long len) {
    register long           x0 asm("x0") = fd;
    register const char    *x1 asm("x1") = buf;
    register unsigned long  x2 asm("x2") = len;
    register long           x8 asm("x8") = 64;   /* __NR_write */
    asm volatile("svc #0"
                 : "+r"(x0)                     /* x0 receives the return value */
                 : "r"(x1), "r"(x2), "r"(x8)
                 : "memory");
    return x0;
}

/* ------------------------------------------------------------------ */
/* Printing helpers                                                    */
/* ------------------------------------------------------------------ */
static int write_float_two_dec(float value, char *out) {
    int o = 0;
    if (value < 0.0f) {
        out[o++] = '-';
        value = -value;
    }
    /* round once to hundredths, then split, so 0.999 -> 1.00 */
    long scaled = (long)(value * 100.0f + 0.5f);
    long ip = scaled / 100;
    long fp = scaled % 100;

    char tmp[32];
    int idx = 0;
    do {
        tmp[idx++] = (char)('0' + (ip % 10));
        ip /= 10;
    } while (ip);
    while (idx--)
        out[o++] = tmp[idx];
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

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {
    /* scalar parameters */
    float   thr2_val   = 1.0f;    /* layer-2 threshold   */
    float   scale2_val = 1.0f;    /* layer-2 spike scale */
    float   thr1_val   = 1.0f;    /* layer-1 threshold   */
    float   scale1_val = 1.0f;    /* layer-1 spike scale */
    float   beta1_val  = 0.85f;   /* layer-1 beta        */
    int64_t i64_val    = 10;
    float   dummy[1]   = { 0.0f };

    MemRefF32R0 arg0 = { &thr2_val,   &thr2_val,   0 };
    MemRefF32R0 arg1 = { &scale2_val, &scale2_val, 0 };
    MemRefI64R0 arg2 = { &i64_val,    &i64_val,    0 };
    MemRefF32R1 arg3 = { dummy, dummy, 0, {0}, {1} };
    MemRefF32R0 arg4 = { &thr1_val,   &thr1_val,   0 };
    MemRefF32R0 arg5 = { &scale1_val, &scale1_val, 0 };
    MemRefI64R0 arg6 = { &i64_val,    &i64_val,    0 };
    MemRefF32R0 arg7 = { &beta1_val,  &beta1_val,  0 };
    MemRefF32R1 arg8 = { dummy, dummy, 0, {0}, {1} };

    /* input: keep values small; quantizer scale is 1/32, so large
       values would saturate the int8 range */
    static float in_data[1 * 128];
    for (int i = 0; i < 128; ++i)
        in_data[i] = (float)i / 32.0f;
    MemRefF32R2 arg9 = {
        in_data, in_data,
        /*offset=*/0,
        /*sizes=*/{1, 128},
        /*strides=*/{128, 1},
    };

    KernelResults results;
    memset(&results, 0, sizeof(results));

    _mlir_ciface_kernel(&results,
                        &arg0, &arg1, &arg2, &arg3, &arg4,
                        &arg5, &arg6, &arg7, &arg8, &arg9);

    print_memref3(&results.result0);
    print_memref3(&results.result1);

    if (results.result0.allocated) free(results.result0.allocated);
    if (results.result1.allocated) free(results.result1.allocated);
    return 0;
}