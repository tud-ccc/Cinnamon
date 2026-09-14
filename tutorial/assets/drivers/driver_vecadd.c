#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <stdio.h>
#include <stdlib.h>

#if defined(_ISOC11_SOURCE) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif
#include <stdint.h>
#include <string.h>

#ifndef N
#define N 104857600  // Must match your MLIR tensor/memref size (100x baseline).
#endif

#if N != 104857600
#error "driver_vecadd expects N == 104857600"
#endif

// 1D memref descriptor for f32, as produced by MLIR's -convert-memref-to-llvm.
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1DF32;

// C interface wrapper emitted by `-llvm-request-c-wrappers`.
// If your function isn't named `vecadd` or you didn't emit the C interface,
// see the notes at the bottom.
extern void _mlir_ciface_vecadd(MemRef1DF32 *ret,
                                MemRef1DF32 *A,
                                MemRef1DF32 *B);

// Simple aligned allocation helper.
static float *alloc_aligned_i64(size_t n, size_t alignment) {
#if defined(_MSC_VER)
  return (float *)_aligned_malloc(n * sizeof(float), alignment);
#elif defined(_ISOC11_SOURCE)
  return (float *)aligned_alloc(alignment, n * sizeof(float));
#else
  void *p = NULL;
  if (posix_memalign(&p, alignment, n * sizeof(float)) != 0) return NULL;
  return (float *)p;
#endif
}

static void free_aligned(void *p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

static MemRef1DF32 make_memref_1d(float *ptr, int64_t n) {
  MemRef1DF32 m;
  m.allocated = ptr;
  m.aligned   = ptr;
  m.offset    = 0;
  m.sizes[0]  = n;
  m.strides[0]= 1;
  return m;
}

int main(void) {
  // Allocate and initialize inputs/outputs.
  float *A = alloc_aligned_i64(N, 64);
  float *B = alloc_aligned_i64(N, 64);
  if (!A || !B) {
    fprintf(stderr, "Allocation failed\n");
    return 1;
  }

  for (int64_t i = 0; i < (int64_t)N; ++i) {
    A[i] = (float)i * 0.5f;
    B[i] = 1.0f;
  }

  MemRef1DF32 mA = make_memref_1d(A, N);
  MemRef1DF32 mB = make_memref_1d(B, N);
  MemRef1DF32 mC = {0};

  // Call the MLIR-generated kernel.
  _mlir_ciface_vecadd(&mC, &mA, &mB);

  if (mC.sizes[0] != N || mC.strides[0] != 1) {
    fprintf(stderr, "Unexpected result memref shape\n");
    return 1;
  }

  float *C = mC.aligned + mC.offset;

  // Quick sanity check: print a few results.
  printf("C[0]   = %f\n", C[0]);
  printf("C[1]   = %f\n", C[1]);
  printf("C[123] = %f\n", C[123]);
  printf("C[N-1] = %f\n", C[N-1]);

  // Optional validation.
  int errors = 0;
  for (int64_t i = 0; i < (int64_t)N; ++i) {
    float want = A[i] + B[i];
    if (C[i] != want) { errors = 1; break; }
  }
  printf("Validation: %s\n", errors ? "FAIL" : "OK");

  free_aligned(A);
  free_aligned(B);
  if (mC.allocated) free_aligned(mC.allocated);
  return errors;
}
