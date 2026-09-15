#ifndef MEMREF_RT_MIN_H
#define MEMREF_RT_MIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int64_t rank;
  void *descriptor; // -> { void* allocated; void* aligned; i64 offset;
                    //      i64 sizes[rank]; i64 strides[rank]; }
} UnrankedMemRefType_char;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  /* followed by: int64_t sizes[rank]; int64_t strides[rank]; */
} __attribute__((packed)) RankedPrefix;

void memrefCopy(int64_t elemSize, UnrankedMemRefType_char *srcArg,
                UnrankedMemRefType_char *dstArg);

#ifdef __cplusplus
}
#endif

#endif /* MEMREF_RT_MIN_H */
