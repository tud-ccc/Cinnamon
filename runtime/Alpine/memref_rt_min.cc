#include "memref_rt_min.h"

static void load_dyn(char **dataPtr, int64_t *offset, int64_t **sizes,
                     int64_t **strides, UnrankedMemRefType_char *arg) {
  int64_t rank = arg->rank;
  RankedPrefix *p = (RankedPrefix *)arg->descriptor;
  *dataPtr = (char *)p->aligned;
  *offset = p->offset;
  int64_t *base = (int64_t *)((char *)p + sizeof(RankedPrefix));
  *sizes = base;
  *strides = base + rank;
}

void memrefCopy(int64_t elemSize, UnrankedMemRefType_char *srcArg,
                UnrankedMemRefType_char *dstArg) {
  int64_t rank = srcArg->rank;
  if (rank != dstArg->rank)
    return;

  char *srcPtr, *dstPtr;
  int64_t srcOff, dstOff;
  int64_t *srcSz, *srcSt, *dstSz, *dstSt;

  load_dyn(&srcPtr, &srcOff, &srcSz, &srcSt, srcArg);
  load_dyn(&dstPtr, &dstOff, &dstSz, &dstSt, dstArg);

  for (int64_t i = 0; i < rank; ++i)
    if (srcSz[i] == 0)
      return;

  srcPtr += srcOff * elemSize;
  dstPtr += dstOff * elemSize;

  if (rank == 0) {
    for (int64_t b = 0; b < elemSize; ++b)
      dstPtr[b] = srcPtr[b];
    return;
  }

  if (rank > 8)
    return; // minimal runtime limit

  int64_t idx[8];
  int64_t srcStrB[8], dstStrB[8];

  for (int64_t d = 0; d < rank; ++d) {
    idx[d] = 0;
    srcStrB[d] = srcSt[d] * elemSize;
    dstStrB[d] = dstSt[d] * elemSize;
  }

  int64_t r = rank - 1;
  int64_t sOff = 0, dOff = 0;

  for (;;) {
    for (int64_t b = 0; b < elemSize; ++b)
      dstPtr[dOff + b] = srcPtr[sOff + b];

    for (int64_t axis = r; axis >= 0; --axis) {
      idx[axis] += 1;
      sOff += srcStrB[axis];
      dOff += dstStrB[axis];

      if (idx[axis] < srcSz[axis]) {
        break;
      }
      if (axis == 0) {
        return; // done
      }
      sOff -= srcSz[axis] * srcStrB[axis];
      dOff -= dstSz[axis] * dstStrB[axis];
      idx[axis] = 0;
    }
  }
}
