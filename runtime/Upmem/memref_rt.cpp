
#include "timers.h"

#include <alloca.h>
#include <cstdint>
#include <cstring>
#include <mlir/ExecutionEngine/CRunnerUtils.h>

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg,
                           UnrankedMemRefType<char> *dstArg) {
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;

#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
  int64_t numElements = 1;
  for (int64_t rankp = 0; rankp < rank; ++rankp)
    numElements *= src.sizes[rankp];
#endif

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0) {
#ifdef UPMEM_RT_STATS
      upmemrt_record_copy(upmemrt_now_ns() - t0, 0);
#endif
      return;
    }

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
#ifdef UPMEM_RT_STATS
    upmemrt_record_copy(upmemrt_now_ns() - t0, elemSize);
#endif
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0) {
#ifdef UPMEM_RT_STATS
        upmemrt_record_copy(upmemrt_now_ns() - t0, numElements * elemSize);
#endif
        return;
      }
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}
