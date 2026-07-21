
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

  // Merge the maximal run of innermost axes that are contiguous in both src
  // and dst into a single bulk memcpy, instead of copying elemSize bytes at
  // a time: e.g. for a [256, 1, 4, 1024] tile whose innermost axis is
  // contiguous in both operands, this does 1024 memcpy(4KB) calls instead of
  // 1048576 memcpy(4B) calls.
  int64_t chunkElems = 1;
  int64_t chunkAxis = rank; // first axis NOT absorbed into the chunk
  while (chunkAxis > 0 && src.strides[chunkAxis - 1] == chunkElems &&
         dst.strides[chunkAxis - 1] == chunkElems) {
    chunkElems *= src.sizes[chunkAxis - 1];
    --chunkAxis;
  }
  int64_t chunkBytes = chunkElems * elemSize;

  if (chunkAxis == 0) {
    // The whole operand is one contiguous run (this also covers rank == 0,
    // where the loop above never executes and chunkElems stays 1).
    memcpy(dstPtr, srcPtr, chunkBytes);
#ifdef UPMEM_RT_STATS
    upmemrt_record_copy(upmemrt_now_ns() - t0, numElements * elemSize);
#endif
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * chunkAxis));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * chunkAxis));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * chunkAxis));

  // Initialize index and scale strides for the remaining (non-contiguous)
  // outer axes.
  for (int64_t rankp = 0; rankp < chunkAxis; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy the largest contiguous chunk at the current position.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, chunkBytes);
    // Advance index and read position over the remaining outer axes.
    for (int64_t axis = chunkAxis - 1; axis >= 0; --axis) {
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
