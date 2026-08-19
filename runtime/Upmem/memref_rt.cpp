
#include "timers.h"

#include <alloca.h>
#include <cstdint>
#include <cstring>
#include <mlir/ExecutionEngine/CRunnerUtils.h>

/// Copies `srcArg` into `dstArg`, both possibly strided, and reports the
/// number of bytes moved through `bytesMoved`.
///
/// Shared by memrefCopy and upmemrt_compact: a repack is the same data
/// movement as any other strided copy and differs only in what it is
/// attributed to, so the two entry points wrap this rather than each carrying
/// their own copy loop.
static void copyStrided(int64_t elemSize, int64_t rank, const int64_t *sizes,
                        const char *srcBase, const int64_t *srcStridesIn,
                        char *dstBase, const int64_t *dstStridesIn,
                        size_t *bytesMoved) {
  struct {
    const int64_t *sizes, *strides;
  } src{sizes, srcStridesIn}, dst{sizes, dstStridesIn};

  int64_t numElements = 1;
  for (int64_t rankp = 0; rankp < rank; ++rankp)
    numElements *= sizes[rankp];
  *bytesMoved = 0;

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (sizes[rankp] == 0)
      return;

  *bytesMoved = (size_t)(numElements * elemSize);

  const char *srcPtr = srcBase;
  char *dstPtr = dstBase;

  // Merge the maximal run of innermost axes that are contiguous in both src
  // and dst into a single bulk memcpy, instead of copying elemSize bytes at
  // a time: e.g. for a [256, 1, 4, 1024] tile whose innermost axis is
  // contiguous in both operands, this does 1024 memcpy(4KB) calls instead of
  // 1048576 memcpy(4B) calls.
  //
  // A size-1 axis never advances, so it joins the chunk whatever its stride
  // says -- a repack's packed shape routinely carries one (paired with a
  // one-element buffer dimension) with a stride of 0, and stopping the merge
  // there would shrink a whole-row memcpy back down to a few bytes.
  int64_t chunkElems = 1;
  int64_t chunkAxis = rank; // first axis NOT absorbed into the chunk
  while (chunkAxis > 0 && (src.sizes[chunkAxis - 1] == 1 ||
                           (src.strides[chunkAxis - 1] == chunkElems &&
                            dst.strides[chunkAxis - 1] == chunkElems))) {
    chunkElems *= src.sizes[chunkAxis - 1];
    --chunkAxis;
  }
  int64_t chunkBytes = chunkElems * elemSize;

  if (chunkAxis == 0) {
    // The whole operand is one contiguous run (this also covers rank == 0,
    // where the loop above never executes and chunkElems stays 1).
    memcpy(dstPtr, srcPtr, chunkBytes);
    return;
  }

  int64_t *indices =
      static_cast<int64_t *>(alloca(sizeof(int64_t) * chunkAxis));
  int64_t *srcStrides =
      static_cast<int64_t *>(alloca(sizeof(int64_t) * chunkAxis));
  int64_t *dstStrides =
      static_cast<int64_t *>(alloca(sizeof(int64_t) * chunkAxis));

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
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg,
                           UnrankedMemRefType<char> *dstArg) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);
  size_t bytes = 0;
  copyStrided(elemSize, src.rank, src.sizes, src.data + src.offset * elemSize,
              src.strides, dst.data + dst.offset * elemSize, dst.strides,
              &bytes);
#ifdef UPMEM_RT_STATS
  upmemrt_record_copy(upmemrt_now_ns() - t0, bytes);
#endif
}

/// A cnm.compact_buffer repack: the same copy as memrefCopy, timed into its
/// own row so it is attributable to the layout decision that required it, and
/// separated by `isStatic` because only a repack of data that is identical on
/// every inference amortizes over the serving lifetime.
///
/// Going through a named entry point rather than memref.copy is deliberate:
/// MemRefToLLVM lowers a copy between two contiguous memrefs to
/// llvm.intr.memcpy, so a tidy repack would otherwise never be measured at
/// all.
/// A cnm.expand_buffer repack: the mirror of upmemrt_compact, writing a
/// contiguous buffer back out to a strided one. It is the gather-side half --
/// a transfer that fills a packed buffer still has to put the data where the
/// consumer expects it -- and is timed into the same rows, being the same
/// cost for the same reason.
extern "C" void upmemrt_expand(void *dst, const void *src, int64_t rank,
                               const int64_t *sizes, const int64_t *dstStrides,
                               int64_t elemSize, int32_t isStatic) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  // The source is contiguous by the op's definition, so its strides are the
  // packed ones and need not be passed.
  int64_t srcStrides[16];
  int64_t packed = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    srcStrides[i] = packed;
    packed *= sizes[i];
  }
  size_t bytes = 0;
  copyStrided(elemSize, rank, sizes, (const char *)src, srcStrides, (char *)dst,
              dstStrides, &bytes);
#ifdef UPMEM_RT_STATS
  upmemrt_record_compact(upmemrt_now_ns() - t0, bytes,
                         isStatic ? "static" : "dyn");
#else
  (void)isStatic;
#endif
}

extern "C" void upmemrt_compact(void *dst, const void *src, int64_t rank,
                                const int64_t *sizes, const int64_t *srcStrides,
                                int64_t elemSize, int32_t isStatic) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  // The target is contiguous by the op's definition, so its strides are the
  // packed ones and need not be passed.
  int64_t dstStrides[16];
  int64_t packed = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    dstStrides[i] = packed;
    packed *= sizes[i];
  }
  size_t bytes = 0;
  copyStrided(elemSize, rank, sizes, (const char *)src, srcStrides, (char *)dst,
              dstStrides, &bytes);
#ifdef UPMEM_RT_STATS
  upmemrt_record_compact(upmemrt_now_ns() - t0, bytes,
                         isStatic ? "static" : "dyn");
#else
  (void)isStatic;
#endif
}
