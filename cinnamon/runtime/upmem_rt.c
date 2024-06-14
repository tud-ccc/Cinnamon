

#include "upmem_rt.h"
#include <assert.h>
#include <dpu.h>
#include <dpu_types.h>
#include <stddef.h>

void do_dpu_transfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                     void *host_buffer, size_t buf_size, size_t copy_bytes,
                     size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  assert(copy_bytes > 0);
  assert(offset_in_dpu >= 0);

  // Retrieve results
  size_t i = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(dpu_set, dpu, i) {
    size_t offset = base_offset(i);
    assert(offset + copy_bytes < buf_size &&
           "Out of bounds index returned by base_offset");
    DPU_ASSERT(dpu_prepare_xfer(dpu, host_buffer + offset));
  }

  DPU_ASSERT(dpu_push_xfer(*dpu_set, xfer_type, DPU_MRAM_HEAP_POINTER_NAME,
                           offset_in_dpu, copy_bytes, DPU_XFER_DEFAULT));
}

void upmemrt_scatter_dpu(struct dpu_set_t *dpu_set, void *hostBuffer,
                         size_t input_size, size_t copy_bytes,
                         size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  do_dpu_transfer(DPU_XFER_TO_DPU, dpu_set, hostBuffer, input_size, copy_bytes,
                  offset_in_dpu, base_offset);
}

void upmemrt_gather_dpu(struct dpu_set_t *dpu_set, void *hostBuffer,
                        size_t input_size, size_t copy_bytes,
                        size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, hostBuffer, input_size,
                  copy_bytes, offset_in_dpu, base_offset);
}