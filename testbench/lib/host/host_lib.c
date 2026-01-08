

#include "host_lib.h"
#include "binary_path.h"
#include <assert.h>

void do_dpu_transfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                     void *host_buffer, size_t buf_size, size_t copy_bytes,
                     size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  // assert(copy_bytes > 0);
  // assert(offset_in_dpu >= 0);

  // // Retrieve results
  // size_t i = 0;
  // struct dpu_set_t dpu;
  // DPU_FOREACH(*dpu_set, dpu, i) {
  //   size_t offset = base_offset(i);
  //   // assert(offset + copy_bytes < buf_size &&
  //   //        "Out of bounds index returned by base_offset");
  //   DPU_ASSERT(dpu_prepare_xfer(dpu, (char*) host_buffer + offset));
  // }

  // DPU_ASSERT(dpu_push_xfer(*dpu_set, xfer_type, DPU_MRAM_HEAP_POINTER_NAME,
  //                          offset_in_dpu, copy_bytes, DPU_XFER_DEFAULT));
}

size_t upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *hostBuffer,
                           size_t input_size, size_t copy_bytes,
                           size_t offset_in_dpu,
                           size_t (*base_offset)(size_t)) {
  do_dpu_transfer(DPU_XFER_TO_DPU, dpu_set, hostBuffer, input_size, copy_bytes,
                  offset_in_dpu, base_offset);
  return offset_in_dpu + copy_bytes;
}

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *hostBuffer,
                        size_t input_size, size_t copy_bytes,
                        size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, hostBuffer, input_size,
                  copy_bytes, offset_in_dpu, base_offset);
}

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus) {
  int32_t num_alloc_dpu = num_ranks * num_dpus;
  struct dpu_set_t *dpu_set =
      (struct dpu_set_t *)malloc(sizeof(struct dpu_set_t));
  DPU_ASSERT(dpu_alloc(1, NULL, dpu_set));
  DPU_ASSERT(dpu_load(*dpu_set, DPU_BINARY, NULL));
  return dpu_set;
}

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
  DPU_ASSERT(dpu_launch(*dpu_set, DPU_SYNCHRONOUS));
}


void upmemrt_dpu_free(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
  DPU_ASSERT(dpu_free(*dpu_set));
}
