

#include "upmem_rt.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void do_dpu_transfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                     void *host_buffer, size_t copy_bytes, size_t offset_in_dpu,
                     size_t padding_ratio, size_t (*base_offset)(size_t)) {
  assert(copy_bytes > 0);
  assert(offset_in_dpu >= 0);

  // Retrieve results
  size_t i = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(*dpu_set, dpu, i) {
    size_t offset = base_offset(i) * padding_ratio; // TODO this used to work with a factor 16 inserted.
    // printf("dpu %lu - offset %lu\n", i, offset);
    // printf("%-4ld: Transfer %ld bytes from offset %ld \n", i, copy_bytes,
    // offset); fflush(stdout);
    //  TODO: This out-of-bounds check does not work when we are scattering a
    //  view, and the base tensor is larger than the view.
    // assert(offset + copy_bytes < buf_size &&
    //        "Out of bounds index returned by base_offset");
    DPU_ASSERT(dpu_prepare_xfer(dpu, (char *)host_buffer + offset));
  }

  DPU_ASSERT(dpu_push_xfer(*dpu_set, xfer_type, DPU_MRAM_HEAP_POINTER_NAME,
                           offset_in_dpu, copy_bytes, DPU_XFER_DEFAULT));
}

size_t upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *hostBuffer,
                           size_t element_size, size_t num_elements,
                           size_t num_elements_per_tasklet, size_t copy_bytes,
                           size_t offset_in_dpu,
                           size_t (*base_offset)(size_t)) {
  do_dpu_transfer(DPU_XFER_TO_DPU, dpu_set, hostBuffer, copy_bytes,
                  offset_in_dpu, 1, base_offset);
  return offset_in_dpu + copy_bytes;
}

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *host_buffer,
                        size_t element_size, size_t num_elements,
                        size_t num_elements_per_tasklet, size_t copy_bytes,
                        size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  if (num_elements_per_tasklet * element_size >= 8) {
    do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, copy_bytes,
                    offset_in_dpu, 1, base_offset);
  } else {
    void *padded_result =
        malloc(num_elements * element_size * (8 / element_size));
    do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, padded_result, copy_bytes,
                    offset_in_dpu, 8 / element_size, base_offset);
    for (size_t i = 0; i < num_elements; i++) {
      memcpy(host_buffer + i * element_size, padded_result + i * 8,
             element_size);
    }
  }
}

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus,
                                    const char *dpu_binary_path) {
  int32_t num_alloc_dpu = num_ranks * num_dpus;
  struct dpu_set_t *dpu_set =
      (struct dpu_set_t *)malloc(sizeof(struct dpu_set_t));
  DPU_ASSERT(dpu_alloc(num_alloc_dpu, getenv("UPMEM_PROFILE"), dpu_set));
  DPU_ASSERT(dpu_load(*dpu_set, dpu_binary_path, NULL));
  return dpu_set;
}

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
  dpu_error_t error = dpu_launch(*dpu_set, DPU_SYNCHRONOUS);
  if (getenv("UPMEM_LOG")) {
    size_t i = 0;
    struct dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, i) { dpu_log_read(dpu, stdout); }
  }
  DPU_ASSERT(error);
}

void upmemrt_dpu_free(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
  DPU_ASSERT(dpu_free(*dpu_set));
}
