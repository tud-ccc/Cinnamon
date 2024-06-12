

#include "upmem_rt.h"

void scatter_dpu(struct dpu_set_t *dpu_set,
                 void *A, size_t input_size, size_t copy_bytes,
                 size_t offset_in_dpu, size_t (*base_offset)(size_t)) {
  size_t i;
  struct dpu_set_t dpu;
  DPU_FOREACH(*dpu_set, dpu, i) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, A + base_offset(i)));
  }
  DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU,
                           DPU_MRAM_HEAP_POINTER_NAME, offset_in_dpu,
                           copy_bytes, DPU_XFER_DEFAULT));
}