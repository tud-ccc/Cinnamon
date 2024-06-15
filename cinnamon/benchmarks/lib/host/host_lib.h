

#include <cstdint>
#include <dpu.h>
#include <dpu_types.h>
#include <stddef.h>
#include <stdint.h>

size_t upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *hostBuffer,
                           size_t input_size, size_t copy_bytes,
                           size_t offset_in_dpu, size_t (*base_offset)(size_t));

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *hostBuffer,
                        size_t input_size, size_t copy_bytes,
                        size_t offset_in_dpu, size_t (*base_offset)(size_t));

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus);

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set);