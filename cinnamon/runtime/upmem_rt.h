
#include "stddef.h"
#include <dpu.h>
#include <dpu_log.h>

/// Scatter a tensor on the given DPU set.
/// For each DPU `x`, copies `copy_bytes` bytes from `A`
/// into DPU memory (at offset `offset_in_dpu`),
/// starting from `A + base_offset(x)`.
///
/// @param dpu_set          Pointer to DPU structure
/// @param A                Input tensor to scatter
/// @param input_size       Total size of tensor in bytes
/// @param copy_bytes       Total number of bytes to copy into each DPU
/// @param offset_in_dpu    Offset in the DPU memory at which to start copying
/// @param base_offset      Function mapping the index of a DPU to an offset in
/// the input tensor.
void upmemrt_scatter_dpu(struct dpu_set_t *dpu_set, void *A, size_t input_size,
                         size_t copy_bytes, size_t offset_in_dpu,
                         size_t (*base_offset)(size_t));

void upmemrt_gather_dpu(struct dpu_set_t *dpu_set, void *A, size_t input_size,
                        size_t copy_bytes, size_t offset_in_dpu,
                        size_t (*base_offset)(size_t));