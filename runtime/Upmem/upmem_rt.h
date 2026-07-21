

#include "timers.h"
#include <dpu.h>
#include <dpu_types.h>
#include <stddef.h>
#include <stdint.h>

/// Scatter a tensor on the given DPU set.
/// For each DPU `x`, copies `copy_bytes` bytes from `A`
/// into DPU memory (at offset `offset_in_dpu`),
/// starting from `A + base_offset(x)`.
///
/// @param dpu_set              Pointer to DPU structure
/// @param host_buffer          Input tensor to scatter
/// @param element_size         Total size of a tensor element in bytes
/// @param num_elements         Total number of elements in tensor
/// @param num_elements_per_tasklet Total number of elements for one tasklet
/// @param copy_bytes           Total number of bytes to copy into each DPU
/// @param buffer_id            Constant string of the buffer ID
/// @param base_offset          Function mapping the index of a DPU to an offset
/// in the input tensor.
void upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *host_buffer,
                         size_t element_size, size_t num_elements,
                         size_t num_elements_per_tasklet, size_t copy_bytes,
                         const char *buffer_id, size_t (*base_offset)(size_t));

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *host_buffer,
                        size_t element_size, size_t num_elements,
                        size_t num_elements_per_tasklet, size_t copy_bytes,
                        const char *buffer_id, size_t (*base_offset)(size_t));

/// Scatter a tensor on the given DPU set using the UPMEM SDK's scatter/gather
/// transfer API (dpu_push_sg_xfer), so that the block scattered to each
/// tasklet of a DPU can come from a location in `host_buffer` that is not
/// contiguous with the other tasklets' blocks. Each individual block must
/// still be contiguous in `host_buffer`.
///
/// For each DPU `x` and each tasklet `t` in `[0, num_tasklets)`, copies
/// `block_num_elements` elements from `host_buffer`, starting at byte offset
/// `base_offset(x, t)`, into the `t`-th block of the DPU's MRAM buffer
/// (i.e. at MRAM byte offset `t * block_num_elements * element_size`).
///
/// @param dpu_set              Pointer to DPU structure
/// @param host_buffer          Input tensor to scatter
/// @param element_size         Size of a tensor element in bytes
/// @param num_tasklets         Number of tasklets (blocks) per DPU
/// @param block_num_elements   Number of elements transferred per tasklet's block
/// @param buffer_id            Constant string of the buffer ID
/// @param base_offset          Function mapping (dpu_index, tasklet_index) to
/// the starting byte offset of that tasklet's block in the host buffer.
void upmemrt_dpu_scatter_to_tasklets(struct dpu_set_t *dpu_set,
                                     void *host_buffer, size_t element_size,
                                     size_t num_tasklets,
                                     size_t block_num_elements,
                                     const char *buffer_id,
                                     size_t (*base_offset)(size_t, size_t));

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus,
                                    const char *dpu_binary_path);

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set);
