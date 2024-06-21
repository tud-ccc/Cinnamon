

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
/// @param input_size_bytes     Total size of tensor in bytes
/// @param copy_bytes           Total number of bytes to copy into each DPU
/// @param offset_in_dpu_bytes  Offset in the DPU memory at which to start copying
/// @param base_offset          Function mapping the index of a DPU to an offset in
/// the input tensor.
size_t upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t input_size_bytes, size_t copy_bytes,
                           size_t offset_in_dpu_bytes, size_t (*base_offset)(size_t));

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *host_buffer,
                        size_t input_size_bytes, size_t copy_bytes,
                        size_t offset_in_dpu_bytes, size_t (*base_offset)(size_t));

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus, const char* dpu_binary_path);

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set);