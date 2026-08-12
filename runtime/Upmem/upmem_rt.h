

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
/// @param tag                  Optional user-supplied label (from the
/// originating op's `upmem.timing_tag` attribute) recorded alongside the
/// transfer's stats, or NULL if the op carried no tag. Ignored unless built
/// with -DUPMEM_RT_STATS.
void upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *host_buffer,
                         size_t element_size, size_t num_elements,
                         size_t num_elements_per_tasklet, size_t copy_bytes,
                         const char *buffer_id, size_t (*base_offset)(size_t),
                         const char *tag);

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *host_buffer,
                        size_t element_size, size_t num_elements,
                        size_t num_elements_per_tasklet, size_t copy_bytes,
                        const char *buffer_id, size_t (*base_offset)(size_t),
                        const char *tag);

/// Transfer several blocks per DPU using the UPMEM SDK's scatter/gather
/// transfer API (dpu_push_sg_xfer), so that a DPU's blocks may come from
/// locations in `host_buffer` that are not contiguous with one another. Each
/// individual block must still be contiguous in `host_buffer`.
///
/// For each DPU `x` and each block `b` in `[0, num_blocks)`, copies
/// `block_num_elements` elements between `host_buffer` -- starting at byte
/// offset `base_offset(x, b)` -- and the `b`-th block of the DPU's MRAM
/// buffer (i.e. at MRAM byte offset `b * block_num_elements * element_size`).
/// `upmemrt_dpu_scatter_blocks` copies host to DPU, `upmemrt_dpu_gather_blocks`
/// the other way round.
///
/// Blocks are units of transfer, not tasklets: a single tasklet's data may
/// well arrive as several of them.
///
/// @param dpu_set              Pointer to DPU structure
/// @param host_buffer          Host-side tensor
/// @param element_size         Size of a tensor element in bytes
/// @param num_blocks           Number of blocks transferred per DPU
/// @param block_num_elements   Number of elements in one block
/// @param buffer_id            Constant string of the buffer ID
/// @param base_offset          Function mapping (dpu_index, block_index) to
/// the starting byte offset of that block in the host buffer.
/// @param tag                  Optional user-supplied label (from the
/// originating op's `upmem.timing_tag` attribute) recorded alongside the
/// transfer's stats, or NULL if the op carried no tag. Ignored unless built
/// with -DUPMEM_RT_STATS.
void upmemrt_dpu_scatter_blocks(struct dpu_set_t *dpu_set, void *host_buffer,
                                size_t element_size, size_t num_blocks,
                                size_t block_num_elements,
                                const char *buffer_id,
                                size_t (*base_offset)(size_t, size_t),
                                const char *tag);

void upmemrt_dpu_gather_blocks(struct dpu_set_t *dpu_set, void *host_buffer,
                               size_t element_size, size_t num_blocks,
                               size_t block_num_elements, const char *buffer_id,
                               size_t (*base_offset)(size_t, size_t),
                               const char *tag);

/// Broadcast a buffer to the MRAM of every DPU in the set, identically.
///
/// @param dpu_set     Pointer to DPU structure
/// @param host_buffer Buffer to broadcast; the `copy_bytes` bytes starting
/// here are copied into every DPU's MRAM buffer, unchanged.
/// @param copy_bytes  Number of bytes to copy into each DPU
/// @param buffer_id   Constant string of the buffer ID
/// @param tag         Optional user-supplied label (from the originating op's
/// `upmem.timing_tag` attribute) recorded alongside the transfer's stats, or
/// NULL if the op carried no tag. Ignored unless built with -DUPMEM_RT_STATS.
void upmemrt_dpu_broadcast(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t copy_bytes, const char *buffer_id,
                           const char *tag);

/// Allocates and loads a DPU set.
///
/// @param num_dpus             Number of DPUs to allocate. Which ranks they
/// land on is the SDK's business and cannot be requested.
/// @param max_blocks_per_dpu   Largest number of blocks any
/// upmemrt_dpu_scatter_blocks/gather_blocks call against this DPU set will
/// use, or 0 if none will. Sets the UPMEM SDK's sgXferMaxBlocksPerDpu option
/// (and enables scatter/gather transfers) only when actually needed, so
/// programs that never use the scatter transfer API don't pay for its
/// (larger) memory footprint.
///
/// Allocation does NOT load a program; pair with upmemrt_dpu_load. The two
/// are separate calls (mirroring upmem.alloc_dpus / upmem.load_program)
/// because a set is acquired once per residency lifetime while the program
/// on it can change per launch -- and because their times are recorded in
/// different categories (alloc is harness overhead, load amortizes only
/// when it happens once per workload lifetime).
struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_dpus,
                                    size_t max_blocks_per_dpu);

/// Load the DPU program at @p dpu_binary_path onto every DPU of @p dpu_set
/// (the SDK's dpu_load), replacing whatever ran there before. Recorded under
/// the "load" timer category.
void upmemrt_dpu_load(struct dpu_set_t *dpu_set, const char *dpu_binary_path);

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set);
