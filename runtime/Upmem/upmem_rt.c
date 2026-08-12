

#include "upmem_rt.h"
#include "timers.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

#ifdef ASYNC_TRANSFERS
#define TRANSFER_FLAGS DPU_XFER_ASYNC
#else
#define TRANSFER_FLAGS DPU_XFER_DEFAULT
#endif

void do_dpu_transfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                     void *host_buffer, size_t copy_bytes, const char *buf_id,
                     size_t padding_ratio, size_t (*base_offset)(size_t)) {
  assert(copy_bytes > 0);

  // Retrieve results
  size_t i = 0;
  struct dpu_set_t dpu;
  DPU_FOREACH(*dpu_set, dpu, i) {
    size_t offset =
        base_offset(i) *
        padding_ratio; // TODO this used to work with a factor 16 inserted.
    // printf("dpu %lu - offset %lu\n", i, offset);
    // printf("%-4ld: Transfer %ld bytes from offset %ld \n", i, copy_bytes,
    // offset); fflush(stdout);
    //  TODO: This out-of-bounds check does not work when we are scattering a
    //  view, and the base tensor is larger than the view.
    // assert(offset + copy_bytes < buf_size &&
    //        "Out of bounds index returned by base_offset");
    DPU_ASSERT(dpu_prepare_xfer(dpu, (char *)host_buffer + offset));
  }

  DPU_ASSERT(dpu_push_xfer(*dpu_set, xfer_type, buf_id, 0, copy_bytes,
                           TRANSFER_FLAGS));
}

void upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *hostBuffer,
                         size_t element_size, size_t num_elements,
                         size_t num_elements_per_tasklet, size_t copy_bytes,
                         const char *bufId, size_t (*base_offset)(size_t),
                         const char *tag) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  do_dpu_transfer(DPU_XFER_TO_DPU, dpu_set, hostBuffer, copy_bytes, bufId, 1,
                  base_offset);
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  upmemrt_record_scatter(upmemrt_now_ns() - t0, copy_bytes, nr_dpus,
                         /*num_blocks=*/1, "array", tag);
#endif
}

void upmemrt_dpu_gather(struct dpu_set_t *dpu_set, void *host_buffer,
                        size_t element_size, size_t num_elements,
                        size_t num_elements_per_tasklet, size_t copy_bytes,
                        const char *bufid, size_t (*base_offset)(size_t),
                        const char *tag) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  if (num_elements * element_size >= 8) {
    do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, copy_bytes, bufid,
                    1, base_offset);
  } else {
    void *padded_result =
        malloc(num_elements * element_size * (8 / element_size));
    do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, padded_result, copy_bytes,
                    bufid, 8 / element_size, base_offset);
    for (size_t i = 0; i < num_elements; i++) {
      memcpy(host_buffer + i * element_size, padded_result + i * 8,
             element_size);
    }
  }
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  upmemrt_record_gather(upmemrt_now_ns() - t0, copy_bytes, nr_dpus,
                        /*num_blocks=*/1, "array", tag);
#endif
}

/// Arguments closed over by get_sg_xfer_block, passed through the UPMEM SDK's
/// get_block_t.args (which the SDK copies internally, so it is safe for this
/// struct to live on the stack of the calling function).
typedef struct sg_xfer_context {
  uint8_t *host_buffer;
  size_t element_size;
  size_t num_blocks;
  size_t block_num_elements;
  size_t (*base_offset)(size_t, size_t);
} sg_xfer_context;

static bool get_sg_xfer_block(struct sg_block_info *out, uint32_t dpu_index,
                              uint32_t block_index, void *args) {
  const sg_xfer_context *ctx = (const sg_xfer_context *)args;
  if (block_index >= ctx->num_blocks)
    return false;

  out->addr = ctx->host_buffer + ctx->base_offset(dpu_index, block_index);
  out->length = ctx->block_num_elements * ctx->element_size;
  return true;
}

/// Both directions of the scatter/gather transfer API; only the xfer_type and
/// which CSV the timing lands in differ.
static void do_sg_xfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                       void *host_buffer, size_t element_size,
                       size_t num_blocks, size_t block_num_elements,
                       const char *buffer_id,
                       size_t (*base_offset)(size_t, size_t), const char *tag) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  sg_xfer_context ctx = {
      .host_buffer = (uint8_t *)host_buffer,
      .element_size = element_size,
      .num_blocks = num_blocks,
      .block_num_elements = block_num_elements,
      .base_offset = base_offset,
  };
  get_block_t get_block_info = {
      .f = get_sg_xfer_block, .args = &ctx, .args_size = sizeof(ctx)};

  size_t length = num_blocks * block_num_elements * element_size;
  DPU_ASSERT(dpu_push_sg_xfer(*dpu_set, xfer_type, buffer_id, 0, length,
                              &get_block_info, DPU_SG_XFER_DEFAULT));
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  if (xfer_type == DPU_XFER_TO_DPU)
    upmemrt_record_scatter(upmemrt_now_ns() - t0, length, nr_dpus, num_blocks,
                           "blocks", tag);
  else
    upmemrt_record_gather(upmemrt_now_ns() - t0, length, nr_dpus, num_blocks,
                          "blocks", tag);
#else
  (void)tag;
#endif
}

void upmemrt_dpu_scatter_blocks(struct dpu_set_t *dpu_set, void *host_buffer,
                                size_t element_size, size_t num_blocks,
                                size_t block_num_elements,
                                const char *buffer_id,
                                size_t (*base_offset)(size_t, size_t),
                                const char *tag) {
  do_sg_xfer(DPU_XFER_TO_DPU, dpu_set, host_buffer, element_size, num_blocks,
             block_num_elements, buffer_id, base_offset, tag);
}

void upmemrt_dpu_gather_blocks(struct dpu_set_t *dpu_set, void *host_buffer,
                               size_t element_size, size_t num_blocks,
                               size_t block_num_elements, const char *buffer_id,
                               size_t (*base_offset)(size_t, size_t),
                               const char *tag) {
  do_sg_xfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, element_size, num_blocks,
             block_num_elements, buffer_id, base_offset, tag);
}

void upmemrt_dpu_broadcast(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t copy_bytes, const char *buffer_id,
                           const char *tag) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_broadcast_to(*dpu_set, buffer_id, 0, host_buffer, copy_bytes,
                              TRANSFER_FLAGS));
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  upmemrt_record_scatter(upmemrt_now_ns() - t0, copy_bytes, nr_dpus,
                         /*num_blocks=*/1, "broadcast", tag);
#endif
}

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_dpus,
                                    size_t max_blocks_per_dpu) {
  int32_t num_alloc_dpu = num_dpus;
  struct dpu_set_t *dpu_set =
      (struct dpu_set_t *)malloc(sizeof(struct dpu_set_t));
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  const char *userProfile = getenv("UPMEM_PROFILE");
  char profile[256];
  if (max_blocks_per_dpu > 0) {
    // sgXferEnable/sgXferMaxBlocksPerDpu are required for
    // upmemrt_dpu_scatter_blocks/gather_blocks (dpu_push_sg_xfer):
    // scatter/gather transfers are disabled by default, and the max number of
    // blocks per DPU otherwise defaults to the number of DPUs in the set, which
    // is too low once we're scattering one block per tasklet/mram-row. Only set
    // this when actually needed: a larger sgXferMaxBlocksPerDpu increases
    // the SDK's memory footprint.
    if (userProfile && userProfile[0] != '\0') {
      snprintf(profile, sizeof(profile),
               "%s,sgXferEnable=true,sgXferMaxBlocksPerDpu=%zu", userProfile,
               max_blocks_per_dpu);
    } else {
      snprintf(profile, sizeof(profile),
               "sgXferEnable=true,sgXferMaxBlocksPerDpu=%zu",
               max_blocks_per_dpu);
    }
  } else if (userProfile && userProfile[0] != '\0') {
    snprintf(profile, sizeof(profile), "%s", userProfile);
  } else {
    profile[0] = '\0';
  }
  DPU_ASSERT(dpu_alloc(num_alloc_dpu, profile[0] ? profile : NULL, dpu_set));
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  upmemrt_record_alloc(upmemrt_now_ns() - t0, nr_dpus);
#endif
  return dpu_set;
}

void upmemrt_dpu_load(struct dpu_set_t *dpu_set, const char *dpu_binary_path) {
  // Its own call and its own timer row, never folded into alloc: whether a
  // load amortizes depends on how often it recurs (once per workload vs per
  // operator switch), which is the analysis's judgment to make, not this
  // file's.
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_load(*dpu_set, dpu_binary_path, NULL));
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  upmemrt_record_load(upmemrt_now_ns() - t0, nr_dpus);
#endif
}

void upmemrt_dpu_launch(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif

#ifdef ASYNC_TRANSFERS
  dpu_sync(*dpu_set); // Wait for asynchronous transfers to finish.
  // This is fucking up our time measurements so I don't include it by default
#endif
  dpu_error_t error = dpu_launch(*dpu_set, DPU_SYNCHRONOUS);
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  upmemrt_record_launch(upmemrt_now_ns() - t0, nr_dpus);
#endif
  if (getenv("UPMEM_LOG")) {
    size_t i = 0;
    struct dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, i) { dpu_log_read(dpu, stdout); }
  }
  DPU_ASSERT(error);
}

void upmemrt_dpu_free(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*dpu_set, &nr_dpus);
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_free(*dpu_set));
#ifdef UPMEM_RT_STATS
  upmemrt_record_free(upmemrt_now_ns() - t0, nr_dpus);
#endif
}
