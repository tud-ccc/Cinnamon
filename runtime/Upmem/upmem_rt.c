

#include "upmem_rt.h"
#include "timers.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ASYNC_TRANSFERS
#define TRANSFER_FLAGS DPU_XFER_ASYNC
#else
#define TRANSFER_FLAGS DPU_XFER_DEFAULT
#endif

// Residency cache, defined below the transfer functions that consult it.
static int rt_transfer_resident(struct dpu_set_t *set, const char *tag,
                                void *host, size_t bytes);

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
  if (rt_transfer_resident(dpu_set, tag, hostBuffer, copy_bytes))
    return;
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
  if (rt_transfer_resident(dpu_set, tag, host_buffer,
                           num_blocks * block_num_elements * element_size))
    return;
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
  if (rt_transfer_resident(dpu_set, tag, host_buffer, copy_bytes))
    return;
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

// ─── Cross-inference residency cache (UPMEM_RT_CACHE=1) ─────────────────────
//
// See upmemrt_dpu_alloc_cached in the header for the contract. The registry
// below is the single mechanism both RQ4 arms run under: a partition that
// fits stays resident (sets, programs, static transfers survive across
// inferences), and sets that cannot coexist evict each other through the
// LRU, physically paying realloc + reload + rescatter per operator switch.

/// One "static:"-tagged transfer known resident on a set: the site (tag),
/// and the payload it holds, so a different payload over the same MRAM
/// symbol is detected as an overwrite rather than skipped.
typedef struct rt_xfer_record {
  char tag[64];
  void *host;
  size_t bytes;
  struct rt_xfer_record *next;
} rt_xfer_record;

typedef struct rt_cache_entry {
  void **slot; // the per-site global; holds the set while cached
  struct dpu_set_t *set;
  char loaded_path[512]; // "" = no program resident
  int live;              // between alloc_cached and dpu_free
  uint64_t lru;
  rt_xfer_record *xfers;
  struct rt_cache_entry *next;
} rt_cache_entry;

static rt_cache_entry *rt_cache_head = NULL;
static uint64_t rt_lru_tick = 0;

static int rt_cache_enabled(void) {
  static int enabled = -1;
  if (enabled < 0) {
    const char *v = getenv("UPMEM_RT_CACHE");
    enabled = v && v[0] && strcmp(v, "0") != 0;
  }
  return enabled;
}

static rt_cache_entry *rt_entry_of(struct dpu_set_t *set) {
  for (rt_cache_entry *e = rt_cache_head; e; e = e->next)
    if (e->set == set)
      return e;
  return NULL;
}

static void rt_drop_xfers(rt_cache_entry *e) {
  while (e->xfers) {
    rt_xfer_record *r = e->xfers;
    e->xfers = r->next;
    free(r);
  }
}

/// Really free a cached set: everything it held resident is gone.
static void rt_evict(rt_cache_entry *victim) {
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*victim->set, &nr_dpus);
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_free(*victim->set));
#ifdef UPMEM_RT_STATS
  upmemrt_record_free(upmemrt_now_ns() - t0, nr_dpus);
#endif
  free(victim->set);
  rt_drop_xfers(victim);
  *victim->slot = NULL;
  rt_cache_entry **link = &rt_cache_head;
  while (*link != victim)
    link = &(*link)->next;
  *link = victim->next;
  free(victim);
}

static void rt_evict_all(void) {
  while (rt_cache_head)
    rt_evict(rt_cache_head);
}

/// The transfer-side half of the cache: true when this static transfer's
/// payload is already resident on the set, in which case the caller skips
/// it. Recording happens here too, so the caller only ever asks.
static int rt_transfer_resident(struct dpu_set_t *set, const char *tag,
                                void *host, size_t bytes) {
  if (!tag || strncmp(tag, "static:", 7) != 0)
    return 0;
  rt_cache_entry *e = rt_entry_of(set);
  if (!e)
    return 0;
  for (rt_xfer_record *r = e->xfers; r; r = r->next)
    if (strncmp(r->tag, tag, sizeof(r->tag)) == 0) {
      if (r->host == host && r->bytes == bytes)
        return 1;
      // Same site, different payload (another member's weights over the
      // same symbol): run the transfer and remember the new occupant.
      r->host = host;
      r->bytes = bytes;
      return 0;
    }
  rt_xfer_record *r = (rt_xfer_record *)calloc(1, sizeof(rt_xfer_record));
  snprintf(r->tag, sizeof(r->tag), "%s", tag);
  r->host = host;
  r->bytes = bytes;
  r->next = e->xfers;
  e->xfers = r;
  return 0;
}

/// The allocation itself, returning the SDK error instead of asserting so
/// the cached path can respond to exhaustion by evicting.
static dpu_error_t rt_alloc_raw(int32_t num_dpus, size_t max_blocks_per_dpu,
                                struct dpu_set_t *out) {
  const char *userProfile = getenv("UPMEM_PROFILE");
  char profile[256];
  if (max_blocks_per_dpu > 0) {
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
  return dpu_alloc(num_dpus, profile[0] ? profile : NULL, out);
}

struct dpu_set_t *upmemrt_dpu_alloc_cached(void **slot, int32_t num_dpus,
                                           size_t max_blocks_per_dpu) {
  if (!rt_cache_enabled())
    return upmemrt_dpu_alloc(num_dpus, max_blocks_per_dpu);
  if (*slot) {
    rt_cache_entry *e = rt_entry_of((struct dpu_set_t *)*slot);
    e->live = 1;
    e->lru = ++rt_lru_tick;
    return e->set;
  }
  static int atexit_registered = 0;
  if (!atexit_registered) {
    atexit(rt_evict_all); // release the ranks cleanly at process exit
    atexit_registered = 1;
  }
  struct dpu_set_t *set = (struct dpu_set_t *)malloc(sizeof(struct dpu_set_t));
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  dpu_error_t err;
  while ((err = rt_alloc_raw(num_dpus, max_blocks_per_dpu, set)) != DPU_OK) {
    // Exhausted: evict the least-recently-used set nothing is holding.
    rt_cache_entry *victim = NULL;
    for (rt_cache_entry *e = rt_cache_head; e; e = e->next)
      if (!e->live && (!victim || e->lru < victim->lru))
        victim = e;
    if (!victim)
      DPU_ASSERT(err); // genuinely over-subscribed; fail as alloc would
    rt_evict(victim);
  }
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*set, &nr_dpus);
  upmemrt_record_alloc(upmemrt_now_ns() - t0, nr_dpus);
#endif
  rt_cache_entry *e = (rt_cache_entry *)calloc(1, sizeof(rt_cache_entry));
  e->slot = slot;
  e->set = set;
  e->live = 1;
  e->lru = ++rt_lru_tick;
  e->next = rt_cache_head;
  rt_cache_head = e;
  *slot = set;
  return set;
}

struct dpu_set_t *upmemrt_dpu_alloc(int32_t num_dpus,
                                    size_t max_blocks_per_dpu) {
  struct dpu_set_t *dpu_set =
      (struct dpu_set_t *)malloc(sizeof(struct dpu_set_t));
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  // sgXferEnable/sgXferMaxBlocksPerDpu (set inside rt_alloc_raw) are
  // required for upmemrt_dpu_scatter_blocks/gather_blocks
  // (dpu_push_sg_xfer): scatter/gather transfers are disabled by default,
  // and the max number of blocks per DPU otherwise defaults to the number
  // of DPUs in the set, which is too low once we're scattering one block
  // per tasklet/mram-row. Only set when actually needed: a larger
  // sgXferMaxBlocksPerDpu increases the SDK's memory footprint.
  DPU_ASSERT(rt_alloc_raw(num_dpus, max_blocks_per_dpu, dpu_set));
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
  rt_cache_entry *cached = rt_entry_of(dpu_set);
  if (cached && strcmp(cached->loaded_path, dpu_binary_path) == 0)
    return; // program resident since the last load of this set
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_load(*dpu_set, dpu_binary_path, NULL));
  if (cached) {
    // A (re)load defines a new MRAM layout: whatever transfers were
    // resident are not any more.
    snprintf(cached->loaded_path, sizeof(cached->loaded_path), "%s",
             dpu_binary_path);
    rt_drop_xfers(cached);
  }
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
    (void)i;
    struct dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, i) { dpu_log_read(dpu, stdout); }
  }
  DPU_ASSERT(error);
}

void upmemrt_dpu_free(struct dpu_set_t *void_dpu_set) {
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
  // A cached set is released, not freed: it stays allocated (program and
  // static transfers resident) until its site asks again or the LRU evicts
  // it to make room. Entries exist only when the cache is enabled.
  rt_cache_entry *cached = rt_entry_of(dpu_set);
  if (cached) {
    cached->live = 0;
    return;
  }
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
