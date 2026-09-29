

#include "upmem_rt.h"
#include "timers.h"
#include <assert.h>
#include <dpu_management.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

// ─── Placement trace (UPMEM_RT_TRACE_PLACEMENT=1) ───────────────────────────
//
// To stderr: for every set allocated, its DPUs, ranks, memory channels and
// sockets; for every transfer site, once, on which NUMA nodes the host pages
// its DPUs' slices start on lie, against each DPU's own rank's node.

static int rt_trace_placement(void) {
  static int enabled = -1;
  if (enabled < 0) {
    const char *v = getenv("UPMEM_RT_TRACE_PLACEMENT");
    enabled = v && v[0] && strcmp(v, "0") != 0;
  }
  return enabled;
}

enum { RT_MAX_RANKS = 256 };

static int rt_channel_of(int rank_id) {
  char path[128];
  snprintf(path, sizeof(path), "/sys/class/dpu_rank/dpu_rank%d/channel_id",
           rank_id);
  FILE *f = fopen(path, "r");
  if (!f)
    return -1;
  int ch = -1;
  if (fscanf(f, "%d", &ch) != 1)
    ch = -1;
  fclose(f);
  return ch;
}

static void rt_trace_set(struct dpu_set_t set, int32_t wanted) {
  struct dpu_set_t dpu;
  uint32_t i, nr_dpus = 0;
  int seen[RT_MAX_RANKS] = {0}, per_channel[RT_MAX_RANKS] = {0};
  int per_node[2] = {0, 0};
  dpu_get_nr_dpus(set, &nr_dpus);
  DPU_FOREACH(set, dpu, i) {
    struct dpu_rank_t *rank = dpu_get_rank(dpu_from_set(dpu));
    int id = dpu_get_rank_id(rank) & 0xFFF;
    if (id < 0 || id >= RT_MAX_RANKS || seen[id]++)
      continue;
    int ch = rt_channel_of(id);
    if (ch >= 0 && ch < RT_MAX_RANKS)
      per_channel[ch]++;
    int node = dpu_get_rank_numa_node(rank);
    if (node == 0 || node == 1)
      per_node[node]++;
  }
  int ranks = 0, channels = 0, busiest = 0;
  for (int r = 0; r < RT_MAX_RANKS; r++) {
    ranks += seen[r] != 0;
    if (per_channel[r]) {
      channels++;
      busiest = per_channel[r] > busiest ? per_channel[r] : busiest;
    }
  }
  fprintf(stderr,
          "[rt placement] alloc %d DPUs: got %u over %d ranks, %d channels "
          "(busiest %d), ranks per socket %d/%d:",
          wanted, nr_dpus, ranks, channels, busiest, per_node[0], per_node[1]);
  for (int r = 0; r < RT_MAX_RANKS; r++)
    if (seen[r])
      fprintf(stderr, " %d(%d DPUs)", r, seen[r]);
  fprintf(stderr, "\n");
}

/// Once per transfer site: where the host pages the DPUs' slices start on
/// lie, local to the DPU's rank, remote, or not present yet.
static void rt_trace_pages(struct dpu_set_t *set, const char *what,
                           const char *tag, void *host, size_t copy_bytes,
                           size_t padding_ratio,
                           size_t (*base_offset)(size_t)) {
  enum { MAX_SITES = 1024 };
  static const char *reported[MAX_SITES];
  static int nr_reported = 0;
  if (!tag)
    tag = "untagged";
  for (int s = 0; s < nr_reported; s++)
    if (reported[s] == tag)
      return;
  if (nr_reported < MAX_SITES)
    reported[nr_reported++] = tag;

  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*set, &nr_dpus);
  void **pages = malloc(nr_dpus * sizeof(void *));
  int *rank_node = malloc(nr_dpus * sizeof(int));
  int *status = malloc(nr_dpus * sizeof(int));
  long page = sysconf(_SC_PAGESIZE);
  struct dpu_set_t dpu;
  size_t i = 0;
  DPU_FOREACH(*set, dpu, i) {
    uintptr_t addr = (uintptr_t)host + base_offset(i) * padding_ratio;
    pages[i] = (void *)(addr & ~(uintptr_t)(page - 1));
    rank_node[i] = dpu_get_rank_numa_node(dpu_get_rank(dpu_from_set(dpu)));
  }
  long rc = syscall(SYS_move_pages, 0, (unsigned long)nr_dpus, pages, NULL,
                    status, 0);
  int local = 0, remote = 0, absent = 0, on_node[2] = {0, 0};
  for (uint32_t d = 0; rc == 0 && d < nr_dpus; d++) {
    if (status[d] < 0) {
      absent++;
      continue;
    }
    if (status[d] == 0 || status[d] == 1)
      on_node[status[d]]++;
    if (status[d] == rank_node[d])
      local++;
    else
      remote++;
  }
  fprintf(stderr,
          "[rt placement] %s %s: %u DPUs x %zu B, host slices local %d, "
          "remote %d, absent %d (pages on node 0/1: %d/%d)%s\n",
          what, tag, nr_dpus, copy_bytes, local, remote, absent, on_node[0],
          on_node[1], rc == 0 ? "" : " -- move_pages failed");
  free(pages);
  free(rank_node);
  free(status);
}

#ifdef ASYNC_TRANSFERS
#define TRANSFER_FLAGS DPU_XFER_ASYNC
#else
#define TRANSFER_FLAGS DPU_XFER_DEFAULT
#endif

/// Whether the `_async` entry points issue asynchronously: unless
/// UPMEM_RT_ASYNC=0, which makes them their synchronous counterparts (timer
/// rows included) and upmemrt_dpu_sync a no-op, so that a binary built with
/// --upmem-async-launches also runs, and is timed, in program order.
static int rt_async_enabled(void) {
  static int enabled = -1;
  if (enabled < 0) {
    const char *v = getenv("UPMEM_RT_ASYNC");
    enabled = !(v && strcmp(v, "0") == 0);
  }
  return enabled;
}

// Residency cache, defined below the transfer functions that consult it.
static int rt_transfer_resident(struct dpu_set_t *set, const char *tag,
                                const char *symbol, size_t symbol_offset,
                                void *host, size_t bytes);

void do_dpu_transfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                     void *host_buffer, size_t copy_bytes, const char *buf_id,
                     size_t symbol_offset, size_t padding_ratio,
                     size_t (*base_offset)(size_t), dpu_xfer_flags_t flags,
                     const char *tag) {
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

  DPU_ASSERT(dpu_push_xfer(*dpu_set, xfer_type, buf_id, symbol_offset,
                           copy_bytes, flags));
  // After the transfer: a gather's pages are only placed by its writes.
  if (rt_trace_placement() && flags != DPU_XFER_ASYNC)
    rt_trace_pages(dpu_set, xfer_type == DPU_XFER_TO_DPU ? "scatter" : "gather",
                   tag, host_buffer, copy_bytes, padding_ratio, base_offset);
}

void upmemrt_dpu_scatter_async(struct dpu_set_t *dpu_set, void *hostBuffer,
                               size_t element_size, size_t num_elements,
                               size_t num_elements_per_tasklet,
                               size_t copy_bytes, const char *bufId,
                               size_t symbol_offset,
                               size_t (*base_offset)(size_t), const char *tag) {
  if (!rt_async_enabled()) {
    upmemrt_dpu_scatter(dpu_set, hostBuffer, element_size, num_elements,
                        num_elements_per_tasklet, copy_bytes, bufId,
                        symbol_offset, base_offset, tag);
    return;
  }
  (void)element_size;
  (void)num_elements;
  (void)num_elements_per_tasklet;
  if (rt_transfer_resident(dpu_set, tag, bufId, symbol_offset, hostBuffer,
                           copy_bytes))
    return;
  do_dpu_transfer(DPU_XFER_TO_DPU, dpu_set, hostBuffer, copy_bytes, bufId,
                  symbol_offset, 1, base_offset, DPU_XFER_ASYNC, tag);
}

void upmemrt_dpu_scatter(struct dpu_set_t *dpu_set, void *hostBuffer,
                         size_t element_size, size_t num_elements,
                         size_t num_elements_per_tasklet, size_t copy_bytes,
                         const char *bufId, size_t symbol_offset,
                         size_t (*base_offset)(size_t), const char *tag) {
  (void)element_size;
  (void)num_elements;
  (void)num_elements_per_tasklet;
  if (rt_transfer_resident(dpu_set, tag, bufId, symbol_offset, hostBuffer,
                           copy_bytes))
    return;
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  do_dpu_transfer(DPU_XFER_TO_DPU, dpu_set, hostBuffer, copy_bytes, bufId,
                  symbol_offset, 1, base_offset, TRANSFER_FLAGS, tag);
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
                        const char *bufid, size_t symbol_offset,
                        size_t (*base_offset)(size_t), const char *tag) {
  (void)num_elements_per_tasklet;
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  if (num_elements * element_size >= 8) {
    do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, copy_bytes, bufid,
                    symbol_offset, 1, base_offset, TRANSFER_FLAGS, tag);
  } else {
    void *padded_result =
        malloc(num_elements * element_size * (8 / element_size));
    do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, padded_result, copy_bytes,
                    bufid, symbol_offset, 8 / element_size, base_offset,
                    TRANSFER_FLAGS, tag);
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

void upmemrt_dpu_gather_async(struct dpu_set_t *dpu_set, void *host_buffer,
                              size_t element_size, size_t num_elements,
                              size_t num_elements_per_tasklet,
                              size_t copy_bytes, const char *bufid,
                              size_t symbol_offset,
                              size_t (*base_offset)(size_t), const char *tag) {
  // Under 8 bytes the result arrives padded and is unpadded on the host,
  // which needs it there: that path waits for the set and gathers in place.
  if (num_elements * element_size < 8 || !rt_async_enabled()) {
    DPU_ASSERT(dpu_sync(*dpu_set));
    upmemrt_dpu_gather(dpu_set, host_buffer, element_size, num_elements,
                       num_elements_per_tasklet, copy_bytes, bufid,
                       symbol_offset, base_offset, tag);
    return;
  }
  do_dpu_transfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, copy_bytes, bufid,
                  symbol_offset, 1, base_offset, DPU_XFER_ASYNC, tag);
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
  /// Padded slots: every blocks_per_slot blocks are followed on the DPU by
  /// slot_padding_bytes that no host block owns. 0 when the blocks are
  /// packed.
  size_t blocks_per_slot;
  size_t slot_padding_bytes;
} sg_xfer_context;

/// Where a slot's padding goes to and comes from: the DPU side of the
/// transfer is one contiguous run, so the padding needs a host address too.
/// What a gather leaves here is never read; a scatter sends its zeros.
static uint8_t sg_padding_scratch[64];

static bool get_sg_xfer_block(struct sg_block_info *out, uint32_t dpu_index,
                              uint32_t block_index, void *args) {
  const sg_xfer_context *ctx = (const sg_xfer_context *)args;
  size_t block = block_index;
  if (ctx->slot_padding_bytes) {
    // Entry i of a slot is its i-th block, and the last entry its padding.
    size_t per_slot = ctx->blocks_per_slot + 1;
    size_t slot = block_index / per_slot, within = block_index % per_slot;
    if (slot >= ctx->num_blocks / ctx->blocks_per_slot)
      return false;
    if (within == ctx->blocks_per_slot) {
      out->addr = sg_padding_scratch;
      out->length = ctx->slot_padding_bytes;
      return true;
    }
    block = slot * ctx->blocks_per_slot + within;
  } else if (block_index >= ctx->num_blocks) {
    return false;
  }

  out->addr = ctx->host_buffer + ctx->base_offset(dpu_index, block);
  out->length = ctx->block_num_elements * ctx->element_size;
  return true;
}

/// Both directions of the scatter/gather transfer API; only the xfer_type and
/// which CSV the timing lands in differ.
static void do_sg_xfer(dpu_xfer_t xfer_type, struct dpu_set_t *dpu_set,
                       void *host_buffer, size_t element_size,
                       size_t num_blocks, size_t block_num_elements,
                       const char *buffer_id, size_t symbol_offset,
                       size_t (*base_offset)(size_t, size_t), const char *tag,
                       size_t blocks_per_slot, size_t slot_padding_bytes) {
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  sg_xfer_context ctx = {
      .host_buffer = (uint8_t *)host_buffer,
      .element_size = element_size,
      .num_blocks = num_blocks,
      .block_num_elements = block_num_elements,
      .base_offset = base_offset,
      .blocks_per_slot = blocks_per_slot,
      .slot_padding_bytes = slot_padding_bytes,
  };
  assert(slot_padding_bytes <= sizeof(sg_padding_scratch) &&
         (!slot_padding_bytes ||
          (blocks_per_slot > 0 && num_blocks % blocks_per_slot == 0)));
  get_block_t get_block_info = {
      .f = get_sg_xfer_block, .args = &ctx, .args_size = sizeof(ctx)};

  size_t length = num_blocks * block_num_elements * element_size;
  if (slot_padding_bytes)
    length += num_blocks / blocks_per_slot * slot_padding_bytes;
  DPU_ASSERT(dpu_push_sg_xfer(*dpu_set, xfer_type, buffer_id, symbol_offset,
                              length, &get_block_info, DPU_SG_XFER_DEFAULT));
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
                                const char *buffer_id, size_t symbol_offset,
                                size_t (*base_offset)(size_t, size_t),
                                const char *tag) {
  if (rt_transfer_resident(dpu_set, tag, buffer_id, symbol_offset, host_buffer,
                           num_blocks * block_num_elements * element_size))
    return;
  do_sg_xfer(DPU_XFER_TO_DPU, dpu_set, host_buffer, element_size, num_blocks,
             block_num_elements, buffer_id, symbol_offset, base_offset, tag, 0,
             0);
}

void upmemrt_dpu_gather_blocks(struct dpu_set_t *dpu_set, void *host_buffer,
                               size_t element_size, size_t num_blocks,
                               size_t block_num_elements, const char *buffer_id,
                               size_t symbol_offset,
                               size_t (*base_offset)(size_t, size_t),
                               const char *tag) {
  do_sg_xfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, element_size, num_blocks,
             block_num_elements, buffer_id, symbol_offset, base_offset, tag, 0,
             0);
}

void upmemrt_dpu_scatter_blocks_padded(
    struct dpu_set_t *dpu_set, void *host_buffer, size_t element_size,
    size_t num_blocks, size_t block_num_elements, const char *buffer_id,
    size_t symbol_offset, size_t (*base_offset)(size_t, size_t),
    const char *tag, size_t blocks_per_slot, size_t slot_padding_bytes) {
  // Not a candidate for residency: a padded buffer is a per-tasklet output.
  do_sg_xfer(DPU_XFER_TO_DPU, dpu_set, host_buffer, element_size, num_blocks,
             block_num_elements, buffer_id, symbol_offset, base_offset, tag,
             blocks_per_slot, slot_padding_bytes);
}

void upmemrt_dpu_gather_blocks_padded(
    struct dpu_set_t *dpu_set, void *host_buffer, size_t element_size,
    size_t num_blocks, size_t block_num_elements, const char *buffer_id,
    size_t symbol_offset, size_t (*base_offset)(size_t, size_t),
    const char *tag, size_t blocks_per_slot, size_t slot_padding_bytes) {
  do_sg_xfer(DPU_XFER_FROM_DPU, dpu_set, host_buffer, element_size, num_blocks,
             block_num_elements, buffer_id, symbol_offset, base_offset, tag,
             blocks_per_slot, slot_padding_bytes);
}

void upmemrt_dpu_broadcast_async(struct dpu_set_t *dpu_set, void *host_buffer,
                                 size_t copy_bytes, const char *buffer_id,
                                 size_t symbol_offset, const char *tag) {
  if (!rt_async_enabled()) {
    upmemrt_dpu_broadcast(dpu_set, host_buffer, copy_bytes, buffer_id,
                          symbol_offset, tag);
    return;
  }
  if (rt_transfer_resident(dpu_set, tag, buffer_id, symbol_offset, host_buffer,
                           copy_bytes))
    return;
  DPU_ASSERT(dpu_broadcast_to(*dpu_set, buffer_id, symbol_offset, host_buffer,
                              copy_bytes, DPU_XFER_ASYNC));
}

void upmemrt_dpu_broadcast(struct dpu_set_t *dpu_set, void *host_buffer,
                           size_t copy_bytes, const char *buffer_id,
                           size_t symbol_offset, const char *tag) {
  if (rt_transfer_resident(dpu_set, tag, buffer_id, symbol_offset, host_buffer,
                           copy_bytes))
    return;
#ifdef UPMEM_RT_STATS
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_broadcast_to(*dpu_set, buffer_id, symbol_offset, host_buffer,
                              copy_bytes, TRANSFER_FLAGS));
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
/// the MRAM symbol it occupies, and the payload it holds.
typedef struct rt_xfer_record {
  char tag[64];
  char symbol[64];
  size_t symbol_offset; // the slot: where in the symbol the payload sits
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

/// The cache flag for the other runtime translation units (the static-repack
/// skip in memref_rt.cpp lives outside this file).
int upmemrt_cache_enabled(void) { return rt_cache_enabled(); }

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

/// Really free a cached set: everything it held resident is gone. Returns
/// the nanoseconds the SDK free took (0 in non-stats builds) so a caller
/// timing its own SDK work -- alloc_cached evicting mid-allocation -- can
/// keep that span out of its record; it is already recorded as a free here,
/// and counting it in both rows would net it out of the total twice.
static uint64_t rt_evict(rt_cache_entry *victim) {
  uint64_t elapsed = 0;
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*victim->set, &nr_dpus);
  uint64_t t0 = upmemrt_now_ns();
#endif
  DPU_ASSERT(dpu_sync(*victim->set)); // settle what was issued on it
  DPU_ASSERT(dpu_free(*victim->set));
#ifdef UPMEM_RT_STATS
  elapsed = upmemrt_now_ns() - t0;
  upmemrt_record_free(elapsed, nr_dpus);
#endif
  free(victim->set);
  rt_drop_xfers(victim);
  *victim->slot = NULL;
  rt_cache_entry **link = &rt_cache_head;
  while (*link != victim)
    link = &(*link)->next;
  *link = victim->next;
  free(victim);
  return elapsed;
}

static void rt_evict_all(void) {
  while (rt_cache_head)
    rt_evict(rt_cache_head);
}

/// The transfer-side half of the cache: true when this static transfer's
/// payload is already resident on the set, in which case the caller skips
/// it. Recording happens here too, so the caller only ever asks.
///
/// A resident payload is a promise the compiler made: the graph allocation
/// planned MRAM for every member's static operand, and a site's data is
/// skipped on later inferences because nothing else was to be written over
/// it. So a transfer -- static or not -- into a symbol that another static
/// site holds resident on this set is not a cache miss but wrong code
/// generation (two members given the same slot), and it is refused rather
/// than let the earlier site silently compute on the later one's data. A
/// slotted symbol holds several payloads at distinct offsets, so the
/// occupancy is per (symbol, offset).
static int rt_transfer_resident(struct dpu_set_t *set, const char *tag,
                                const char *symbol, size_t symbol_offset,
                                void *host, size_t bytes) {
  rt_cache_entry *e = rt_entry_of(set);
  if (!e)
    return 0;
  const int isStatic = tag && strncmp(tag, "static:", 7) == 0;
  for (rt_xfer_record *r = e->xfers; r; r = r->next) {
    if (symbol && strncmp(r->symbol, symbol, sizeof(r->symbol)) == 0 &&
        r->symbol_offset == symbol_offset &&
        !(isStatic && strncmp(r->tag, tag, sizeof(r->tag)) == 0)) {
      fprintf(stderr,
              "upmemrt: transfer '%s' writes MRAM symbol '%s' at offset %zu, "
              "which site '%s' holds resident on the same DPU set: the code "
              "generator gave two members one slot\n",
              tag ? tag : "(untagged)", symbol, symbol_offset, r->tag);
      abort();
    }
  }
  if (!isStatic)
    return 0;
  // Occupancy is per slot -- (symbol, offset) -- not per site: one site
  // inside a layer loop fills every slot of its buffer in turn, and each of
  // them stays resident for the next inference.
  for (rt_xfer_record *r = e->xfers; r; r = r->next)
    if (symbol && strncmp(r->symbol, symbol, sizeof(r->symbol)) == 0 &&
        r->symbol_offset == symbol_offset) {
      if (r->host == host && r->bytes == bytes)
        return 1;
      // Same slot, different payload (the host re-materialised its static
      // operand): run the transfer and remember the new occupant.
      r->host = host;
      r->bytes = bytes;
      return 0;
    }
  rt_xfer_record *r = (rt_xfer_record *)calloc(1, sizeof(rt_xfer_record));
  snprintf(r->tag, sizeof(r->tag), "%s", tag);
  snprintf(r->symbol, sizeof(r->symbol), "%s", symbol ? symbol : "");
  r->symbol_offset = symbol_offset;
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
  dpu_error_t err = dpu_alloc(num_dpus, profile[0] ? profile : NULL, out);
  if (err == DPU_OK && rt_trace_placement())
    rt_trace_set(*out, num_dpus);
  return err;
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
  uint64_t evict_ns = 0;
  while ((err = rt_alloc_raw(num_dpus, max_blocks_per_dpu, set)) != DPU_OK) {
    // Exhausted: evict the least-recently-used set nothing is holding.
    rt_cache_entry *victim = NULL;
    for (rt_cache_entry *e = rt_cache_head; e; e = e->next)
      if (!e->live && (!victim || e->lru < victim->lru))
        victim = e;
    if (!victim)
      DPU_ASSERT(err); // genuinely over-subscribed; fail as alloc would
    evict_ns += rt_evict(victim);
  }
#ifdef UPMEM_RT_STATS
  uint32_t nr_dpus = 0;
  dpu_get_nr_dpus(*set, &nr_dpus);
  upmemrt_record_alloc(upmemrt_now_ns() - t0 - evict_ns, nr_dpus);
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
  DPU_ASSERT(dpu_sync(*dpu_set)); // nothing issued may run past the load
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

void upmemrt_dpu_launch_async(struct dpu_set_t *void_dpu_set) {
  if (!rt_async_enabled()) {
    upmemrt_dpu_launch(void_dpu_set);
    return;
  }
  struct dpu_set_t *dpu_set = (struct dpu_set_t *)void_dpu_set;
  DPU_ASSERT(dpu_launch(*dpu_set, DPU_ASYNCHRONOUS));
  if (getenv("UPMEM_LOG")) {
    DPU_ASSERT(dpu_sync(*dpu_set));
    size_t i = 0;
    (void)i;
    struct dpu_set_t dpu;
    DPU_FOREACH(*dpu_set, dpu, i) { dpu_log_read(dpu, stdout); }
  }
}

void upmemrt_dpu_sync(struct dpu_set_t *void_dpu_set) {
  if (!rt_async_enabled())
    return; // everything issued already completed
  DPU_ASSERT(dpu_sync(*(struct dpu_set_t *)void_dpu_set));
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
  DPU_ASSERT(dpu_sync(*dpu_set)); // settle what was issued on it
  DPU_ASSERT(dpu_free(*dpu_set));
#ifdef UPMEM_RT_STATS
  upmemrt_record_free(upmemrt_now_ns() - t0, nr_dpus);
#endif
}
