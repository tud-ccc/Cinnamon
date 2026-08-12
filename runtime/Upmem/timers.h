#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All stats collection is a no-op unless compiled with -DUPMEM_RT_STATS.

#ifdef UPMEM_RT_STATS

// Public API ─────────────────────────────────────────────────────────────────

// Mark the start of iteration `iter`. All subsequent timed operations are
// tagged with this number until the next call.
void upmemrt_start_stat_collection(int iter);

// Write seven CSV files: {prefix}_scatter.csv, {prefix}_gather.csv,
// {prefix}_launch.csv, {prefix}_free.csv, {prefix}_alloc.csv,
// {prefix}_load.csv, {prefix}_copy.csv, containing all rows collected since
// program start.
void upmemrt_dump_stats(const char *prefix);

// Internal: called by upmem_rt.c / memref_rt.cpp ────────────────────────────

uint64_t upmemrt_now_ns(void);
// `kind` names the transfer op that produced this row, and matches the op's
// mnemonic: "on_array"/"from_array" for upmem.scatter_on_array and
// upmem.gather_from_array (the flat per-DPU memcpy), "blocks" for
// upmem.scatter_blocks/gather_blocks (the UPMEM
// SDK scatter/gather transfer API, dpu_push_sg_xfer), "broadcast" for
// upmem.broadcast. Must be a string literal (or otherwise live for the
// process lifetime): it is stored by pointer, not copied.
// `num_blocks` is the number of per-DPU blocks the transfer was split into --
// only "blocks" can be >1 (the other two always pass 1, one contiguous copy
// per DPU). Not necessarily the tasklet count: a single tasklet's get_block
// callback can be invoked for more than one block (see get_sg_xfer_block in
// upmem_rt.c), so this must come from the call site's own num_blocks, not be
// inferred from NR_TASKLETS.
// `tag` is an optional, user-supplied label (see upmem.timing_tag) identifying
// which MLIR op produced this transfer; NULL if the op carried no tag. Like
// `kind`, must be a string literal (or otherwise live for the process
// lifetime): it is stored by pointer, not copied.
void upmemrt_record_scatter(uint64_t elapsed_ns, size_t bytes_per_dpu,
                            uint32_t num_dpus, size_t num_blocks,
                            const char *kind, const char *tag);
void upmemrt_record_gather(uint64_t elapsed_ns, size_t bytes_per_dpu,
                           uint32_t num_dpus, size_t num_blocks,
                           const char *kind, const char *tag);
void upmemrt_record_launch(uint64_t elapsed_ns, uint32_t num_dpus);
void upmemrt_record_free(uint64_t elapsed_ns, uint32_t num_dpus);
void upmemrt_record_alloc(uint64_t elapsed_ns, uint32_t num_dpus);
// DPU program load (dpu_load), timed separately from allocation: whether a
// load amortizes is a per-occurrence question (once per workload lifetime vs
// once per operator switch -- see the paper's amortizability rule), so it
// must never be folded into the alloc row the analysis blanket-subtracts.
void upmemrt_record_load(uint64_t elapsed_ns, uint32_t num_dpus);
// Host-side memrefCopy calls (e.g. the strided repack copies feeding
// upmem.scatter buffers) -- not a DPU-facing operation, hence no num_dpus.
void upmemrt_record_copy(uint64_t elapsed_ns, size_t bytes);

#else // UPMEM_RT_STATS not defined ─────────────────────────────────────────

static inline void upmemrt_start_stat_collection(int iter) { (void)iter; }
static inline void upmemrt_dump_stats(const char *prefix) { (void)prefix; }

#endif // UPMEM_RT_STATS

#ifdef __cplusplus
}
#endif
