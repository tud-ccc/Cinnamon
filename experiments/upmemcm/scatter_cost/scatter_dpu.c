#include <mram.h>
#include <stdint.h>

// MAX_BLOCKS_PER_DPU (24) * MAX_BLOCK_SIZE (8192): large enough to receive
// any (blocks_per_dpu, block_size) combination used by the host sweep.
// Compiled once and reused for every config; dpu_push_sg_xfer only needs the
// DPU loaded so the SDK can resolve this symbol's MRAM address, no tasklet
// code ever runs. _keep is required: nothing on the DPU side ever reads or
// writes this symbol, so a plain __mram_noinit would be garbage-collected
// by the linker.
__mram_noinit_keep uint8_t buffer[24 * 8192];

int main() { return 0; }
