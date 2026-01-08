#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MRAMToWRAM 1
#define WRAMToMRAM 2

#define assertMsg(cond, ...)                                                   \
  do {                                                                         \
    if (!cond) {                                                               \
      printf("Assertion failed: " __VA_ARGS__);                                \
      abort();                                                                 \
    }                                                                          \
  } while (false)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

void init_tasklet() {
  unsigned int tasklet_id = me();
  if (tasklet_id == 0) {
    mem_reset();
  }
  barrier_wait(&my_barrier);
}

void *dpu_wram_alloc(size_t size) { return mem_alloc(size); }

void loadRow(void *buffer, uint32_t mem_addr, uint32_t offset, int size) {
  mram_read((__mram_ptr void const *)(mem_addr + offset), buffer, size);
}

void storeRow(void *buffer, uint32_t mem_addr, uint32_t offset, int size) {
  mram_write(buffer, (__mram_ptr void *)(mem_addr + offset),
             size * sizeof(int));
}

void dpu_memcpy(int direction, void *wram_addr, uint32_t size,
                uint32_t mram_addr) {
  assertMsg(
      ((intptr_t)wram_addr % 8 == 0),
      "The source or target address in WRAM must be aligned on 8 bytes (%x)\n",
      (uintptr_t)wram_addr);
  assertMsg(
      (mram_addr % 8 == 0),
      "The source or target address in MRAM must be aligned on 8 bytes (%u)\n",
      mram_addr);
  assertMsg(
      (size >= 8 && size <= 2048),
      "The size of the transfer must be at least equal to 8 and not greater "
      "than 2048 but it is %u\n",
      size);
  assertMsg((size % 8 == 0),
            "The size of the transfer must be a multiple of 8 but it is %u\n",
            size);

  if (direction == MRAMToWRAM) {
    mram_read((__mram_ptr void const *)(mram_addr), wram_addr, size);
  } else if (direction == WRAMToMRAM) {
    mram_write(wram_addr, (__mram_ptr void *)(mram_addr), size);
  }
}
