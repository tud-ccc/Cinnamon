#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#define MRAMToWRAM 1
#define WRAMToMRAM 2

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

void loadRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size) {
  mram_read((__mram_ptr void const *)(mem_addr + offset), buffer, size);
}

void storeRow(int *buffer, uint32_t mem_addr, uint32_t offset, int size) {
  mram_write(buffer, (__mram_ptr void *)(mem_addr + offset),
             size * sizeof(int));
}

void dpu_memcpy(int direction, int *buffer, uint32_t size, uint32_t address) {
  if (direction == MRAMToWRAM) {
    mram_read((__mram_ptr void const *)(address), buffer, size);
  } else if (direction == WRAMToMRAM) {
    mram_write(buffer, (__mram_ptr void *)(address), size);
  }
}