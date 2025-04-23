#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
    uint32_t size;
    uint32_t buffer_size;

} dpu_arguments_t;

typedef struct {
    uint32_t t_count;
} dpu_results_t;

bool pred(const int x){
  return (x % 2) == 0;
}

#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

#define BLOCK_DIM 128
// Data type
#define T uint64_t
#define REGS (BLOCK_SIZE >> 3) // 64 bits

#endif
