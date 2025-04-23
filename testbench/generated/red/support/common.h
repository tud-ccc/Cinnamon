#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
    uint32_t size;
    uint32_t buffer_size;
} dpu_arguments_t;

typedef struct {
    uint64_t cycles;
    int output;
} dpu_results_t;

#endif
