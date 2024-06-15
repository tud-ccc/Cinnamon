
#ifndef COMMON_H_
#define COMMON_H_
#include <stdint.h>

#define T uint32_t


#define ITERATION 1

typedef struct {
    uint32_t cycles;
} dpu_result_t;

typedef struct {
    dpu_result_t tasklet_result[NR_TASKLETS];
} dpu_results_t;

typedef struct {
    uint32_t m_size;
    uint32_t n_size;
    uint32_t q_size;
} dpu_arguments_t;

// Specific information for each DPU
struct dpu_info_t {
    uint32_t m_size;
    uint32_t n_size;
    uint32_t q_size;
};
struct dpu_info_t *dpu_info;

#endif // COMMON_H_