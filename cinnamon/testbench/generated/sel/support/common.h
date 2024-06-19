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

#endif
