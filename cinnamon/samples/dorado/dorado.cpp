/* Inference for Llama-2 Transformer model in pure C, based on
 * https://github.com/karpathy/llama2.c/ with modifications to run the inference
 * using cinm */

/*
  MIT License

  Copyright (c) 2023 Andrej

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  uint32_t dim;        // transformer dimension (512)
  uint32_t hidden_dim; // for ffn layers (2048)
  uint32_t n_layers;   // number of layers (18)
  uint32_t n_heads;    // number of query heads (8)
  uint32_t seq_len;    // max sequence length (512)
} Config;

// Struct contains all model weights
typedef struct {
  // weights for rmsnorms
  float *rms1; // (layer, dim) rmsnorm weights
  float *rms2; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wqkv; // (layer, dim, n_heads * head_size * 3)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float *w1; // (layer, hidden_dim, dim)
  float *w2; // (layer, dim, hidden_dim)
  float *upscale; // (dim, dim x2) upscaling layer, must reshape to len x2
  float *crf; // (dim x2, 4096)
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim, seq_len)
  float *xb;     // same, but inside a residual branch (dim, seq_len)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *qkv;    // query, key, value (T, C*3,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *cos_freqs; // Used by RoPE
  float *sin_freqs; // Used by RoPE
  float *upscaled; // buffer for upscaled output before crf (dim, seq_len*2)
  float *crf;    // buffer for crf output (4096, seq_len*2)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  s->x   = (float *)calloc(p->seq_len * p->dim, sizeof(float));
  s->xb  = (float *)calloc(p->seq_len * p->dim, sizeof(float));
  s->hb  = (float *)calloc(p->seq_len * p->hidden_dim * 2, sizeof(float));
  s->qkv = (float *)calloc(p->seq_len * p->dim * 3, sizeof(float));
  s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
  s->upscaled = (float *)calloc(p->dim * p->seq_len * 2, sizeof(float));
  s->crf = (float *)calloc(4096 * p->seq_len * 2, sizeof(float)); // TODO: check dim
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->hb || !s->qkv || !s->att || !s->upscaled || !s->crf) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->hb);
  free(s->qkv);
  free(s->att);
  free(s->upscaled);
  free(s->crf);
}

void create_rope_freqs(RunState *s, Config *p) {
  const float theta = 10000.0f;
  const int64_t max_seq_len = 2048;

  double *vec = (double *)calloc(p->dim / 2, sizeof(double));
  if (!vec) {fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE);}

  for (int i = 0; i < p->dim / 2; i++) {
    double a = i * 2;
    vec[i] = 1.0 / std::pow(static_cast<double>(theta), a / static_cast<double>(p->dim));
  }

  // torch::arange(max_seq_len).reshape({max_seq_len, 1, 1, 1}) * inv_freq [< invfreq is vec]
  // Causes broadcasting semantics resulting in {max_seq_len, 1, 1, dim/2} or in our case {max_seq_len, dim/2}
  float *cos_freqs = (float *)calloc(max_seq_len * p->dim / 2, sizeof(float));
  float *sin_freqs = (float *)calloc(max_seq_len * p->dim / 2, sizeof(float));
  if (!cos_freqs || !sin_freqs) {fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE);}
  for (int i = 0; i < max_seq_len; i++) {
    for (int j = 0; j < p->dim / 2; j++) {
      cos_freqs[i * p->dim / 2 + j] = (float) std::cos(i * vec[j]);
      sin_freqs[i * p->dim / 2 + j] = (float) std::sin(i * vec[j]);
    }
  }

  s->cos_freqs = cos_freqs;
  s->sin_freqs = sin_freqs;
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, ssize_t file_size) {
  float *ptr_orig = ptr;
  // make sure the multiplications below are done in 64bit to fit the parameter
  // counts of 13B+ models
  uint64_t n_layers = p->n_layers;

  w->rms1 = ptr;
  ptr += n_layers * p->dim; // pointer arithmetic automatically multiplies sizeof(float)
  w->rms2 = ptr;
  ptr += n_layers * p->dim;

  w->wqkv = ptr;
  ptr += n_layers * p->dim * p->dim * 3;
  w->wo = ptr;
  ptr += n_layers * p->dim * p->dim;

  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim * 2;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;

  w->upscale = ptr;
  ptr += p->dim * p->dim * 2; // looks like in>out is 512>1024 but it's reshaped to 512 with seq_len*2
  w->crf = ptr;
  ptr += p->dim * 4096;

  if (ptr > ptr_orig + file_size / 4) {
    fprintf(stderr, "Too many bytes read!\nFile size: %lu kb; bytes read: %lu kb\n", file_size / 4 / 1024, (ptr - ptr_orig) / 1024);
  } else if (ptr < ptr_orig + file_size / 4) {
    fprintf(stderr, "Too few bytes read!\nFile size: %lu kb; bytes read: %lu kb\n", file_size / 4 / 1024, (ptr - ptr_orig) / 1024);
  }
}

void read_checkpoint(char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, float **data,
                     ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  uint32_t *magic_and_version = (uint32_t *) calloc(2, sizeof(uint32_t));
  if (fread(magic_and_version, sizeof(uint32_t), 2, file) != 2) {
    fprintf(stderr, "Couldn't read magic number and version from file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  if ((magic_and_version[0] != 0x616b3432) || (magic_and_version[1] != 1)) {
    fprintf(stderr, "Wrong magic number or incompatible version detected in file %s.\nMagic: %u; version: %d", checkpoint, magic_and_version[0], magic_and_version[1]);
  }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }

  printf("dim: %d\n", config->dim);
  printf("hidden_dim: %d\n", config->hidden_dim);
  printf("n_layers: %d\n", config->n_layers);
  printf("n_heads: %d\n", config->n_heads);
  printf("seq_len: %d\n", config->seq_len);

  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    fprintf(stderr, "%s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  float *weights_ptr = *data + 256; // Fixed 256 byte header+padding
  memory_map_weights(weights, config, weights_ptr, *file_size - 256);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data,
                  &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
  create_rope_freqs(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
extern "C" {
// actual forward() is implemented in mlir
/* func.func @forward(%token : index, %pos : index,
    // state
    %key_cache : tensor<6x256x288xf32>,
    %value_cache : tensor<6x256x288xf32>,
    // weights
    %embedding_table : tensor<32000x288xf32>,
    %rms_att_weights : tensor<6x288xf32>,
    %wq : tensor<6x288x288xf32>,
    %wk : tensor<6x288x288xf32>,
    %wv : tensor<6x288x288xf32>,
    %wo : tensor<6x288x288xf32>,
    %w1 : tensor<6x768x288xf32>,
    %w2 : tensor<6x288x768xf32>,
    %w3 : tensor<6x768x288xf32>,
    %rms_ffn_weights : tensor<6x288xf32>,
    %rms_final_weight : tensor<288xf32>,
    %wcls : tensor<32000x288xf32>
) -> (tensor<32000xf32>, tensor<6x256x288xf32>, tensor<6x256x288xf32>) {
*/

float *forward(uint64_t index, uint64_t pos, float *key_cache, float *value_cache, float *embedding_table,
               const float *rms_att_weights, const float *wq, const float *wk, const float *wv, const float *wo,
               const float *w1, const float *w2, const float *w3, const float *rms_ffn_weights,
               const float *rms_final_weight, const float *wcls);

float *mha(const float *q, const float *kc, const float *vc, int64_t pos);

// v: tensor<512xf32>; w: tensor<512xf32>.
float *rmsnorm(const float *v, const float *w);
// v: tensor<262144xf32>; w: tensor<262144xf32>.
float *rmsnorm_large(const float *v, const float *w);
// v: tensor<512x512xf322> (seq_len x dim); w: tensor<512xf32> (dim).
float *rmsnorm_batched(const float *v, const float *w);
float *softmax(const float *x);
}

// Matrix-vector multiplication: W @ x -> xout.
// xout: pointer to output data (d,);
// x: input data vector (n,);
// w: weight matrix (d, n);
void matmul(float *xout, float *x, float *w, int A_cols, int A_rows) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < A_rows; i++) {
    float val = 0.0f;
    for (int j = 0; j < A_cols; j++) {
      val += w[i * A_cols + j] * x[j];
    }
    xout[i] = val;
  }
}

// Matrix-matrix multiplication: A @ B -> C
// A: (A_rows x A_cols);
// B: (A_cols x B_cols);
// C: (A_rows x B_cols).
static void gemm(float *A, float *B, float *C, const int64_t A_rows, const int64_t A_cols, const int64_t B_cols,
                 const int64_t A_rows_offset = 0, const int64_t A_cols_offset = 0, const int64_t B_cols_offset = 0,
                 int64_t A_rows_end = -1, int64_t A_cols_end = -1, int64_t B_cols_end = -1) {
  //printf("GEMM   with A_rows = %ld, A_cols = %ld, B_cols = %ld\n", A_rows, A_cols, B_cols);
  if (A_rows_end == -1) A_rows_end = A_rows;
  if (A_cols_end == -1) A_cols_end = A_cols;
  if (B_cols_end == -1) B_cols_end = B_cols;
  assert(A_rows_end <= A_rows);
  assert(A_cols_end <= A_cols);
  assert(B_cols_end <= B_cols);

  int64_t i, k, j;
  float temp;
#pragma omp parallel for default(shared) private(i, k, j, temp)
  for (i = A_rows_offset; i < A_rows_end; i++) {
    for (k = A_cols_offset; k < A_cols_end; k++) {
      temp = A[i * A_cols + k];
      for (j = B_cols_offset; j < B_cols_end; j++) {
        C[i * B_cols + j] += temp * B[k * B_cols + j];
      }
    }
  }
}

// Matrix-matrix multiplication transposing B: A @ B^T -> C.
// A: (A_rows x A_cols);
// B: (B_rows x A_cols);
// C: (A_rows x B_rows).
static void gemm_t(float *A, float *B, float *C, const int64_t A_rows, const int64_t A_cols, const int64_t B_rows,
                   const int64_t A_rows_offset = 0, const int64_t A_cols_offset = 0, const int64_t B_rows_offset = 0,
                   int64_t A_rows_end = -1, int64_t A_cols_end = -1, int64_t B_rows_end = -1) {
  //printf("GEMM_T with A_rows = %ld, A_cols = %ld, B_rows = %ld\n", A_rows, A_cols, B_rows);
  if (A_rows_end == -1) A_rows_end = A_rows;
  if (A_cols_end == -1) A_cols_end = A_cols;
  if (B_rows_end == -1) B_rows_end = B_rows;
  assert(A_rows_end <= A_rows);
  assert(A_cols_end <= A_cols);
  assert(B_rows_end <= B_rows);

  int64_t i, k, j;
  float temp;
#pragma omp parallel for default(shared) private(i, k, j, temp)
  for (i = A_rows_offset; i < A_rows_end; i++) {
    for (k = A_cols_offset; k < A_cols_end; k++) {
      temp = A[i * A_cols + k];
      for (j = B_rows_offset; j < B_rows_end; j++) {
        // For inserting into submatrix of original size matrix
        C[i * B_rows + j] += temp * B[k + j * B_rows];
        // For outputting to perfectly sized smaller matrix
//        C[(i - A_rows_offset) * (B_rows_end - B_rows_offset) + j - B_rows_offset] += temp * B[k + j * B_rows];
      }
    }
  }
}

void rmsnorm_cpu(float *out, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    out[j] = weight[j] * (ss * x[j]);
  }
}

// TODO: add support for extra dim (sequence length) and ensure RMS is only taken and used along that dim.
void rmsnorm_upmem(float *out, float *x, float *weight, int size) {
  float *r = rmsnorm(x, weight);
  memcpy(out, r, size * sizeof(float));
  free(r);
}

void rmsnorm_upmem_large(float *out, float *x, float *weight, int size, int extra_dim_size = 1) {
  float *r = rmsnorm_large(x, weight);
  memcpy(out, r, size * extra_dim_size * sizeof(float));
  free(r);
}

void rmsnorm_upmem_batched(float *out, float *x, float *weight, int size, int extra_dim_size = 1) {
  float *r = rmsnorm_batched(x, weight);
  memcpy(out, r, size * extra_dim_size * sizeof(float));
  free(r);
}

void softmax_cpu(float *x, int size, int extra_dim_size = 1) {
  for (int dim = 0; dim < extra_dim_size; dim++) {
    // find max value (for numerical stability)
    float max_val = x[0 + dim * size];
    for (int i = 1; i < size; i++) {
      if (x[i + dim * size] > max_val) {
        max_val = x[i + dim * size];
      }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      x[i + dim * size] = expf(x[i + dim * size] - max_val);
      sum += x[i + dim * size];
    }
    // normalize
    for (int i = 0; i < size; i++) {
      x[i + dim * size] /= sum;
    }
  }
}

// TODO: add support for extra dim in CINM kernel and ensure max is only taken and normalized along that dim.
void softmax_upmem(float *x, int size, int extra_dim_size = 1) {
  for (int dim = 0; dim < extra_dim_size; dim++) {
    float *r = softmax(x);
    memcpy(x + dim * size, r, size * sizeof(float));
    free(r);
  }
}

// x: output; q/k/v: inputs; qb/qe: query begin/end indices; kvb/kve: key/value begin/end indices.
void scaled_dot_product_attention(float *x, float *q, float *k, float *v, int qb, int qe, int kvb, int kve, int seq_len, int heads, int head_size) {
  // q/k/v is NHTD (batch size, heads, seq_len, head size)
  float *matmul_qk = (float *) malloc(seq_len * seq_len * sizeof(float));

  float sqrt_d_k = std::sqrt(head_size);

  for (int h = 0; h < heads; h++) {
    memset(matmul_qk, 0, seq_len * seq_len * sizeof(float));
    gemm_t(q + h * head_size, k + h * head_size, matmul_qk,
           seq_len, head_size, seq_len,
           qb, 0, kvb,
           qe, head_size, kve);

    for (int i = 0; i < seq_len * seq_len; i++) {
      matmul_qk[i] /= sqrt_d_k;
    }

    // Performs softmax along last dim only
    softmax_cpu(matmul_qk, seq_len, seq_len);

    // x.slice(-2 (T), qb, qe)
    gemm(matmul_qk, v + h * head_size, x + h * head_size + qb,
         seq_len, seq_len, head_size,
         qb, kvb, 0,
         qe, kve, head_size);
  }
}

void mha_cpu(float *x, float *att, float *qkv, int seq_len, int num_heads, int head_size) {
  int dim = num_heads * head_size;
  float *q = qkv;
  float *k = qkv + dim * seq_len;
  float *v = qkv + dim * seq_len * 2;

  int win_upper = 128, win_lower = 127; // FIXME: get from config
  int num_splits = 8;                   // FIXME: get from config
  int elems_per_split = (seq_len + (num_splits - 1)) / num_splits; // round up. TODO: does not pad to 4

  for (int i = 0; i < num_splits; i++) {
    int qb = i * elems_per_split;
    if (qb >= seq_len) {
      break;
    }

    int qe = std::min(seq_len, qb + elems_per_split);
    int kvb = std::max(0, qb - win_lower);
    int kve = std::min(seq_len, qe + win_upper);
    //printf("qb %d, qe %d, kvb %d, kve %d\n", qb, qe, kvb, kve);
    // qkv is now supposed to be 3NHTD ({Q|K|V}, batch size, heads, seq_len, head size)

    // x.slice(-2 {seq_len dim}, qb, qe) =
    scaled_dot_product_attention(x, q, k, v, qb, qe, kvb, kve, seq_len, num_heads, head_size);
  }
}

void mha_upmem(float *x, float *att, float *q, float *kc, float *vc,
               int64_t pos) {
  float *r = mha(q, kc, vc, pos);
  memcpy(x, r, 288 * sizeof(float));
  free(r);
}

/////-----------------------
///// FORWARD
/////-----------------------
//
// Input dims: (T, D) -> (sequence length, dimensions) -> (512, 512)
float *forward2(Transformer *transformer, float *input) {
  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  int head_size = p->dim / p->n_heads;
  float deepnorm_alpha = 2.4494897f; // TODO: from config

  // TODO: implement convs

  // forward all the layers
  for (unsigned int l = 0; l < p->n_layers; l++) {
    if (l == 0) {
      for (int i = 0; i < p->dim * p->seq_len; i++) {
        s->x[i] = input[i];
      }
    }
    printf("Layer %d of %d. x[0] = %f\n", l + 1, p->n_layers, s->x[0]);

    // QKV matmul
    if (l == 0) printf("  QKV...\n");
    gemm(w->wqkv + l * p->dim * p->dim * 3, s->x, s->qkv, // A, B, C
         p->dim * 3, p->dim, p->seq_len); // dims

    // RoPE relative positional encoding: complex-valued rotate q and k in each head (ROPE)
    if (l == 0) printf("  ROPE...      x[0] = %f\n", s->qkv[0]);
    for (int i = 0; i < p->seq_len; i++) {
      for (int j = 0; j < p->dim; j += 2) {
        int head_dim = j % head_size;
        float cos_factor = s->cos_freqs[i * p->dim / 2 + j];
        float sin_factor = s->sin_freqs[i * p->dim / 2 + j];

        float v0 = s->qkv[i * p->dim * 3 + j];
        float v1 = s->qkv[i * p->dim * 3 + j + 1];
        s->qkv[i * p->dim * 3 + j] = v0 * cos_factor - v1 * sin_factor;
        s->qkv[i * p->dim * 3 + j + 1] = v0 * sin_factor + v1 * cos_factor;
      }
    }

    // MHA
    if (l == 0) printf("  MHA...       x[0] = %f\n", s->qkv[0]);
    mha_cpu(s->xb, s->att, s->qkv, p->seq_len, p->n_heads, head_size);

    // final matmul to get the output of the attention
    if (l == 0) printf("  OUTPROJ...   x[0] = %f\n", s->x[0]);
    gemm(w->wo + l * p->dim * p->dim, s->xb, s->x, // A, B, C
         p->dim, p->dim, p->seq_len); // dims

    // residual connection back into x
    for (int i = 0; i < p->seq_len * p->dim; i++) {
      s->x[i] += s->xb[i] * deepnorm_alpha;
    }

    // Attention rmsnorm (NORM1)
    if (l == 0) printf("  RES&NORM1... x[0] = %f\n", s->xb[0]);
    for (int i = 0; i < p->seq_len; i++) {
      rmsnorm_cpu(s->x + i * p->dim, s->xb + i * p->dim, w->rms1 + l * p->dim, p->dim);
    }

    // Gated MLP

    // In PyTorch:
    // y = fc1(x)
    // y, gate = y.chunk(2 parts, dim=-1 (last dim))
    // y = swiglu(y, gate)
    // y = fc2(y)
    // return y

    if (l == 0) printf("  FC1...       x[0] = %f\n", s->x[0]);
    gemm(w->w1 + l * p->dim * p->hidden_dim * 2, s->x, s->hb, p->hidden_dim * 2, p->dim, p->seq_len);

    if (l == 0) printf("  SWIGLU...    x[0] = %f\n", s->hb[0]);
    for (int i = 0; i < p->seq_len; i++) {
      // SwiGLU non-linearity
      for (int j = 0; j < p->hidden_dim; j++) {
        float val = s->hb[i * p->hidden_dim * 2 + j];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        val *= s->hb[i * p->hidden_dim * 2 + p->hidden_dim + i];
        s->hb[i * p->hidden_dim * 2 + i] = val;
      }
    }

    if (l == 0) printf("  FC2...       x[0] = %f\n", s->hb[0]);
    for (int i = 0; i < p->seq_len; i++) {
      // final matmul to get the output of the ffn
      matmul(s->xb + i * p->dim,
             s->hb + i * p->hidden_dim * 2,
             w->w2 + l * p->dim * p->hidden_dim,
             p->hidden_dim, p->dim);
    }

    // residual connection
    for (int i = 0; i < p->seq_len * p->dim; i++) {
      s->xb[i] += s->x[i] * deepnorm_alpha;
    }

    // FF rmsnorm (NORM2)
    if (l == 0) printf("  RES&NORM2... x[0] = %f\n", s->xb[0]);
    rmsnorm_upmem_batched(s->x, s->xb, w->rms2 + l * p->dim, p->dim, p->seq_len);
  }

  // upscale layer and reshape (implicit)
  printf("Upscale layer\n");
  gemm(w->upscale, s->x, s->upscaled, p->dim * 2, p->dim, p->seq_len);

  // TODO: check if indexing is correct after upscaling.
  printf("CRF layer\n");
  gemm(w->crf, s->upscaled, s->crf, 4096, p->dim, p->seq_len * 2);

  return s->crf;
}

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> <todo add pod5 file>\n");
  fprintf(stderr, "Example: run model.bin\n");
  exit(EXIT_FAILURE);
}

#define benchmark(name, num_runs, body)                                        \
  do {                                                                         \
    {body};                                                                    \
    {body};                                                                    \
    clock_t duration = 0;                                                      \
    for (size_t i = 0; i < num_runs; i++) {                                    \
      clock_t start = clock();                                                 \
      {body};                                                                  \
      duration += clock() - start;                                             \
    }                                                                          \
    printf(name ": %fms\n", (double)(duration) * 1000.0 /                      \
                                (double)CLOCKS_PER_SEC / (double)num_runs);    \
  } while (false)

extern "C" void *upmemrt_dpu_alloc(int32_t num_ranks, int32_t num_dpus,
                                   const char *dpu_binary_path);

extern "C" void upmemrt_dpu_free(void *dpu_set);

void run_benchmarks() {
  float *a = (float *)malloc(262144 * sizeof(float));
  float *b = (float *)malloc(262144 * sizeof(float));

  float *q = (float *)malloc(32768 * sizeof(float));
  float *kc = (float *)malloc(1024 * 32768 * sizeof(float));
  float *vc = (float *)malloc(1024 * 32768 * sizeof(float));
  float *qkv = (float *)malloc(3 * 1024 * 32768 * sizeof(float));
  float *att = (float *)malloc(1024 * sizeof(float));

  float *r_cpu = (float *)malloc(262144 * sizeof(float));
  float *r_upmem;
  unsigned long long rng = 1234;
  for (size_t i = 0; i < 262144; i++) {
    a[i] = random_f32(&rng);
    b[i] = random_f32(&rng);
  }

  benchmark("vector add cpu", 32, {
    for (size_t i = 0; i < 262144; i++) {
      r_cpu[i] = a[i] + b[i];
    }
  });

  benchmark("softmax cpu", 32, { softmax_cpu(a, 262144); });
  benchmark("rmsnorm cpu", 32, { rmsnorm_cpu(r_cpu, a, b, 262144); });
  benchmark("mha cpu", 32, { mha_cpu(r_cpu, att, qkv, 1024, 8, 4096); });
}
static void print_matrix(float *A, const int64_t A_rows, const int64_t A_cols) {

  int64_t i, j;
  printf("[");
  for (i = 0; i < A_rows; ++i) {
    for (j = 0; j < A_cols; ++j) {
      printf("%f, ", A[i * A_cols + j]);
    }
    printf("\n");
  }
  printf("]\n");
}
int main(int argc, char *argv[]) {
  if (getenv("BENCHMARK")) {
    run_benchmarks();
    return 0;
  }

  {
    size_t seq_len = 512;
    size_t dim_size = 512;
    float *sample_input = (float *)malloc(seq_len * dim_size * sizeof(float));
    float *sample_weights = (float *)malloc(dim_size * sizeof(float));
    float *sample_weights_large = (float *)malloc(seq_len * dim_size * sizeof(float));
    unsigned long long rng = time_in_ms();
    for (size_t i = 0; i < seq_len * dim_size; i++) {
      sample_input[i] = random_f32(&rng);
    }
    for (size_t i = 0; i < dim_size; i++) {
      sample_weights[i] = random_f32(&rng);
    }
    float *out1 = (float *)malloc(dim_size * sizeof(float));
    float *out2 = (float *)malloc(seq_len * dim_size * sizeof(float));
    printf("Comparing rmsnorm upmem (un)batched...\n");
    printf("Running rmsnorm upmem...\n");
    rmsnorm_upmem(out1, sample_input, sample_weights, dim_size);
    printf("Running rmsnorm upmem large...\n");
    rmsnorm_upmem_large(out2, sample_input, sample_weights_large, dim_size, seq_len);
    printf("Running rmsnorm upmem batched...\n");
    rmsnorm_upmem_batched(out2, sample_input, sample_weights, dim_size, seq_len);
    printf("Comparing...\n");
    for (size_t i = 0; i < dim_size; i++) {
      if (fabsf(out1[i] - out2[i]) > 0.00001f) {
        printf("[ERROR] %f != %f\n", out1[i], out2[i]);
        break;
      }
    }
    free(sample_input);
    free(sample_weights);
    free(out1);
    free(out2);
  }

  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  const char *mode = "basecall";

  // poor man's C argparse so we can override the defaults above from the
  // command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);

  float *sample_input = (float *)malloc(transformer.config.seq_len * transformer.config.dim * sizeof(float));
  unsigned long long rng = time_in_ms();
  for (size_t i = 0; i < transformer.config.seq_len * transformer.config.dim; i++) {
    sample_input[i] = random_f32(&rng);
  }
  // run!
  if (strcmp(mode, "basecall") == 0) {
    printf("Running forward pass. x[0] = %f\n", sample_input[0]);
    forward2(&transformer, sample_input);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_transformer(&transformer);
  return 0;
}
#endif
