#include "common.cuh"

#define CUDA_OPT_STEP_ADAM_BLOCK_SIZE 256

void ggml_cuda_opt_step_adam(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
