#include "common.cuh"

#define CUDA_UNFOLD_1D_BLOCK_SIZE 256

void ggml_cuda_op_unfold_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
