#include "common.cuh"

#define CUDA_COL2IM_BLOCK_SIZE 256

void ggml_cuda_op_col2im(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
