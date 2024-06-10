#include "conv-transpose-1d.cuh"

static  __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0,
        const int kernel_size, const int input_size, const int output_size,
        const float * src0, const float * src1,  float * dst) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= output_size) {
        return;
    }

    int upper_bound = idx > input_size-1 ? input_size-1 : idx; //inclusive
    int lower_bound = idx - kernel_size + 1 >= 0 ? idx - kernel_size + 1 : 0;

    int initial_weight_idx = idx > kernel_size -1 ? kernel_size-1 : idx;


    printf("idx: %d initial_weight_idx: %d\n", idx,initial_weight_idx);
    printf("idx: %d upper bound: %d\n", idx, upper_bound);
    printf("idx: %d lower bound: %d\n", idx, lower_bound);


    for (int i = lower_bound; i <= upper_bound; i++)
    {
        dst[idx] += src0[initial_weight_idx-(i-lower_bound)] * src1[i];
    }
    //dst[idx] = 7;
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0,
        const int kernel_size, const int input_size, const int output_size,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(s0,p0,d0,kernel_size, input_size, output_size, src0,src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = 1;//opts[2];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t kernel_size = src0->ne[0];
    const int64_t input_size = src1->ne[0];
    const int64_t output_size = dst->ne[0];


    conv_transpose_1d_f32_f32_cuda( s0,p0,d0,kernel_size, input_size, output_size, src0_d, src1_d, dst_d, stream);
}
