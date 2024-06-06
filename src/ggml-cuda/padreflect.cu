#include "padreflect.cuh"

static __global__ void pad_reflect_1d_f32(const float * x, float * dst,
        const int nb00, const int nb01,
        const int ne10, const int ne11, const int p0,
        const int p1, const int inp_size, const int dst_size
        ) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ne10 * ne11) {
        return;
    }


    const int row_size = ne10;
    int column_index = index % row_size;
    const int row_index = index / row_size;

    if (column_index < p0)
    {
        column_index =  p0 - column_index;
    }
    else if(column_index < row_size -p1)
    {
        column_index = column_index - p0;
    }
    else
    {
        column_index = (row_size - p1 - p0) - (p1+1-(row_size-column_index)) - 1;
    }

    int i00 = column_index;
    int i01 = row_index;



    dst[index] = *(float *)((char *)x + i01 * nb01 + i00 * nb00);
}

static void pad_reflect_1d_f32_cuda(const float * x, float * dst,
        const int nb00, const int nb01,
        const int ne10, const int ne11, 
        const int p0, const int p1, 
        const int inp_size, const int dst_size,
        cudaStream_t stream) {
    int num_blocks = (dst_size + CUDA_PAD_REFLECT_BLOCK_SIZE - 1) / CUDA_PAD_REFLECT_BLOCK_SIZE;

    pad_reflect_1d_f32<<<num_blocks, CUDA_PAD_REFLECT_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, ne10, ne11,p0,p1, inp_size,dst_size);
}

void ggml_cuda_op_pad_reflect_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int inp_size = src0->ne[0] * src0->ne[1];
    const int dst_size = dst->ne[0] * dst->ne[1];



    pad_reflect_1d_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], dst->ne[0], dst->ne[1], dst->op_params[0],dst->op_params[1], inp_size,dst_size, stream);
}