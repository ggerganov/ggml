#include "upscale.cuh"

static __global__ void upscale_f32(const float * x, float * dst, const int ne00, const int ne01, const int ne02, const int ne03,
                             const int ne10,  const int ne11, const int ne12, const int ne13, const float ne0_scale_factor, 
                             const float ne1_scale_factor, const float ne2_scale_factor, const float ne3_scale_factor) {


    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index>= ne10 * ne11 * ne12 * ne13)
    {
        return;
    }

    int i10 = index % ne10;
    int i11 = (index / ne10)  % ne11;
    int i12 = (index / (ne10* ne11))  % ne12;
    int i13 = (index / (ne10* ne11 * ne12))  % ne13;

    int i00 = i10 / ne0_scale_factor;
    int i01 = i11 / ne1_scale_factor;
    int i02 = i12 / ne2_scale_factor;
    int i03 = i13 / ne3_scale_factor;

    int src_index = i00 + (i01 * ne00) + (i02 * ne00 * ne01) + (i02 * ne00 * ne01 * ne02);


    dst[index] = x[src_index];
}


static void upscale_f32_cuda(const float * x, float * dst, const int ne00, const int ne01, const int ne02, const int ne03,
                             const int ne10,  const int ne11, const int ne12, const int ne13, float ne0_scale_factor, float ne1_scale_factor,
                             float ne2_scale_factor, float ne3_scale_factor, cudaStream_t stream) {
    int dst_size = ne10 * ne11 * ne12* ne13;
    int num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    upscale_f32<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, ne00,ne01,ne02,ne03, ne10,ne11, ne12,ne13, 
    ne0_scale_factor, ne1_scale_factor, ne2_scale_factor, ne3_scale_factor);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    //GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    const float ne0_scale_factor = (float)dst->ne[0]/src0->ne[0];
    const float ne1_scale_factor = (float)dst->ne[1]/src0->ne[1];
    const float ne2_scale_factor = (float)dst->ne[2]/src0->ne[2];
    const float ne3_scale_factor = (float)dst->ne[3]/src0->ne[3];

    upscale_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0],dst->ne[1],dst->ne[2],dst->ne[3],  ne0_scale_factor,ne1_scale_factor,ne2_scale_factor,ne3_scale_factor, stream);
}
