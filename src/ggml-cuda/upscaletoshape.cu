#include "upscaletoshape.cuh"

static __global__ void upscale_to_shape_f32(const float * x, float * dst, const int ne00, const int ne01, const int ne02, const int ne03,
                             const int ne10,  const int ne11, const int ne12, const int ne13, const float ne0_scale_factor, 
                             const float ne1_scale_factor, const float ne2_scale_factor, const float ne3_scale_factor) {


    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index>= ne10 * ne11 * ne12 * ne13)
    {
        return;
    }
    dst[index] = 9;
    // blockIdx.z: idx of ne02*ne03
    // blockIdx.y: idx of ne01*scale_factor， aka ne1
    // blockIDx.x: idx of ne00*scale_factor / BLOCK_SIZE
    // ne00xne01: ne00 * ne01
    /*
    int ne0 = ne00 * scale_factor;
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    // operation
    int i00 = nidx / scale_factor;
    int i01 = blockIdx.y / scale_factor;
    int offset_src =
        i00 +
        i01 * ne00 +
        blockIdx.z * ne00xne01;
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    dst[offset_dst] = x[offset_src];
    */

}


static void upscale_to_shape_f32_cuda(const float * x, float * dst, const int ne00, const int ne01, const int ne02, const int ne03,
                             const int ne10,  const int ne11, const int ne12, const int ne13, float ne0_scale_factor, float ne1_scale_factor,
                             float ne2_scale_factor, float ne3_scale_factor, cudaStream_t stream) {
    int dst_size = ne10 * ne11 * ne12* ne13;
    int num_blocks = (dst_size + CUDA_UPSCALE_TO_SHAPE_BLOCK_SIZE - 1) / CUDA_UPSCALE_TO_SHAPE_BLOCK_SIZE;

    upscale_to_shape_f32<<<num_blocks, CUDA_UPSCALE_TO_SHAPE_BLOCK_SIZE,0,stream>>>(x, dst, ne00,ne01,ne02,ne03, ne10,ne11, ne12,ne13, 
    ne0_scale_factor, ne1_scale_factor, ne2_scale_factor, ne3_scale_factor);
}

void ggml_cuda_op_upscale_to_shape(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    //GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    const float ne0_scale_factor = (float)dst->ne[0]/dst->op_params[0];
    const float ne1_scale_factor = (float)dst->ne[1]/dst->op_params[1];
    const float ne2_scale_factor = (float)dst->ne[2]/dst->op_params[2];
    const float ne3_scale_factor = (float)dst->ne[3]/dst->op_params[3];

    upscale_to_shape_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0],dst->ne[1],dst->ne[2],dst->ne[3],  ne0_scale_factor,ne1_scale_factor,ne2_scale_factor,ne3_scale_factor, stream);
}
