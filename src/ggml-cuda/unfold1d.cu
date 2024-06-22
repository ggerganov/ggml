#include "unfold1d.cuh"

static __global__ void unfold_1d_f32(const float * x, float * dst, const int s, const int ne0, const int ne1, const int ne2,
    const int ne3, const int ne00, const int ne01, const int ne02, const int ne03) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0 * ne1 * ne2 * ne3) {
        return;
    }

    const int i3 = nidx/(ne0 * ne1 * ne2);
    const int i2 = (nidx - i3*ne0*ne1*ne2 )/ (ne0*ne1);
    const int i1 = (nidx - i3*ne0*ne1*ne2  -  i2*ne1*ne0) / ne0;
    const int i0 = nidx - i3*ne0*ne1*ne2 - i2*ne1*ne0 - i1*ne0;    
                    
    const int src_idx = i3 *(ne00*ne01) + i2 * (ne00) + i1*s + i0;

    dst[nidx] = x[src_idx];
}

static void unfold_1d_f32_cuda(const float * x, float * dst, const int s,
    const int ne0, const int ne1, const int ne2, const int ne3,
    const int ne00, const int ne01, const int ne02, const int ne03, cudaStream_t stream) {
    int num_blocks = ((ne0 * ne1 * ne2 * ne3) + CUDA_UNFOLD_1D_BLOCK_SIZE - 1) / CUDA_UNFOLD_1D_BLOCK_SIZE;

    unfold_1d_f32<<<num_blocks, CUDA_UNFOLD_1D_BLOCK_SIZE,0,stream>>>(x, dst, s, ne0, ne1, ne2, ne3, ne00, ne01, ne02, ne03);
}

void ggml_cuda_op_unfold_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1); // only up to 3 dimensions for input tensor

    const int32_t * opts = (const int32_t *)dst->op_params;
    const int s = opts[1];

    unfold_1d_f32_cuda(src0_d, dst_d, s,
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], stream);
}
