#include "col2im.cuh"

static  __global__ void col2im_kernel(
                    const float * src, float* dst,
                    const int64_t IW, const int64_t KW, const int64_t OC, const int64_t N, int64_t OW,
                    const int64_t ioc_offs, const int64_t ikw_offs, const int64_t in_offs,
                    const int32_t s0, const int32_t p0, const int32_t d0) {
        const auto batch_size = OC * OW;
        for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
             i < batch_size * N;
             i += blockDim.x * gridDim.x) {
            const auto in = i / batch_size;
            const auto iow = i % batch_size;  // = ioc * OW + ikw*d0 - p0 + iiws*s0
            const auto kwiw = iow % OW + p0;  // = ikw*d0 + iiw*s0
            const auto ioc = iow / OW;
            const auto max_kernel = (KW - 1) * d0;
            // iter iiw only over values that have
            // a chance of being valid
            // i.e. values that will satisfy:
            // 0 < ikw*d0 - p0 + iiw*s0 < OW
            const auto iiws = ::max(0L, (kwiw - max_kernel + s0 - 1) / s0);
            const auto iiwe = ::min(IW, kwiw / s0 + 1);

            const float *const input = src + in * in_offs;
            float val = 0;
            for (auto iiw = iiws; iiw < iiwe; ++iiw) {
                const auto ikw_d = (kwiw - iiw * s0);
                if (ikw_d % d0 == 0) {
                    const auto ikw = ikw_d / d0;
                    const auto input_index = ioc * ioc_offs + ikw * ikw_offs + iiw;
                    val += input[input_index];
                }
            }
            dst[in * OC * OW + iow] = val;
        }
}

void ggml_cuda_op_col2im(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const auto * src0_d = (const float *)src0->data;
    auto * dst_d = (float *)dst->data;
    auto stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    assert(ggml_is_contiguous(dst));
    assert(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));

    const auto s0 = dst->op_params[0];
    const auto p0 = dst->op_params[2];
    const auto d0 = dst->op_params[4];

    assert(s0 >= 1);
    assert(p0 >= 0);
    assert(d0 >= 1);

    const auto IW = src0->ne[0];
    const auto KW = src0->ne[1];
    const auto OC = src0->ne[2];
    const auto N  = src0->ne[3];
    const auto OW = dst->ne[0];

    const auto ioc_offs = src0->nb[2] / src0->nb[0];
    const auto ikw_offs = src0->nb[1] / src0->nb[0];
    const auto in_offs  = src0->nb[3] / src0->nb[0];

    const int parallel_elements = N * OC * OW;
    const int num_blocks = (parallel_elements + CUDA_COL2IM_BLOCK_SIZE - 1) / CUDA_COL2IM_BLOCK_SIZE;

    col2im_kernel<<<num_blocks, CUDA_COL2IM_BLOCK_SIZE, 0, stream>>>(src0_d, dst_d,
                                                                     IW, KW, OC, N, OW,
                                                                     ioc_offs, ikw_offs, in_offs,
                                                                     s0, p0, d0);
}
