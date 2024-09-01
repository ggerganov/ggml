#include "col2im.hpp"

#include "common.hpp"

static void col2im_kernel(const float *src,
                          float *dst,
                          const int64_t IW,
                          const int64_t KW,
                          const int64_t OC,
                          const int64_t N,
                          int64_t OW,
                          const int64_t ioc_offs,
                          const int64_t ikw_offs,
                          const int64_t in_offs,
                          const int32_t s0,
                          const int32_t p0,
                          const int32_t d0,
                          const sycl::nd_item<1> &item) {
    const int64_t global_id = item.get_global_linear_id();
    const int64_t batch_size = OC * OW;
    const int64_t osize = N * batch_size;

    for (int64_t i = global_id; i < osize; i += batch_size) {
        const auto in = i / batch_size;
        const auto iow = i % batch_size;  // = ioc * OW + ikw*d0 - p0 + iiws*s0
        const auto kwiw = iow % OW + p0;  // = ikw*d0 + iiw*s0
        const auto ioc = iow / OW;
        const auto max_kernel = (KW - 1) * d0;
        // iter iiw only over values that have
        // a chance of being valid
        // i.e. values that will satisfy:
        // 0 < ikw*d0 - p0 + iiw*s0 < OW
        const auto iiws = std::max(0L, (kwiw - max_kernel + s0 - 1) / s0);
        const auto iiwe = std::min(IW, kwiw / s0 + 1);

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

static void col2im_sycl(const float *src,
                        float *dst,
                        const int64_t IW,
                        const int64_t KW,
                        const int64_t OC,
                        const int64_t N,
                        int64_t OW,
                        const int64_t ioc_offs,
                        const int64_t ikw_offs,
                        const int64_t in_offs,
                        const int32_t s0,
                        const int32_t p0,
                        const int32_t d0,
                        queue_ptr stream) {
    const int64_t batch_size = OC * OW;
    const size_t num_blocks =
        (batch_size + SYCL_COL2IM_BLOCK_SIZE - 1) / SYCL_COL2IM_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_COL2IM_BLOCK_SIZE, SYCL_COL2IM_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            col2im_kernel(
                src, dst, IW, KW, OC, N, OW, ioc_offs, ikw_offs, in_offs, s0, p0, d0, item);
        });
}

void ggml_sycl_op_col2im(ggml_backend_sycl_context &,
                         const ggml_tensor *src0,
                         const ggml_tensor *,
                         ggml_tensor *dst,
                         const float *src0_dd,
                         const float *,
                         float *dst_dd,
                         const queue_ptr &main_stream) {
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
    const auto N = src0->ne[3];
    const auto OW = dst->ne[0];

    const auto ioc_offs = src0->nb[2] / src0->nb[0];
    const auto ikw_offs = src0->nb[1] / src0->nb[0];
    const auto in_offs = src0->nb[3] / src0->nb[0];

    col2im_sycl(src0_dd, dst_dd,
                IW, KW, OC, N, OW,
                ioc_offs, ikw_offs, in_offs,
                s0, p0, d0,
                main_stream);
}
