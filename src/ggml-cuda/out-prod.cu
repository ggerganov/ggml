#include "out-prod.cuh"
#include "opt-step-adam.cuh"
#include "vendors/cuda.h"

#include <cstdint>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    GGML_ASSERT(ne01 == ne11);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 == src0->ne[2]);
    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src0->ne[3]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    GGML_ASSERT(ne2 == 1);
    GGML_ASSERT(ne3 == 1);
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const cublasOperation_t src1_cublas_op = ggml_is_transposed(src1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = ggml_is_transposed(src1) ? ne11        : ne10;
    CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                ne0, ne1, ne01,
                &alpha, src0_d, ne00,
                        src1_d, ldb,
                &beta,  dst_d,  ne0));
}
