#ifndef GGML_SYCL_COl2IM_HPP
#define GGML_SYCL_COl2IM_HPP

#include "common.hpp"

void ggml_sycl_op_col2im(ggml_backend_sycl_context &ctx,
                         const ggml_tensor *src0,
                         const ggml_tensor *src1,
                         ggml_tensor *dst,
                         const float *src0_dd,
                         const float *src1_dd,
                         float *dst_dd,
                         const queue_ptr &main_stream);

#endif  // GGML_SYCL_COl2IM_HPP
