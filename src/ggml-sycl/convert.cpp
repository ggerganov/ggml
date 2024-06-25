#include "convert.hpp"
#include "dequantize.hpp"
#include "presets.hpp"

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k,
                             const sycl::nd_item<3> &item_ct1) {
    const int i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void *__restrict__ vx,
                                  dst_t *__restrict__ y, const int k,
                                  dpct::queue_ptr stream) {
    const int num_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q4_1_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_iq1_s_sycl(const void *vx, dst_t *y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_sycl(const void *vx, dst_t *y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_sycl(const void *vx, dst_t *y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_sycl(const void *vx, dst_t *y, const int k,
                                       dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_sycl(const void *vx, dst_t *y, const int k,
                                      dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


template <typename dst_t>
static void dequantize_row_iq3_xxs_sycl(const void *vx, dst_t *y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_sycl(const void *vx, dst_t *y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_sycl(const void *vx, dst_t *y, const int k,
                                       dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#else
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#endif
}

template <typename dst_t>
static void dequantize_row_iq4_nl_sycl(const void *vx, dst_t *y, const int k,
                                       dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

template <typename src_t, typename dst_t>
static void convert_unary(const void * __restrict__ vx, dst_t * __restrict__ y, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    const src_t * x = (src_t *) vx;

    y[i] = x[i];
}

template <typename src_t, typename dst_t>
static void convert_unary_sycl(const void *__restrict__ vx,
                               dst_t *__restrict__ y, const int k,
                               dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / SYCL_DEQUANTIZE_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                convert_unary<src_t>(vx, y, k, item_ct1);
            });
    }
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_sycl;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_sycl;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
        default:
            return nullptr;
    }
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_sycl;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_sycl;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_sycl;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_sycl;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
        default:
            return nullptr;
    }
}
