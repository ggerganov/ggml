// quantized matrix multiplication

#include "ggml.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif

#ifndef MIN
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

const int M = 1280;
const int N = 1536;
const int K = 1280;

const int QK = 64;
#define QB 4

//#define GGML_GQ_USE_FP16_SCALE

#if defined(GGML_GQ_USE_FP16_SCALE)
#define gq_scale_t ggml_fp16_t
#define GGML_FP32_TO_GQ(x) ggml_fp32_to_fp16(x)
#define GGML_GQ_TO_FP32(x) ggml_fp16_to_fp32(x)
#else
#define gq_scale_t float
#define GGML_FP32_TO_GQ(x) (x)
#define GGML_GQ_TO_FP32(x) (x)
#endif

#define gq_quant_t uint64_t
#define gq_t_bits 64

float frand() {
    return (float) rand() / (float) RAND_MAX;
}

uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

//
// naive implementation
//

void mul_mat_f32_naive(
    const float * restrict src0, // M x K
    const float * restrict src1, // N x K (transposed)
    float * dst,
    int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += src0[i*k + l] * src1[j*k + l];
            }
            dst[i*n + j] = sum;
        }
    }
}

//
// method 1
//

static inline int quantize_1_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_1_quants_per_block() {
    return QK/gq_t_bits;
}

static inline int quantize_1_row_size(int k) {
    const int nb = quantize_1_blocks_per_row(k);
    const int nq = quantize_1_quants_per_block();

    return nb*(2*sizeof(gq_scale_t) + nq*QB*sizeof(gq_quant_t));
}

void quantize_1(const float * src, void * dst, int n, int k) {
    char * p0 = dst;

    gq_quant_t pp[QB];

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k/QK; i++) {
            float min = FLT_MAX;
            float max = -FLT_MAX;

            // find min/max
#ifdef __ARM_NEON
            {
                float32x4_t minv = vdupq_n_f32(FLT_MAX);
                float32x4_t maxv = vdupq_n_f32(-FLT_MAX);

                for (int l = 0; l < QK; l += 4) {
                    float32x4_t v = vld1q_f32(src + j*k + i*QK + l);
                    minv = vminq_f32(minv, v);
                    maxv = vmaxq_f32(maxv, v);
                }

                float32x2_t minv32 = vpmin_f32(vget_low_f32(minv), vget_high_f32(minv));
                float32x2_t maxv32 = vpmax_f32(vget_low_f32(maxv), vget_high_f32(maxv));

                min = MIN(vget_lane_f32(minv32, 0), vget_lane_f32(minv32, 1));
                max = MAX(vget_lane_f32(maxv32, 0), vget_lane_f32(maxv32, 1));

                //printf("SIMD min/max: %f %f\n", min, max);
            }
#else
            {
                for (int l = 0; l < QK; l++) {
                    const float v = src[j*k + i*QK + l];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                //printf("NORM min/max: %f %f\n", min, max);
            }
#endif

            const float d = (max - min) / ((1 << QB) - 1);
            const float id = d ? 1.0/d : 0.0;

            memcpy(p0, &min, sizeof(float)); p0 += sizeof(float);
            memcpy(p0, &d,   sizeof(float)); p0 += sizeof(float);

            //printf("min/max/d/id: %f %f %f %f\n", min, max, d, id);

            for (int s = 0; s < QK/gq_t_bits; ++s) {
                memset(pp, 0, sizeof(pp));

                for (int l = 0; l < gq_t_bits; l++) {
                    const   float v = src[j*k + i*QK + s*gq_t_bits + l];
                    const uint8_t q = (v - min)*id;

                    for (int b = 0; b < QB; b++) {
                        pp[b] |= q & (1 << b) ? (1ULL << l) : 0;
                    }
                }

                for (int b = 0; b < QB; b++) {
                    memcpy(p0, &pp[b], sizeof(gq_quant_t)); p0 += sizeof(gq_quant_t);
                }
            }
        }
    }
}

void mul_mat_gq_1(
    const void * src0,
    const void * src1,
         float * dst,
    int m, int n, int k) {
    const int kp = k & ~(gq_t_bits - 1);

    const char * restrict p0 = src0;
    const char * restrict p1 = src1;

    float s0[QB + 1];
    float s1[QB + 1];

    gq_quant_t m0[QB + 1];
    gq_quant_t m1[QB + 1];

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            float sumf = 0.0;

            const char * restrict pp0 = p0 + ir0*((2*sizeof(float) + (QK/gq_t_bits)*QB*sizeof(gq_quant_t))*(k/QK));
            const char * restrict pp1 = p1 + ir1*((2*sizeof(float) + (QK/gq_t_bits)*QB*sizeof(gq_quant_t))*(k/QK));

            for (int i = 0; i < kp/QK; i++) {
                float min0, d0;
                memcpy(&min0, pp0, sizeof(float)); pp0 += sizeof(float);
                memcpy(&d0,   pp0, sizeof(float)); pp0 += sizeof(float);

                float min1, d1;
                memcpy(&min1, pp1, sizeof(float)); pp1 += sizeof(float);
                memcpy(&d1,   pp1, sizeof(float)); pp1 += sizeof(float);

                //printf("min0/d0 = %f %f | min1/d1 = %f %f\n", min0, d0, min1, d1);

#if 1
                // >>> General case for any QB

                s0[0] = min0;
                s1[0] = min1;

                for (int b = 0; b < QB; b++) {
                    s0[b + 1] = d0*(1 << b);
                    s1[b + 1] = d1*(1 << b);
                }

                m0[0] = -1ULL;
                m1[0] = -1ULL;

                for (int s = 0; s < QK/gq_t_bits; ++s) {
                    for (int b = 0; b < QB; b++) {
                        memcpy(&m0[b + 1], pp0, sizeof(gq_quant_t)); pp0 += sizeof(gq_quant_t);
                        memcpy(&m1[b + 1], pp1, sizeof(gq_quant_t)); pp1 += sizeof(gq_quant_t);
                    }

                    for (int q0 = 0; q0 < QB + 1; q0++) {
                        for (int q1 = 0; q1 < QB + 1; q1++) {
                            sumf += s0[q0]*s1[q1]*__builtin_popcountll(m0[q0] & m1[q1]);
                        }
                    }
                }
#else
#endif
            }

            dst[ir0*n + ir1] = sumf;
        }
    }
}

//
// method 2
//

static inline int quantize_2_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_2_quants_per_block() {
    return QK/gq_t_bits;
}

static inline int quantize_2_row_size(int k) {
    const int nb = quantize_2_blocks_per_row(k);
    const int nq = quantize_2_quants_per_block();

    return nb*(2*sizeof(gq_scale_t) + nq*QB*sizeof(gq_quant_t));
}

void quantize_2_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % QK == 0);

    const int nb = quantize_2_blocks_per_row(k);
    const int nq = quantize_2_quants_per_block();

    gq_scale_t * restrict pm = (gq_scale_t *) (dst);
    gq_scale_t * restrict pd = (gq_scale_t *) (pm + nb);
    gq_quant_t * restrict pb = (gq_quant_t *) (pd + nb);

    gq_quant_t pp[QB];

    static const int32_t sh[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

#ifdef __ARM_NEON
        {
            float32x4_t minv = vdupq_n_f32(FLT_MAX);
            float32x4_t maxv = vdupq_n_f32(-FLT_MAX);

            for (int l = 0; l < QK; l += 4) {
                float32x4_t v = vld1q_f32(src + i*QK + l);
                minv = vminq_f32(minv, v);
                maxv = vmaxq_f32(maxv, v);
            }

            float32x2_t minv32 = vpmin_f32(vget_low_f32(minv), vget_high_f32(minv));
            float32x2_t maxv32 = vpmax_f32(vget_low_f32(maxv), vget_high_f32(maxv));

            min = MIN(vget_lane_f32(minv32, 0), vget_lane_f32(minv32, 1));
            max = MAX(vget_lane_f32(maxv32, 0), vget_lane_f32(maxv32, 1));
        }
#else
        {
            for (int l = 0; l < QK; l++) {
                const float v = src[i*QK + l];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
#endif

        const float d = (max - min) / ((1 << QB) - 1);
        const float id = d ? 1.0/d : 0.0;

        pm[i] = GGML_FP32_TO_GQ(min);
        pd[i] = GGML_FP32_TO_GQ(d);

        for (int s = 0; s < nq; ++s) {
            memset(pp, 0, sizeof(pp));

#if 0
            for (int l = 0; l < gq_t_bits; l++) {
                const   float v = src[i*QK + s*gq_t_bits + l];
                const uint8_t q = (v - min)*id + frand();

                for (int b = 0; b < QB; b++) {
                    pp[b] |= q & (1 << b) ? (1ULL << l) : 0;
                }
            }
#elif defined(__ARM_NEON)
#if 1
            {
                uint32_t ppt[2*4*QB];

                float32x4_t minv = vdupq_n_f32(min);
                float32x4_t idv  = vdupq_n_f32(id);

                assert(gq_t_bits == 64);

                uint32x4_t p0[QB] = { vdupq_n_u32(0) };
                uint32x4_t p1[QB] = { vdupq_n_u32(0) };

                for (int l = 0; l < gq_t_bits; l += 16) {
                    float32x4_t v0 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 0);
                    float32x4_t v1 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 4);
                    float32x4_t v2 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 8);
                    float32x4_t v3 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 12);

                    v0 = vsubq_f32(v0, minv);
                    v1 = vsubq_f32(v1, minv);
                    v2 = vsubq_f32(v2, minv);
                    v3 = vsubq_f32(v3, minv);

                    v0 = vmulq_f32(v0, idv);
                    v1 = vmulq_f32(v1, idv);
                    v2 = vmulq_f32(v2, idv);
                    v3 = vmulq_f32(v3, idv);

#if 1
                    v0[0] += frand(); v0[1] += frand(); v0[2] += frand(); v0[3] += frand();
                    v1[0] += frand(); v1[1] += frand(); v1[2] += frand(); v1[3] += frand();
                    v2[0] += frand(); v2[1] += frand(); v2[2] += frand(); v2[3] += frand();
                    v3[0] += frand(); v3[1] += frand(); v3[2] += frand(); v3[3] += frand();
#endif

                    uint32x4_t q0 = vcvtq_u32_f32(v0);
                    uint32x4_t q1 = vcvtq_u32_f32(v1);
                    uint32x4_t q2 = vcvtq_u32_f32(v2);
                    uint32x4_t q3 = vcvtq_u32_f32(v3);

                    for (int b = 0; b < QB; ++b) {
                        uint32x4_t m = vdupq_n_u32(1 << b);
                        uint32x4_t r = vdupq_n_u32(-b);

                        if (l < 32) {
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q0, m), r), vld1q_s32(sh + l + 0)));
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q1, m), r), vld1q_s32(sh + l + 4)));
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q2, m), r), vld1q_s32(sh + l + 8)));
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q3, m), r), vld1q_s32(sh + l + 12)));
                        } else {
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q0, m), r), vld1q_s32(sh + l - 32)));
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q1, m), r), vld1q_s32(sh + l - 28)));
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q2, m), r), vld1q_s32(sh + l - 24)));
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q3, m), r), vld1q_s32(sh + l - 20)));
                        }
                    }
                }

                vst1q_u32((uint32_t *) ppt + 0,  p0[0]);
                vst1q_u32((uint32_t *) ppt + 4,  p1[0]);
                vst1q_u32((uint32_t *) ppt + 8,  p0[1]);
                vst1q_u32((uint32_t *) ppt + 12, p1[1]);
                vst1q_u32((uint32_t *) ppt + 16, p0[2]);
                vst1q_u32((uint32_t *) ppt + 20, p1[2]);
                vst1q_u32((uint32_t *) ppt + 24, p0[3]);
                vst1q_u32((uint32_t *) ppt + 28, p1[3]);

                pp[0] = (ppt[0]  | ppt[1]  | ppt[2]  | ppt[3] ) | ((uint64_t) (ppt[4]  | ppt[5]  | ppt[6]  | ppt[7]) ) << 32;
                pp[1] = (ppt[8]  | ppt[9]  | ppt[10] | ppt[11]) | ((uint64_t) (ppt[12] | ppt[13] | ppt[14] | ppt[15])) << 32;
                pp[2] = (ppt[16] | ppt[17] | ppt[18] | ppt[19]) | ((uint64_t) (ppt[20] | ppt[21] | ppt[22] | ppt[23])) << 32;
                pp[3] = (ppt[24] | ppt[25] | ppt[26] | ppt[27]) | ((uint64_t) (ppt[28] | ppt[29] | ppt[30] | ppt[31])) << 32;
            }
#else
            // less optimal SIMD
            {
                float32x4_t minv = vdupq_n_f32(min);
                float32x4_t idv  = vdupq_n_f32(id);

                assert(gq_t_bits == 64);
                uint8_t qq[gq_t_bits];

                for (int l = 0; l < gq_t_bits; l += 16) {
                    float32x4_t v0 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 0);
                    float32x4_t v1 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 4);
                    float32x4_t v2 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 8);
                    float32x4_t v3 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 12);

                    v0 = vsubq_f32(v0, minv);
                    v1 = vsubq_f32(v1, minv);
                    v2 = vsubq_f32(v2, minv);
                    v3 = vsubq_f32(v3, minv);

                    v0 = vmulq_f32(v0, idv);
                    v1 = vmulq_f32(v1, idv);
                    v2 = vmulq_f32(v2, idv);
                    v3 = vmulq_f32(v3, idv);

#if 0
                    v0[0] += frand(); v0[1] += frand(); v0[2] += frand(); v0[3] += frand();
                    v1[0] += frand(); v1[1] += frand(); v1[2] += frand(); v1[3] += frand();
                    v2[0] += frand(); v2[1] += frand(); v2[2] += frand(); v2[3] += frand();
                    v3[0] += frand(); v3[1] += frand(); v3[2] += frand(); v3[3] += frand();
#endif

                    uint32x4_t q0 = vcvtq_u32_f32(v0);
                    uint32x4_t q1 = vcvtq_u32_f32(v1);
                    uint32x4_t q2 = vcvtq_u32_f32(v2);
                    uint32x4_t q3 = vcvtq_u32_f32(v3);

                    // store in qq as uint8_t
                    vst1_u8(qq + l + 0, vmovn_u16(vcombine_u16(vmovn_u32(q0), vmovn_u32(q1))));
                    vst1_u8(qq + l + 8, vmovn_u16(vcombine_u16(vmovn_u32(q2), vmovn_u32(q3))));
                }

                for (int l = 0; l < gq_t_bits; l++) {
                    for (int b = 0; b < QB; b++) {
                        const uint64_t ql = qq[l];
                        /*pp[b] |= qq[l] & (1 << b) ? (1ULL << l) : 0;*/
                        pp[b] |= ((ql & (1 << b)) >> b) << l;
                    }
                }
            }
#endif
#endif
            memcpy(pb + i*nq*QB + s*QB, pp, sizeof(pp));
        }
    }
}

// reimplementation of quantize_2 using quantize_2_row
void quantize_2(const float * restrict src, char * restrict dst, int n, int k) {
    assert(k % QK == 0);

    for (int j = 0; j < n; j++) {
        quantize_2_row(src + j*k, dst, k);
        dst = (char *) dst + quantize_2_row_size(k);
    }
}

void vec_dot_gq_2(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    float sumf[(QB + 1)*(QB + 1)];
    memset(sumf, 0, sizeof(sumf));

    const int nb = quantize_2_blocks_per_row(n);
    const int nq = quantize_2_quants_per_block();

    const gq_scale_t * restrict pm0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pm1 = (const gq_scale_t *) y;

    const gq_scale_t * restrict pd0 = pm0 + nb;
    const gq_scale_t * restrict pd1 = pm1 + nb;

    const gq_quant_t * restrict pb0 = (const gq_quant_t *) (pd0 + nb);
    const gq_quant_t * restrict pb1 = (const gq_quant_t *) (pd1 + nb);

#if 1
    float s0[QB + 1];
    float s1[QB + 1];

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        s0[0] = m0;
        s1[0] = m1;

        for (int b = 0; b < QB; b++) {
            s0[b + 1] = d0*(1 << b);
            s1[b + 1] = d1*(1 << b);
        }

        for (int s = 0; s < nq; ++s) {
            for (int q0 = 0; q0 < QB + 1; q0++) {
                const gq_quant_t mm0 = q0 ? pb0[i*nq*QB + s*QB + q0 - 1] : -1ULL;
                for (int q1 = 0; q1 < QB + 1; q1++) {
                    const gq_quant_t mm1 = q1 ? pb1[i*nq*QB + s*QB + q1 - 1] : -1ULL;
                    sumf[q0*(QB + 1) + q1] += s0[q0]*s1[q1]*__builtin_popcountll(mm0 & mm1);
                }
            }
        }
    }
#else
    // SIMD-ify with the assumptions:
    // - nb is a multiple of 4
    // - gq_scale_t is float
    // - gq_quant_t is uint64_t
    // - QB == 7
    assert(nb % 4 == 0);

#ifdef __ARM_NEON
#else
    // TODO
#endif

#endif

    for (int q0 = 0; q0 < QB + 1; q0++) {
        for (int q1 = 1; q1 < QB + 1; q1++) {
            sumf[q0*(QB + 1)] += sumf[q0*(QB + 1) + q1];
        }
    }

    *s = sumf[0];
    for (int q0 = 1; q0 < QB + 1; q0++) {
        *s += sumf[q0*(QB + 1)];
    }
}

// use vec_dot_gq_2 to compute the dot product of two rows
void mul_mat_gq_2(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

    const int nb = quantize_2_blocks_per_row(k);
    const int nq = quantize_2_quants_per_block();

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_2(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_2_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_2_row_size(k);
        src1 = (const char *) src1 - n*quantize_2_row_size(k);

        dst = (float *) dst + n;
    }
}

//
// method 3
//

static inline int quantize_3_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_3_quants_per_block() {
    return QK/gq_t_bits;
}

static inline int quantize_3_row_size(int k) {
    const int nb = quantize_3_blocks_per_row(k);
    const int nq = quantize_3_quants_per_block();

    return nb*(sizeof(gq_scale_t) + nq*QB*sizeof(gq_quant_t));
}

void quantize_3_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % QK == 0);

    const int nb = quantize_3_blocks_per_row(k);
    const int nq = quantize_3_quants_per_block();

    gq_scale_t * restrict pd = (gq_scale_t *) (dst);
    gq_quant_t * restrict pb = (gq_quant_t *) (pd + nb);

    gq_quant_t pp[QB];

    static const int32_t sh[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // abs max

#ifdef __ARM_NEON
        {
            // min / max
            //float32x4_t minv = vdupq_n_f32(FLT_MAX);
            //float32x4_t maxv = vdupq_n_f32(-FLT_MAX);

            //for (int l = 0; l < QK; l += 4) {
            //    float32x4_t v = vld1q_f32(src + i*QK + l);
            //    minv = vminq_f32(minv, v);
            //    maxv = vmaxq_f32(maxv, v);
            //}

            //float32x2_t minv32 = vpmin_f32(vget_low_f32(minv), vget_high_f32(minv));
            //float32x2_t maxv32 = vpmax_f32(vget_low_f32(maxv), vget_high_f32(maxv));

            //min = MIN(vget_lane_f32(minv32, 0), vget_lane_f32(minv32, 1));
            //max = MAX(vget_lane_f32(maxv32, 0), vget_lane_f32(maxv32, 1));

            // abs max
            float32x4_t amaxv = vdupq_n_f32(0.0f);

            for (int l = 0; l < QK; l += 4) {
                float32x4_t v = vld1q_f32(src + i*QK + l);
                amaxv = vmaxq_f32(amaxv, vabsq_f32(v));
            }

            float32x2_t amaxv32 = vpmax_f32(vget_low_f32(amaxv), vget_high_f32(amaxv));

            amax = MAX(vget_lane_f32(amaxv32, 0), vget_lane_f32(amaxv32, 1));
        }
#else
        {
            for (int l = 0; l < QK; l++) {
                const float v = src[i*QK + l];
                amax = MAX(amax, fabsf(v));
            }
        }
#endif

        const float d = amax / ((1 << (QB - 1)) - 1);
        const float id = d ? 1.0/d : 0.0;

        pd[i] = GGML_FP32_TO_GQ(d);

        for (int s = 0; s < nq; ++s) {
            memset(pp, 0, sizeof(pp));

#if 0
            for (int l = 0; l < gq_t_bits; l++) {
                const   float v = src[i*QK + s*gq_t_bits + l];
                const uint8_t q = v*id + frand();

                for (int b = 0; b < QB; b++) {
                    pp[b] |= q & (1 << b) ? (1ULL << l) : 0;
                }
            }
#elif defined(__ARM_NEON)
            {
                uint32_t ppt[2*4*QB];

                float32x4_t idv  = vdupq_n_f32(id);

                assert(gq_t_bits == 64);

                uint32x4_t p0[QB] = { vdupq_n_u32(0) };
                uint32x4_t p1[QB] = { vdupq_n_u32(0) };

                for (int l = 0; l < gq_t_bits; l += 16) {
                    float32x4_t v0 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 0);
                    float32x4_t v1 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 4);
                    float32x4_t v2 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 8);
                    float32x4_t v3 = vld1q_f32(src + i*QK + s*gq_t_bits + l + 12);

                    v0 = vmulq_f32(v0, idv);
                    v1 = vmulq_f32(v1, idv);
                    v2 = vmulq_f32(v2, idv);
                    v3 = vmulq_f32(v3, idv);

#if 1
                    v0[0] += frand(); v0[1] += frand(); v0[2] += frand(); v0[3] += frand();
                    v1[0] += frand(); v1[1] += frand(); v1[2] += frand(); v1[3] += frand();
                    v2[0] += frand(); v2[1] += frand(); v2[2] += frand(); v2[3] += frand();
                    v3[0] += frand(); v3[1] += frand(); v3[2] += frand(); v3[3] += frand();
#endif

                    uint32x4_t q0 = vcvtq_u32_f32(v0);
                    uint32x4_t q1 = vcvtq_u32_f32(v1);
                    uint32x4_t q2 = vcvtq_u32_f32(v2);
                    uint32x4_t q3 = vcvtq_u32_f32(v3);

                    for (int b = 0; b < QB; ++b) {
                        uint32x4_t m = vdupq_n_u32(1 << b);
                        uint32x4_t r = vdupq_n_u32(-b);

                        if (l < 32) {
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q0, m), r), vld1q_s32(sh + l + 0)));
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q1, m), r), vld1q_s32(sh + l + 4)));
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q2, m), r), vld1q_s32(sh + l + 8)));
                            p0[b] = vorrq_u32(p0[b], vshlq_u32(vshlq_u32(vandq_u32(q3, m), r), vld1q_s32(sh + l + 12)));
                        } else {
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q0, m), r), vld1q_s32(sh + l - 32)));
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q1, m), r), vld1q_s32(sh + l - 28)));
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q2, m), r), vld1q_s32(sh + l - 24)));
                            p1[b] = vorrq_u32(p1[b], vshlq_u32(vshlq_u32(vandq_u32(q3, m), r), vld1q_s32(sh + l - 20)));
                        }
                    }
                }

                vst1q_u32((uint32_t *) ppt + 0,  p0[0]);
                vst1q_u32((uint32_t *) ppt + 4,  p1[0]);
                vst1q_u32((uint32_t *) ppt + 8,  p0[1]);
                vst1q_u32((uint32_t *) ppt + 12, p1[1]);
                vst1q_u32((uint32_t *) ppt + 16, p0[2]);
                vst1q_u32((uint32_t *) ppt + 20, p1[2]);
                vst1q_u32((uint32_t *) ppt + 24, p0[3]);
                vst1q_u32((uint32_t *) ppt + 28, p1[3]);

                pp[0] = (ppt[0]  | ppt[1]  | ppt[2]  | ppt[3] ) | ((uint64_t) (ppt[4]  | ppt[5]  | ppt[6]  | ppt[7]) ) << 32;
                pp[1] = (ppt[8]  | ppt[9]  | ppt[10] | ppt[11]) | ((uint64_t) (ppt[12] | ppt[13] | ppt[14] | ppt[15])) << 32;
                pp[2] = (ppt[16] | ppt[17] | ppt[18] | ppt[19]) | ((uint64_t) (ppt[20] | ppt[21] | ppt[22] | ppt[23])) << 32;
                pp[3] = (ppt[24] | ppt[25] | ppt[26] | ppt[27]) | ((uint64_t) (ppt[28] | ppt[29] | ppt[30] | ppt[31])) << 32;
            }
#endif
            memcpy(pb + i*nq*QB + s*QB, pp, sizeof(pp));
        }
    }
}

// reimplementation of quantize_3 using quantize_3_row
void quantize_3(const float * restrict src, char * restrict dst, int n, int k) {
    assert(k % QK == 0);

    for (int j = 0; j < n; j++) {
        quantize_3_row(src + j*k, dst, k);
        dst = (char *) dst + quantize_3_row_size(k);
    }
}

void vec_dot_gq_3(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    float sumf[QB*QB];
    memset(sumf, 0, sizeof(sumf));

    const int nb = quantize_3_blocks_per_row(n);
    const int nq = quantize_3_quants_per_block();

    const gq_scale_t * restrict pd0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pd1 = (const gq_scale_t *) y;

    const gq_quant_t * restrict pb0 = (const gq_quant_t *) (pd0 + nb);
    const gq_quant_t * restrict pb1 = (const gq_quant_t *) (pd1 + nb);

#if 1
    float s0[QB];
    float s1[QB];

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        for (int b = 0; b < QB; b++) {
            s0[b] = d0*(1 << b);
            s1[b] = d1*(1 << b);
        }

        for (int s = 0; s < nq; ++s) {
            for (int q0 = 0; q0 < QB; q0++) {
                const gq_quant_t mm0 = pb0[i*nq*QB + s*QB + q0];
                for (int q1 = 0; q1 < QB; q1++) {
                    const gq_quant_t mm1 = pb1[i*nq*QB + s*QB + q1];
                    sumf[q0*QB + q1] += s0[q0]*s1[q1]*__builtin_popcountll(mm0 & mm1);
                }
            }
        }
    }
#else
    // SIMD-ify with the assumptions:
    // - nb is a multiple of 4
    // - gq_scale_t is float
    // - gq_quant_t is uint64_t
    // - QB == 7
    assert(nb % 4 == 0);

#ifdef __ARM_NEON
#else
    // TODO
#endif

#endif

    for (int q0 = 0; q0 < QB; q0++) {
        for (int q1 = 1; q1 < QB; q1++) {
            sumf[q0*QB] += sumf[q0*QB + q1];
        }
    }

    *s = sumf[0];
    for (int q0 = 1; q0 < QB; q0++) {
        *s += sumf[q0*QB];
    }
}

// use vec_dot_gq_3 to compute the dot product of two rows
void mul_mat_gq_3(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

    const int nb = quantize_3_blocks_per_row(k);
    const int nq = quantize_3_quants_per_block();

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_3(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_3_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_3_row_size(k);
        src1 = (const char *) src1 - n*quantize_3_row_size(k);

        dst = (float *) dst + n;
    }
}

int main(int argc, const char ** argv) {
    assert(sizeof(gq_quant_t)*8 == gq_t_bits);

    int method = 0;
    if (argc > 1) {
        method = atoi(argv[1]);
    }

    float * src0 = (float *)malloc(sizeof(float)*M*K);
    float * src1 = (float *)malloc(sizeof(float)*N*K);
    float * dst  = (float *)malloc(sizeof(float)*M*N);

    for (int i = 0; i < M*K; i++) {
        /*src0[i] = rand() / (float)RAND_MAX;*/
        src0[i] = i % 3;
    }

    for (int i = 0; i < N*K; i++) {
        /*src1[i] = rand() / (float)RAND_MAX;*/
        src1[i] = i % 4;
    }

    void * src0_gq = NULL;
    void * src1_gq = NULL;

    size_t sizegq = 0;

    {
        if (method == 1) {
            src0_gq = calloc(1, quantize_1_row_size(K)*M);
            src1_gq = calloc(1, quantize_1_row_size(K)*N);

            sizegq  = quantize_1_row_size(K)*M + quantize_1_row_size(K)*N;
        }

        if (method == 2) {
            src0_gq = calloc(1, quantize_2_row_size(K)*M);
            src1_gq = calloc(1, quantize_2_row_size(K)*N);

            sizegq  = quantize_2_row_size(K)*M + quantize_2_row_size(K)*N;
        }

        if (method == 3) {
            src0_gq = calloc(1, quantize_3_row_size(K)*M);
            src1_gq = calloc(1, quantize_3_row_size(K)*N);

            sizegq  = quantize_3_row_size(K)*M + quantize_3_row_size(K)*N;
        }
    }

    const size_t sizef16 = sizeof(ggml_fp16_t)*M*K + sizeof(ggml_fp16_t)*N*K;

    printf("compression: %f\n", (float)sizegq/sizef16);

    // convert fp32 -> gq
    {
        const uint64_t t_start = get_time_us();

        if (method == 1) {
            quantize_1(src0, src0_gq, M, K);
            quantize_1(src1, src1_gq, N, K);
        }

        if (method == 2) {
            quantize_2(src0, src0_gq, M, K);
            quantize_2(src1, src1_gq, N, K);
        }

        if (method == 3) {
            quantize_3(src0, src0_gq, M, K);
            quantize_3(src1, src1_gq, N, K);
        }

        const uint64_t t_end = get_time_us();
        printf("convert time: %f ms / method = %d\n", (t_end - t_start) / 1000.0, method);
    }

    for (int i = 0; i < 16; ++i) {
        printf("%f %f\n", src0[i], src1[i]);
    }

    const int nIter = 1;

    const clock_t start = clock();
    const uint64_t start_us = get_time_us();

    double iM = 1.0/M;
    double sum = 0.0f;
    for (int i = 0; i < nIter; i++) {
        if (method == 0) {
            mul_mat_f32_naive(src0, src1, dst, M, N, K);
        }

        if (method == 1) {
            mul_mat_gq_1(src0_gq, src1_gq, dst, M, N, K);
        }

        if (method == 2) {
            mul_mat_gq_2(src0_gq, src1_gq, dst, M, N, K);
        }

        if (method == 3) {
            mul_mat_gq_3(src0_gq, src1_gq, dst, M, N, K);
        }
    }

    for (int i = 0; i < N; i++) {
        sum += dst[i]*iM;
    }

    {
        const clock_t end = clock();
        const uint64_t end_us = get_time_us();
        printf("%s: elapsed ticks: %ld\n",  __func__, end - start);
        printf("%s: elapsed us:    %d / %f ms\n",  __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0 / nIter);
    }

    printf("%f\n", sum);

    free(src0);
    free(src1);
    free(dst);

    if (src0_gq) free(src0_gq);
    if (src1_gq) free(src1_gq);

    return 0;
}
