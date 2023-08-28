// quantized matrix multiplication

#include "ggml.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON)
#include "arm_neon.h"
#elif defined(__AVX__) || defined(__AVX2__)
#include "immintrin.h"
#endif

#ifndef MIN
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#include <intrin.h>
#define __builtin_popcountll __popcnt64
#endif

const int M = 1280;
const int N = 1536;
const int K = 1280;

//const int M = 64;
//const int N = 64;
//const int K = 64;

#define QK 64
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

#define gq_t_bits 64
#define gq_quant_t uint64_t

float frand(void) {
    return (float) rand() / (float) RAND_MAX;
}

#if defined(__AVX2__)
// horizontally reduce 8 32-bit integers
static inline uint32_t _mm256_hadd_epi32_gg(__m256i v) {
    __m128i v0 = _mm256_extractf128_si256(v, 0);
    __m128i v1 = _mm256_extractf128_si256(v, 1);

    v0 = _mm_add_epi32(v0, v1);

    v1 = _mm_shuffle_epi32(v0, 0x0e);
    v0 = _mm_add_epi32(v0, v1);

    v1 = _mm_shuffle_epi32(v0, 0x01);
    v0 = _mm_add_epi32(v0, v1);

    return _mm_cvtsi128_si32(v0);
}

//static inline float _mm256_hadd_epi32_gg(__m256i v) {
//    const __m256 v0 = _mm256_cvtepi32_ps(v);
//    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(v0), _mm256_extractf128_ps(v0, 1));
//    const __m128 t1 = _mm_hadd_ps(t0, t0);
//
//    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
//}

// horizontally reduce 32 8-bit integers
static inline int32_t _mm256_hadd_epi8_gg(__m256i v0) {
    __m256i v1 = _mm256_maddubs_epi16(v0, _mm256_set1_epi8(1));
    __m256i v2 = _mm256_madd_epi16   (v1, _mm256_set1_epi16(1));

    return _mm256_hadd_epi32_gg(v2);
}

static inline float _mm256_hadd_ps_gg(__m256 v) {
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);

    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}
#endif

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

static inline int quantize_1_quants_per_block(void) {
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

                m0[0] = 0-1ULL;
                m1[0] = 0-1ULL;

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
// n-bit quantization (2nd attempt)
//

static inline int quantize_2_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_2_quants_per_block(void) {
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

#if 1
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

                assert(gq_t_bits % 16 == 0);

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

#if QB == 4
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
#else
                for (int b = 0; b < QB; ++b) {
                    vst1q_u32((uint32_t *) ppt + 0,  p0[b]);
                    vst1q_u32((uint32_t *) ppt + 4,  p1[b]);

                    pp[b] = (ppt[0] | ppt[1] | ppt[2] | ppt[3]) | ((uint64_t) (ppt[4] | ppt[5] | ppt[6] | ppt[7])) << 32;
                }
#endif
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
    const int nb = quantize_2_blocks_per_row(n);
    const int nq = quantize_2_quants_per_block();

    const gq_scale_t * restrict pm0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pm1 = (const gq_scale_t *) y;

    const gq_scale_t * restrict pd0 = pm0 + nb;
    const gq_scale_t * restrict pd1 = pm1 + nb;

    const gq_quant_t * restrict pb0 = (const gq_quant_t *) (pd0 + nb);
    const gq_quant_t * restrict pb1 = (const gq_quant_t *) (pd1 + nb);

    float sumf = 0.0;

#if 1
    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

#if QB == 4
        int isum01 = 0;
        int isum10 = 0;
        int isum11 = 0;

        for (int s = 0; s < nq; ++s) {
            const gq_quant_t * restrict mm0 = pb0 + i*nq*QB + s*QB;
            const gq_quant_t * restrict mm1 = pb1 + i*nq*QB + s*QB;

#define bpcnt(x) __builtin_popcountll(x)
            isum01 += (1 << 0)*(bpcnt(mm1[0]));
            isum01 += (1 << 1)*(bpcnt(mm1[1]));
            isum01 += (1 << 2)*(bpcnt(mm1[2]));
            isum01 += (1 << 3)*(bpcnt(mm1[3]));

            isum10 += (1 << 0)*(bpcnt(mm0[0]));
            isum10 += (1 << 1)*(bpcnt(mm0[1]));
            isum10 += (1 << 2)*(bpcnt(mm0[2]));
            isum10 += (1 << 3)*(bpcnt(mm0[3]));

            isum11 += (1 << 0)*(bpcnt(mm0[0] & mm1[0]));
            isum11 += (1 << 1)*(bpcnt(mm0[0] & mm1[1]) + bpcnt(mm0[1] & mm1[0]));
            isum11 += (1 << 2)*(bpcnt(mm0[0] & mm1[2]) + bpcnt(mm0[1] & mm1[1]) + bpcnt(mm0[2] & mm1[0]));
            isum11 += (1 << 3)*(bpcnt(mm0[0] & mm1[3]) + bpcnt(mm0[1] & mm1[2]) + bpcnt(mm0[2] & mm1[1]) + bpcnt(mm0[3] & mm1[0]));
            isum11 += (1 << 4)*(bpcnt(mm0[1] & mm1[3]) + bpcnt(mm0[2] & mm1[2]) + bpcnt(mm0[3] & mm1[1]));
            isum11 += (1 << 5)*(bpcnt(mm0[2] & mm1[3]) + bpcnt(mm0[3] & mm1[2]));
            isum11 += (1 << 6)*(bpcnt(mm0[3] & mm1[3]));
#undef bpcnt
        }

        sumf += nq*gq_t_bits*(m0*m1) + isum01*(m0*d1) + isum10*(m1*d0) + isum11*(d0*d1);
#elif QB == 3
        int isum01 = 0;
        int isum10 = 0;
        int isum11 = 0;

        for (int s = 0; s < nq; ++s) {
            const gq_quant_t * restrict mm0 = pb0 + i*nq*QB + s*QB;
            const gq_quant_t * restrict mm1 = pb1 + i*nq*QB + s*QB;

#if gq_t_bits == 32
#define bpcnt(x) __builtin_popcount(x)
#else
#define bpcnt(x) __builtin_popcountll(x)
#endif
            isum01 += (1 << 0)*(bpcnt(mm1[0]));
            isum01 += (1 << 1)*(bpcnt(mm1[1]));
            isum01 += (1 << 2)*(bpcnt(mm1[2]));

            isum10 += (1 << 0)*(bpcnt(mm0[0]));
            isum10 += (1 << 1)*(bpcnt(mm0[1]));
            isum10 += (1 << 2)*(bpcnt(mm0[2]));

            isum11 += (1 << 0)*(bpcnt(mm0[0] & mm1[0]));
            isum11 += (1 << 1)*(bpcnt(mm0[0] & mm1[1]) + bpcnt(mm0[1] & mm1[0]));
            isum11 += (1 << 2)*(bpcnt(mm0[0] & mm1[2]) + bpcnt(mm0[1] & mm1[1]) + bpcnt(mm0[2] & mm1[0]));
            isum11 += (1 << 3)*(bpcnt(mm0[1] & mm1[2]) + bpcnt(mm0[2] & mm1[1]));
            isum11 += (1 << 4)*(bpcnt(mm0[2] & mm1[2]));
#undef bpcnt
        }

        sumf += nq*gq_t_bits*(m0*m1) + isum01*(m0*d1) + isum10*(m1*d0) + isum11*(d0*d1);
#elif QB == 2
        int isum01 = 0;
        int isum10 = 0;
        int isum11 = 0;

        for (int s = 0; s < nq; ++s) {
            const gq_quant_t * restrict mm0 = pb0 + i*nq*QB + s*QB;
            const gq_quant_t * restrict mm1 = pb1 + i*nq*QB + s*QB;

#if gq_t_bits == 32
#define bpcnt(x) __builtin_popcount(x)
#else
#define bpcnt(x) __builtin_popcountll(x)
#endif
            isum01 += (1 << 0)*(bpcnt(mm1[0]));
            isum01 += (1 << 1)*(bpcnt(mm1[1]));

            isum10 += (1 << 0)*(bpcnt(mm0[0]));
            isum10 += (1 << 1)*(bpcnt(mm0[1]));

            isum11 += (1 << 0)*(bpcnt(mm0[0] & mm1[0]));
            isum11 += (1 << 1)*(bpcnt(mm0[0] & mm1[1]) + bpcnt(mm0[1] & mm1[0]));
            isum11 += (1 << 2)*(bpcnt(mm0[1] & mm1[1]));
#undef bpcnt
        }

        sumf += nq*gq_t_bits*(m0*m1) + isum01*(m0*d1) + isum10*(m1*d0) + isum11*(d0*d1);
#else
        float s0[QB + 1];
        float s1[QB + 1];

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
                    sumf += s0[q0]*s1[q1]*__builtin_popcountll(mm0 & mm1);
                }
            }
        }
#endif
    }
#else
#error "not implemented"
#endif

    *s = sumf;
}

// use vec_dot_gq_2 to compute the dot product of two rows
void mul_mat_gq_2(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

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
// (does not work)
//

static inline int quantize_3_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_3_quants_per_block(void) {
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
                        int32x4_t r = vdupq_n_s32(-b);

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

#if QB == 4
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
#else
                for (int q = 0; q < QB; ++q) {
                    vst1q_u32((uint32_t *) ppt + 0,  p0[q]);
                    vst1q_u32((uint32_t *) ppt + 4,  p1[q]);

                    pp[q] = (ppt[0] | ppt[1] | ppt[2] | ppt[3]) | ((uint64_t) (ppt[4] | ppt[5] | ppt[6] | ppt[7])) << 32;
                }
#endif
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
    float sumf = 0.0f;

    const int nb = quantize_3_blocks_per_row(n);
    const int nq = quantize_3_quants_per_block();

    const gq_scale_t * restrict pd0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pd1 = (const gq_scale_t *) y;

    const gq_quant_t * restrict pb0 = (const gq_quant_t *) (pd0 + nb);
    const gq_quant_t * restrict pb1 = (const gq_quant_t *) (pd1 + nb);

#if 1
    for (int i = 0; i < nb; i++) {
        int isum = 0;

#if QB == 4
        for (int s = 0; s < nq; ++s) {
            const gq_quant_t * restrict m0 = pb0 + i*nq*QB + s*QB;
            const gq_quant_t * restrict m1 = pb1 + i*nq*QB + s*QB;

            isum += (1 << 0)*(__builtin_popcountll(m0[0] & m1[0]));
            isum += (1 << 1)*(__builtin_popcountll(m0[0] & m1[1]) + __builtin_popcountll(m0[1] & m1[0]));
            isum += (1 << 2)*(__builtin_popcountll(m0[0] & m1[2]) + __builtin_popcountll(m0[1] & m1[1]) + __builtin_popcountll(m0[2] & m1[0]));
            isum += (1 << 3)*(__builtin_popcountll(m0[0] & m1[3]) + __builtin_popcountll(m0[1] & m1[2]) + __builtin_popcountll(m0[2] & m1[1]) + __builtin_popcountll(m0[3] & m1[0]));
            isum += (1 << 4)*(__builtin_popcountll(m0[1] & m1[3]) + __builtin_popcountll(m0[2] & m1[2]) + __builtin_popcountll(m0[3] & m1[1]));
            isum += (1 << 5)*(__builtin_popcountll(m0[2] & m1[3]) + __builtin_popcountll(m0[3] & m1[2]));
            isum += (1 << 6)*(__builtin_popcountll(m0[3] & m1[3]));
        }
#else
        for (int s = 0; s < nq; ++s) {
            for (int q0 = 0; q0 < QB; q0++) {
                const gq_quant_t mm0 = pb0[i*nq*QB + s*QB + q0];
                for (int q1 = 0; q1 < QB; q1++) {
                    const gq_quant_t mm1 = pb1[i*nq*QB + s*QB + q1];
                    isum += (1 << (q0 + q1))*(__builtin_popcountll(mm0 & mm1));
                }
            }
        }
#endif

        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        sumf += d0*d1*isum;
    }
#else
#ifdef __ARM_NEON
    // gq_quant_t == uint64_t
    for (int i = 0; i < nb; i += 4) {
        int isum[4] = {0, 0, 0, 0};

        for (int k = 0; k < 4; ++k) {
            for (int s = 0; s < nq; ++s) {
                const gq_quant_t * restrict m0 = pb0 + (i+k)*nq*QB + s*QB;
                const gq_quant_t * restrict m1 = pb1 + (i+k)*nq*QB + s*QB;

#if QB == 4
#define bpcnt(x) __builtin_popcountll(x)
                //isum[k] += (1ULL << 0)*(bpcnt(m0[0] & m1[0])) +
                //           (1ULL << 1)*(bpcnt(m0[0] & m1[1]) + bpcnt(m0[1] & m1[0])) +
                //           (1ULL << 2)*(bpcnt(m0[0] & m1[2]) + bpcnt(m0[1] & m1[1]) + bpcnt(m0[2] & m1[0])) +
                //           (1ULL << 3)*(bpcnt(m0[0] & m1[3]) + bpcnt(m0[1] & m1[2]) + bpcnt(m0[2] & m1[1]) + bpcnt(m0[3] & m1[0])) +
                //           (1ULL << 4)*(bpcnt(m0[1] & m1[3]) + bpcnt(m0[2] & m1[2]) + bpcnt(m0[3] & m1[1])) +
                //           (1ULL << 5)*(bpcnt(m0[2] & m1[3]) + bpcnt(m0[3] & m1[2])) +
                //           (1ULL << 6)*(bpcnt(m0[3] & m1[3]));
#undef bpcnt

                const uint8x8_t m00 = vld1_u8((const uint8_t *) (m0 + 0));
                const uint8x8_t m01 = vld1_u8((const uint8_t *) (m0 + 1));
                const uint8x8_t m02 = vld1_u8((const uint8_t *) (m0 + 2));
                const uint8x8_t m03 = vld1_u8((const uint8_t *) (m0 + 3));

                const uint8x8_t m10 = vld1_u8((const uint8_t *) (m1 + 0));
                const uint8x8_t m11 = vld1_u8((const uint8_t *) (m1 + 1));
                const uint8x8_t m12 = vld1_u8((const uint8_t *) (m1 + 2));
                const uint8x8_t m13 = vld1_u8((const uint8_t *) (m1 + 3));

                const uint8x8_t m00m10 = vand_u8(m00, m10);

                const uint8x8_t m00m11 = vand_u8(m00, m11);
                const uint8x8_t m01m10 = vand_u8(m01, m10);

                const uint8x8_t m00m12 = vand_u8(m00, m12);
                const uint8x8_t m01m11 = vand_u8(m01, m11);
                const uint8x8_t m02m10 = vand_u8(m02, m10);

                const uint8x8_t m00m13 = vand_u8(m00, m13);
                const uint8x8_t m01m12 = vand_u8(m01, m12);
                const uint8x8_t m02m11 = vand_u8(m02, m11);
                const uint8x8_t m03m10 = vand_u8(m03, m10);

                const uint8x8_t m01m13 = vand_u8(m01, m13);
                const uint8x8_t m02m12 = vand_u8(m02, m12);
                const uint8x8_t m03m11 = vand_u8(m03, m11);

                const uint8x8_t m02m13 = vand_u8(m02, m13);
                const uint8x8_t m03m12 = vand_u8(m03, m12);

                const uint8x8_t m03m13 = vand_u8(m03, m13);

#define bpcnt(x) vaddv_u8(vcnt_u8(x))
                isum[k] += (1ULL << 0)*(bpcnt(m00m10)) +
                           (1ULL << 1)*(bpcnt(m00m11) + bpcnt(m01m10)) +
                           (1ULL << 2)*(bpcnt(m00m12) + bpcnt(m01m11) + bpcnt(m02m10)) +
                           (1ULL << 3)*(bpcnt(m00m13) + bpcnt(m01m12) + bpcnt(m02m11) + bpcnt(m03m10)) +
                           (1ULL << 4)*(bpcnt(m01m13) + bpcnt(m02m12) + bpcnt(m03m11)) +
                           (1ULL << 5)*(bpcnt(m02m13) + bpcnt(m03m12)) +
                           (1ULL << 6)*(bpcnt(m03m13));
#undef bpcnt
#else
                for (int q0 = 0; q0 < QB; q0++) {
                    const gq_quant_t mm0 = m0[q0];
                    for (int q1 = 0; q1 < QB; q1++) {
                        const gq_quant_t mm1 = m1[q1];
                        isum[k] += (1ULL << (q0 + q1))*(__builtin_popcountll(mm0 & mm1));
                    }
                }
#endif
            }
        }

        int32x4_t isumv = vld1q_s32(isum);

        float32x4_t d0v = vld1q_f32(pd0 + i);
        float32x4_t d1v = vld1q_f32(pd1 + i);

        float32x4_t sumfv = vmulq_f32(d0v, d1v);

        sumfv = vmulq_f32(sumfv, vcvtq_f32_s32(isumv));
        sumf += vaddvq_f32(sumfv);
    }
#else
#error "not implemented"
#endif

#endif
    *s = sumf;
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

//
// method 4
// 4-bit quantization
//

static inline int quantize_4_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_4_row_size(int k) {
    const int nb = quantize_4_blocks_per_row(k);

    return nb*(2*sizeof(gq_scale_t) + QK/2);
}

void quantize_4_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % QK == 0);
    assert(QB == 4);

    const int nb = quantize_4_blocks_per_row(k);

    gq_scale_t * restrict pm = (gq_scale_t *) (dst);
    gq_scale_t * restrict pd = (gq_scale_t *) (pm + nb);
    uint8_t    * restrict pb = (uint8_t *)    (pd + nb);

    uint8_t pp[QK/2];

    for (int i = 0; i < nb; i++) {
        memset(pp, 0, sizeof(pp));

        float min = FLT_MAX;
        float max = -FLT_MAX;

#if defined(__AVX2__)
        {
            assert(QK == 64);
            enum { QK8 = QK/8 };

            __m256 srcv[QK8];
            __m256 minv[QK8];
            __m256 maxv[QK8];

            for (int l = 0; l < QK8; l++) {
                srcv[l] = _mm256_loadu_ps(src + i*QK + 8*l);
            }

            for (int l = 0; l < QK8/2; l++) {
                minv[2*l] = _mm256_min_ps(srcv[2*l], srcv[2*l+1]);
                maxv[2*l] = _mm256_max_ps(srcv[2*l], srcv[2*l+1]);
            }

            for (int l = 0; l < QK8/4; l++) {
                minv[4*l] = _mm256_min_ps(minv[4*l], minv[4*l+2]);
                maxv[4*l] = _mm256_max_ps(maxv[4*l], maxv[4*l+2]);
            }

            for (int l = 0; l < QK8/8; l++) {
                minv[8*l] = _mm256_min_ps(minv[8*l], minv[8*l+4]);
                maxv[8*l] = _mm256_max_ps(maxv[8*l], maxv[8*l+4]);
            }

            //min = MIN(minv[0][0], MIN(minv[0][1], MIN(minv[0][2], MIN(minv[0][3], MIN(minv[0][4], MIN(minv[0][5], MIN(minv[0][6], minv[0][7])))))));
            //max = MAX(maxv[0][0], MAX(maxv[0][1], MAX(maxv[0][2], MAX(maxv[0][3], MAX(maxv[0][4], MAX(maxv[0][5], MAX(maxv[0][6], maxv[0][7])))))));

            const __m256 minv0_0 = _mm256_permute2f128_ps(minv[0], minv[0], 3);
            const __m256 minv0_1 = _mm256_min_ps(minv[0], minv0_0);
            const __m256 minv0_2 = _mm256_permute_ps(minv0_1, 0x4e);
            const __m256 minv0_3 = _mm256_min_ps(minv0_1, minv0_2);
            const __m256 minv0_4 = _mm256_permute_ps(minv0_3, 0xb1);
            const __m256 minv0_5 = _mm256_min_ps(minv0_3, minv0_4);

            const __m256 maxv0_0 = _mm256_permute2f128_ps(maxv[0], maxv[0], 3);
            const __m256 maxv0_1 = _mm256_max_ps(maxv[0], maxv0_0);
            const __m256 maxv0_2 = _mm256_permute_ps(maxv0_1, 0x4e);
            const __m256 maxv0_3 = _mm256_max_ps(maxv0_1, maxv0_2);
            const __m256 maxv0_4 = _mm256_permute_ps(maxv0_3, 0xb1);
            const __m256 maxv0_5 = _mm256_max_ps(maxv0_3, maxv0_4);

            min = _mm256_cvtss_f32(minv0_5);
            max = _mm256_cvtss_f32(maxv0_5);

            const float d = (max - min) / ((1 << QB) - 2);
            const float id = d ? 1.0/d : 0.0;

            pm[i] = GGML_FP32_TO_GQ(min);
            pd[i] = GGML_FP32_TO_GQ(d);

            const __m256 idv = _mm256_set1_ps(id);

            for (int l = 0; l < QK/8; l++) {
                __m256 v = _mm256_mul_ps(_mm256_sub_ps(srcv[l], _mm256_set1_ps(min)), idv);
#if 0
                v[0] += frand(); v[1] += frand(); v[2] += frand(); v[3] += frand();
                v[4] += frand(); v[5] += frand(); v[6] += frand(); v[7] += frand();
#endif

                // convert to uint8
                __m256i vi = _mm256_cvtps_epi32(v);

                uint32_t vi_0 = _mm256_extract_epi32(vi, 0);
                uint32_t vi_1 = _mm256_extract_epi32(vi, 1);
                uint32_t vi_2 = _mm256_extract_epi32(vi, 2);
                uint32_t vi_3 = _mm256_extract_epi32(vi, 3);

                uint32_t vi_4 = _mm256_extract_epi32(vi, 4);
                uint32_t vi_5 = _mm256_extract_epi32(vi, 5);
                uint32_t vi_6 = _mm256_extract_epi32(vi, 6);
                uint32_t vi_7 = _mm256_extract_epi32(vi, 7);

                // convert to 4-bit, 2 consecutive packed into 1 byte
                pp[4*l + 0] = vi_0 | (vi_1 << 4);
                pp[4*l + 1] = vi_2 | (vi_3 << 4);
                pp[4*l + 2] = vi_4 | (vi_5 << 4);
                pp[4*l + 3] = vi_6 | (vi_7 << 4);

                //printf("vi: %7d %7d %7d %7d %7d %7d %7d %7d\n", vi_0, vi_1, vi_2, vi_3, vi_4, vi_5, vi_6, vi_7);
                //printf("v : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            }

            memcpy(pb + i*QK/2, pp, sizeof(pp));
        }
#elif defined(__ARM_NEON) && 0
        {
            // TODO
        }
#else
        {
            for (int l = 0; l < QK; l++) {
                const float v = src[i*QK + l];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            const float d = (max - min) / ((1 << QB) - 1);
            const float id = d ? 1.0/d : 0.0;

            pm[i] = GGML_FP32_TO_GQ(min);
            pd[i] = GGML_FP32_TO_GQ(d);

            for (int l = 0; l < QK; l++) {
                const float v = (src[i*QK + l] - min) * id;
                const uint8_t vi = (uint8_t) (v + frand());
                pp[l/2] |= (vi & 0xf) << (4*(l & 1));
            }

            memcpy(pb + i*QK/2, pp, sizeof(pp));
        }
#endif
        //printf("min %f max %f\n", min, max);
    }
}

// reimplementation of quantize_4 using quantize_4_row
void quantize_4(const float * restrict src, char * restrict dst, int n, int k) {
    assert(k % QK == 0);

    for (int j = 0; j < n; j++) {
        quantize_4_row(src + j*k, dst, k);
        dst = (char *) dst + quantize_4_row_size(k);
    }
}

void vec_dot_gq_4(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = quantize_4_blocks_per_row(n);

    const gq_scale_t * restrict pm0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pm1 = (const gq_scale_t *) y;

    const gq_scale_t * restrict pd0 = pm0 + nb;
    const gq_scale_t * restrict pd1 = pm1 + nb;

    const uint8_t * restrict pb0 = (const uint8_t *) (pd0 + nb);
    const uint8_t * restrict pb1 = (const uint8_t *) (pd1 + nb);

    float sumf = 0.0;

#if 0
    // scalar
    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        for (int j = 0; j < QK/2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*(v0 & 0xf) + m0;
            const float f1 = d0*(v0 >> 4)  + m0;

            const float f2 = d1*(v1 & 0xf) + m1;
            const float f3 = d1*(v1 >> 4)  + m1;

            sumf += f0*f2 + f1*f3;
        }
    }
#else
#if defined(__AVX2__)
#if QK == 64 && 0
    __m256 sumv0 = _mm256_setzero_ps();
    __m256 sumv1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        const __m256 m0v = _mm256_set1_ps(m0);
        const __m256 d0v = _mm256_set1_ps(d0);

        const __m256 m1v = _mm256_set1_ps(m1);
        const __m256 d1v = _mm256_set1_ps(d1);

        const __m256i m4b = _mm256_set1_epi8(0xf);

        __m256i v0 = _mm256_loadu_si256((__m256i *) p0);

        //_mm_prefetch((const char *) (p0 + 32), _MM_HINT_T0);
        //_mm_prefetch((const char *) (p1 + 32), _MM_HINT_T0);
        //_mm_prefetch((const char *) (pm0 + i + 1), _MM_HINT_T0);
        //_mm_prefetch((const char *) (pm1 + i + 1), _MM_HINT_T0);
        //_mm_prefetch((const char *) (pd0 + i + 1), _MM_HINT_T0);
        //_mm_prefetch((const char *) (pd1 + i + 1), _MM_HINT_T0);

        __m256i v00 = _mm256_and_si256(v0, _mm256_set1_epi32(0x000000FF));
        __m256i v01 = _mm256_srli_epi32(_mm256_and_si256(v0, _mm256_set1_epi32(0x0000FFFF)), 8);
        __m256i v02 = _mm256_srli_epi32(_mm256_and_si256(v0, _mm256_set1_epi32(0x00FFFFFF)), 16);
        __m256i v03 = _mm256_srli_epi32(v0, 24);

        //////////////////////

        //{
        //    uint32_t vi_0 = _mm256_extract_epi32(v00, 0);
        //    uint32_t vi_1 = _mm256_extract_epi32(v00, 1);
        //    uint32_t vi_2 = _mm256_extract_epi32(v00, 2);
        //    uint32_t vi_3 = _mm256_extract_epi32(v00, 3);
        //    uint32_t vi_4 = _mm256_extract_epi32(v00, 4);
        //    uint32_t vi_5 = _mm256_extract_epi32(v00, 5);
        //    uint32_t vi_6 = _mm256_extract_epi32(v00, 6);
        //    uint32_t vi_7 = _mm256_extract_epi32(v00, 7);
        //    printf("v0: %7d %7d %7d %7d %7d %7d %7d %7d\n", vi_0, vi_1, vi_2, vi_3, vi_4, vi_5, vi_6, vi_7);
        //    printf("p0: %7d %7d %7d %7d %7d %7d %7d %7d\n", p0[0], p0[4], p0[8], p0[12], p0[16], p0[20], p0[24], p0[28]);
        //    printf("p1: %7d %7d %7d %7d %7d %7d %7d %7d\n", p0[1], p0[5], p0[9], p0[13], p0[17], p0[21], p0[25], p0[29]);
        //    printf("p2: %7d %7d %7d %7d %7d %7d %7d %7d\n", p0[2], p0[6], p0[10], p0[14], p0[18], p0[22], p0[26], p0[30]);
        //    printf("p3: %7d %7d %7d %7d %7d %7d %7d %7d\n", p0[3], p0[7], p0[11], p0[15], p0[19], p0[23], p0[27], p0[31]);
        //}

        // compute 32 x 4-bit values (low and high)
        __m256i v00l = _mm256_and_si256(v00, m4b);
        __m256i v01l = _mm256_and_si256(v01, m4b);
        __m256i v02l = _mm256_and_si256(v02, m4b);
        __m256i v03l = _mm256_and_si256(v03, m4b);

        __m256i v00h = _mm256_srli_epi32(v00, 4);
        __m256i v01h = _mm256_srli_epi32(v01, 4);
        __m256i v02h = _mm256_srli_epi32(v02, 4);
        __m256i v03h = _mm256_srli_epi32(v03, 4);

        //{
        //    uint32_t vi_0 = _mm256_extract_epi32(v00l, 0);
        //    uint32_t vi_1 = _mm256_extract_epi32(v00l, 1);
        //    uint32_t vi_2 = _mm256_extract_epi32(v00l, 2);
        //    uint32_t vi_3 = _mm256_extract_epi32(v00l, 3);
        //    uint32_t vi_4 = _mm256_extract_epi32(v00l, 4);
        //    uint32_t vi_5 = _mm256_extract_epi32(v00l, 5);
        //    uint32_t vi_6 = _mm256_extract_epi32(v00l, 6);
        //    uint32_t vi_7 = _mm256_extract_epi32(v00l, 7);

        //    printf("v0l: %7d %7d %7d %7d %7d %7d %7d %7d\n", vi_0, vi_1, vi_2, vi_3, vi_4, vi_5, vi_6, vi_7);

        //    vi_0 = _mm256_extract_epi32(v00h, 0);
        //    vi_1 = _mm256_extract_epi32(v00h, 1);
        //    vi_2 = _mm256_extract_epi32(v00h, 2);
        //    vi_3 = _mm256_extract_epi32(v00h, 3);
        //    vi_4 = _mm256_extract_epi32(v00h, 4);
        //    vi_5 = _mm256_extract_epi32(v00h, 5);
        //    vi_6 = _mm256_extract_epi32(v00h, 6);
        //    vi_7 = _mm256_extract_epi32(v00h, 7);

        //    printf("v0h: %7d %7d %7d %7d %7d %7d %7d %7d\n", vi_0, vi_1, vi_2, vi_3, vi_4, vi_5, vi_6, vi_7);
        //}

        // convert to float
        __m256 vf00l = _mm256_cvtepi32_ps(v00l);
        __m256 vf01l = _mm256_cvtepi32_ps(v01l);
        __m256 vf02l = _mm256_cvtepi32_ps(v02l);
        __m256 vf03l = _mm256_cvtepi32_ps(v03l);

        __m256 vf00h = _mm256_cvtepi32_ps(v00h);
        __m256 vf01h = _mm256_cvtepi32_ps(v01h);
        __m256 vf02h = _mm256_cvtepi32_ps(v02h);
        __m256 vf03h = _mm256_cvtepi32_ps(v03h);

        //{
        //    printf("vf00l: %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n", vf00l[0], vf00l[1], vf00l[2], vf00l[3], vf00l[4], vf00l[5], vf00l[6], vf00l[7]);
        //    printf("vf01l: %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n", vf01l[0], vf01l[1], vf01l[2], vf01l[3], vf01l[4], vf01l[5], vf01l[6], vf01l[7]);
        //    printf("vf02l: %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n", vf02l[0], vf02l[1], vf02l[2], vf02l[3], vf02l[4], vf02l[5], vf02l[6], vf02l[7]);
        //    printf("vf03l: %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n", vf03l[0], vf03l[1], vf03l[2], vf03l[3], vf03l[4], vf03l[5], vf03l[6], vf03l[7]);
        //}

        // multiply by scale and add offset
        vf00l = _mm256_fmadd_ps(vf00l, d0v, m0v);
        vf01l = _mm256_fmadd_ps(vf01l, d0v, m0v);
        vf02l = _mm256_fmadd_ps(vf02l, d0v, m0v);
        vf03l = _mm256_fmadd_ps(vf03l, d0v, m0v);

        vf00h = _mm256_fmadd_ps(vf00h, d0v, m0v);
        vf01h = _mm256_fmadd_ps(vf01h, d0v, m0v);
        vf02h = _mm256_fmadd_ps(vf02h, d0v, m0v);
        vf03h = _mm256_fmadd_ps(vf03h, d0v, m0v);

        __m256i v1 = _mm256_loadu_si256((__m256i *) p1);

        __m256i v10 = _mm256_and_si256(v1, _mm256_set1_epi32(0x000000FF));
        __m256i v11 = _mm256_srli_epi32(_mm256_and_si256(v1, _mm256_set1_epi32(0x0000FFFF)), 8);
        __m256i v12 = _mm256_srli_epi32(_mm256_and_si256(v1, _mm256_set1_epi32(0x00FFFFFF)), 16);
        __m256i v13 = _mm256_srli_epi32(v1, 24);

        __m256i v10l = _mm256_and_si256(v10, m4b);
        __m256i v11l = _mm256_and_si256(v11, m4b);
        __m256i v12l = _mm256_and_si256(v12, m4b);
        __m256i v13l = _mm256_and_si256(v13, m4b);

        __m256i v10h = _mm256_srli_epi32(v10, 4);
        __m256i v11h = _mm256_srli_epi32(v11, 4);
        __m256i v12h = _mm256_srli_epi32(v12, 4);
        __m256i v13h = _mm256_srli_epi32(v13, 4);

        __m256 vf10l = _mm256_cvtepi32_ps(v10l);
        __m256 vf11l = _mm256_cvtepi32_ps(v11l);
        __m256 vf12l = _mm256_cvtepi32_ps(v12l);
        __m256 vf13l = _mm256_cvtepi32_ps(v13l);

        __m256 vf10h = _mm256_cvtepi32_ps(v10h);
        __m256 vf11h = _mm256_cvtepi32_ps(v11h);
        __m256 vf12h = _mm256_cvtepi32_ps(v12h);
        __m256 vf13h = _mm256_cvtepi32_ps(v13h);

        vf10l = _mm256_fmadd_ps(vf10l, d1v, m1v);
        vf11l = _mm256_fmadd_ps(vf11l, d1v, m1v);
        vf12l = _mm256_fmadd_ps(vf12l, d1v, m1v);
        vf13l = _mm256_fmadd_ps(vf13l, d1v, m1v);

        vf10h = _mm256_fmadd_ps(vf10h, d1v, m1v);
        vf11h = _mm256_fmadd_ps(vf11h, d1v, m1v);
        vf12h = _mm256_fmadd_ps(vf12h, d1v, m1v);
        vf13h = _mm256_fmadd_ps(vf13h, d1v, m1v);

        // compute dot product
        sumv0 = _mm256_fmadd_ps(vf00l, vf10l, sumv0);
        sumv0 = _mm256_fmadd_ps(vf01l, vf11l, sumv0);
        sumv0 = _mm256_fmadd_ps(vf02l, vf12l, sumv0);
        sumv0 = _mm256_fmadd_ps(vf03l, vf13l, sumv0);

        sumv1 = _mm256_fmadd_ps(vf00h, vf10h, sumv1);
        sumv1 = _mm256_fmadd_ps(vf01h, vf11h, sumv1);
        sumv1 = _mm256_fmadd_ps(vf02h, vf12h, sumv1);
        sumv1 = _mm256_fmadd_ps(vf03h, vf13h, sumv1);
    }

    // accumulate (horizontal sum)
    const __m256 vdot = _mm256_add_ps(sumv0, sumv1);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(vdot), _mm256_extractf128_ps(vdot, 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);

    sumf += _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
#elif QK == 64 && 0
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    const __m256i m4b = _mm256_set1_epi8(0xf);

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        // 64 x 4
        const __m256i v0 = _mm256_loadu_si256((__m256i *) p0);
        const __m256i v1 = _mm256_loadu_si256((__m256i *) p1);

        // 32 x 8
        const __m256i v0l = _mm256_and_si256(v0, m4b);
        const __m256i v1l = _mm256_and_si256(v1, m4b);

        const __m256i v0h = _mm256_and_si256(_mm256_srli_epi16(v0, 4), m4b);
        const __m256i v1h = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m4b);

        const __m256i pl = _mm256_maddubs_epi16(v0l, v1l);
        const __m256i ph = _mm256_maddubs_epi16(v0h, v1h);

        const __m256i p16 = _mm256_add_epi16(ph, pl);
        const __m256i p = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

        sum00 += m0*m1;
        sum01 += m1*d0*(_mm256_hadd_epi8_gg(_mm256_add_epi8(v0l, v0h)));
        sum10 += m0*d1*(_mm256_hadd_epi8_gg(_mm256_add_epi8(v1l, v1h)));
        sum11 += d0*d1*(_mm256_hadd_epi32_gg(p));
    }

    sumf = 64.0*sum00 + sum01 + sum10 + sum11;
#elif QK == 64 && 1 // this is the best when using min + d
    float sum00 = 0.0f;

    __m256 sum01 = _mm256_setzero_ps();
    __m256 sum10 = _mm256_setzero_ps();
    __m256 sum11 = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        const __m256 m0v = _mm256_set1_ps(m0);
        const __m256 d0v = _mm256_set1_ps(d0);

        const __m256 m1v = _mm256_set1_ps(m1);
        const __m256 d1v = _mm256_set1_ps(d1);

        const __m256 m1d0v = _mm256_mul_ps(m1v, d0v);
        const __m256 m0d1v = _mm256_mul_ps(m0v, d1v);
        const __m256 d0d1v = _mm256_mul_ps(d0v, d1v);

        const __m256i m4b = _mm256_set1_epi8(0xf);

        // 64 x 4
        const __m256i v0 = _mm256_loadu_si256((__m256i *) p0);
        const __m256i v1 = _mm256_loadu_si256((__m256i *) p1);

        // 32 x 8
        const __m256i v0l = _mm256_and_si256(v0, m4b);
        const __m256i v1l = _mm256_and_si256(v1, m4b);

        const __m256i v0h = _mm256_and_si256(_mm256_srli_epi16(v0, 4), m4b);
        const __m256i v1h = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m4b);

        const __m256i v0a = _mm256_add_epi8(v0l, v0h);
        const __m256i v1a = _mm256_add_epi8(v1l, v1h);

        const __m128i v0al = _mm256_extracti128_si256(v0a, 0);
        const __m128i v0ah = _mm256_extracti128_si256(v0a, 1);

        const __m128i v1al = _mm256_extracti128_si256(v1a, 0);
        const __m128i v1ah = _mm256_extracti128_si256(v1a, 1);

        const __m128i v0as = _mm_add_epi8(v0al, v0ah);
        const __m128i v1as = _mm_add_epi8(v1al, v1ah);

        const __m256i v0as_0 = _mm256_cvtepu8_epi32(v0as);
        const __m256i v0as_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(v0as, 8));

        const __m256i v1as_0 = _mm256_cvtepu8_epi32(v1as);
        const __m256i v1as_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(v1as, 8));

        const __m256i v0ass = _mm256_add_epi32(v0as_0, v0as_1);
        const __m256i v1ass = _mm256_add_epi32(v1as_0, v1as_1);

        const __m256 v0f = _mm256_cvtepi32_ps(v0ass);
        const __m256 v1f = _mm256_cvtepi32_ps(v1ass);

        const __m256i pl = _mm256_maddubs_epi16(v0l, v1l);
        const __m256i ph = _mm256_maddubs_epi16(v0h, v1h);

        const __m256i p16 = _mm256_add_epi16(ph, pl);
        const __m256i p = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

        sum00 += m0*m1;
        sum01 = _mm256_fmadd_ps(m1d0v, v0f, sum01);
        sum10 = _mm256_fmadd_ps(m0d1v, v1f, sum10);
        sum11 = _mm256_fmadd_ps(d0d1v, _mm256_cvtepi32_ps(p), sum11);
    }

    sumf = 64.0*sum00 + _mm256_hadd_ps_gg(sum01) + _mm256_hadd_ps_gg(sum10) + _mm256_hadd_ps_gg(sum11);
#endif
#elif defined (__ARM_NEON)
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        const uint8x16_t m4b = vdupq_n_u8(0xf);

        const uint8x16_t v0_0 = vld1q_u8(p0);
        const uint8x16_t v0_1 = vld1q_u8(p0 + 16);
        const uint8x16_t v1_0 = vld1q_u8(p1);
        const uint8x16_t v1_1 = vld1q_u8(p1 + 16);

        // and with 0xf
        const uint8x16_t v0_0l = vandq_u8(v0_0, m4b);
        const uint8x16_t v0_1l = vandq_u8(v0_1, m4b);
        const uint8x16_t v1_0l = vandq_u8(v1_0, m4b);
        const uint8x16_t v1_1l = vandq_u8(v1_1, m4b);

        const uint8x16_t v0_0h = vshrq_n_u8(v0_0, 4);
        const uint8x16_t v0_1h = vshrq_n_u8(v0_1, 4);
        const uint8x16_t v1_0h = vshrq_n_u8(v1_0, 4);
        const uint8x16_t v1_1h = vshrq_n_u8(v1_1, 4);

        // dot product into uint16x8_t
        const uint16x8_t pl0l = vmull_u8(vget_low_u8 (v0_0l), vget_low_u8 (v1_0l));
        const uint16x8_t pl0h = vmull_u8(vget_high_u8(v0_0l), vget_high_u8(v1_0l));
        const uint16x8_t pl1l = vmull_u8(vget_low_u8 (v0_1l), vget_low_u8 (v1_1l));
        const uint16x8_t pl1h = vmull_u8(vget_high_u8(v0_1l), vget_high_u8(v1_1l));

        const uint16x8_t ph0l = vmull_u8(vget_low_u8 (v0_0h), vget_low_u8 (v1_0h));
        const uint16x8_t ph0h = vmull_u8(vget_high_u8(v0_0h), vget_high_u8(v1_0h));
        const uint16x8_t ph1l = vmull_u8(vget_low_u8 (v0_1h), vget_low_u8 (v1_1h));
        const uint16x8_t ph1h = vmull_u8(vget_high_u8(v0_1h), vget_high_u8(v1_1h));

        const uint16x8_t pl0 = vaddq_u16(pl0l, pl0h);
        const uint16x8_t pl1 = vaddq_u16(pl1l, pl1h);
        const uint16x8_t ph0 = vaddq_u16(ph0l, ph0h);
        const uint16x8_t ph1 = vaddq_u16(ph1l, ph1h);

        const uint16x8_t pl = vaddq_u16(pl0, pl1);
        const uint16x8_t ph = vaddq_u16(ph0, ph1);

        sum00 += m0*m1;
        sum01 += m1*d0*(vaddvq_u8(v0_0l) + vaddvq_u8(v0_0h) + vaddvq_u8(v0_1l) + vaddvq_u8(v0_1h));
        sum10 += m0*d1*(vaddvq_u8(v1_0l) + vaddvq_u8(v1_0h) + vaddvq_u8(v1_1l) + vaddvq_u8(v1_1h));
        //sum11 += d0*d1*(
        //        vaddvq_u16(vaddq_u16(vaddq_u16(pl0l, pl0h), vaddq_u16(pl1l, pl1h))) +
        //        vaddvq_u16(vaddq_u16(vaddq_u16(ph0l, ph0h), vaddq_u16(ph1l, ph1h))));
        sum11 += d0*d1*vaddvq_u16(vaddq_u16(pl, ph));
    }

    sumf = 64.0*sum00 + sum01 + sum10 + sum11;
#endif
#endif

    *s = sumf;
}

// use vec_dot_gq_4 to compute the dot product of two rows
void mul_mat_gq_4(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

    const int nb = quantize_4_blocks_per_row(k);

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_4(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_4_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_4_row_size(k);
        src1 = (const char *) src1 - n*quantize_4_row_size(k);

        dst = (float *) dst + n;
    }
}

//
// method 5
// 4-bit quantization (without min, only delta)
//

static inline int quantize_5_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_5_row_size(int k) {
    const int nb = quantize_5_blocks_per_row(k);

    return nb*(sizeof(gq_scale_t) + QK/2);
}

void quantize_5_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % QK == 0);
    assert(QB == 4);

    const int nb = quantize_5_blocks_per_row(k);

    gq_scale_t * restrict pd = (gq_scale_t *) (dst);
    uint8_t    * restrict pb = (uint8_t *)    (pd + nb);

    uint8_t pp[QK/2];

    for (int i = 0; i < nb; i++) {
        memset(pp, 0, sizeof(pp));

        float amax = 0.0f; // absolute max

#if defined(__AVX2__)
        {
            assert(QK == 64);
            enum { QK8 = QK/8 };

            __m256 srcv [QK8];
            __m256 asrcv[QK8];
            __m256 amaxv[QK8];

            for (int l = 0; l < QK8; l++) {
                srcv[l]  = _mm256_loadu_ps(src + i*QK + 8*l);
            }

            for (int l = 0; l < QK8; l++) {
                asrcv[l] = _mm256_and_ps(srcv[l], _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
            }


            for (int l = 0; l < QK8/2; l++) {
                amaxv[2*l] = _mm256_max_ps(asrcv[2*l], asrcv[2*l+1]);
            }

            for (int l = 0; l < QK8/4; l++) {
                amaxv[4*l] = _mm256_max_ps(amaxv[4*l], amaxv[4*l+2]);
            }

            for (int l = 0; l < QK8/8; l++) {
                amaxv[8*l] = _mm256_max_ps(amaxv[8*l], amaxv[8*l+4]);
            }

            //amax = MAX(amaxv[0][0], MAX(amaxv[0][1], MAX(amaxv[0][2], MAX(amaxv[0][3], MAX(amaxv[0][4], MAX(amaxv[0][5], MAX(amaxv[0][6], amaxv[0][7])))))));

            const __m256 amaxv0_0 = _mm256_permute2f128_ps(amaxv[0], amaxv[0], 3);
            const __m256 amaxv0_1 = _mm256_max_ps(amaxv[0], amaxv0_0);
            const __m256 amaxv0_2 = _mm256_permute_ps(amaxv0_1, 0x4e);
            const __m256 amaxv0_3 = _mm256_max_ps(amaxv0_1, amaxv0_2);
            const __m256 amaxv0_4 = _mm256_permute_ps(amaxv0_3, 0xb1);
            const __m256 amaxv0_5 = _mm256_max_ps(amaxv0_3, amaxv0_4);

            amax = _mm256_cvtss_f32(amaxv0_5);

            //printf("amax = %f\n", amax);

            const float d = amax / ((1 << (QB - 1)) - 1);
            const float id = d ? 1.0/d : 0.0;

            pd[i] = GGML_FP32_TO_GQ(d);

            const __m256 idv = _mm256_set1_ps(id);

            for (int l = 0; l < QK/8; l++) {
                __m256 v = _mm256_mul_ps(srcv[l], idv);
#if 0
                v[0] += frand(); v[1] += frand(); v[2] += frand(); v[3] += frand();
                v[4] += frand(); v[5] += frand(); v[6] += frand(); v[7] += frand();
#endif

                // convert to int8
                __m256i vi = _mm256_cvtps_epi32(v);
                vi = _mm256_add_epi32(vi, _mm256_set1_epi32(8));

                int32_t vi_0 = _mm256_extract_epi32(vi, 0);
                int32_t vi_1 = _mm256_extract_epi32(vi, 1);
                int32_t vi_2 = _mm256_extract_epi32(vi, 2);
                int32_t vi_3 = _mm256_extract_epi32(vi, 3);

                int32_t vi_4 = _mm256_extract_epi32(vi, 4);
                int32_t vi_5 = _mm256_extract_epi32(vi, 5);
                int32_t vi_6 = _mm256_extract_epi32(vi, 6);
                int32_t vi_7 = _mm256_extract_epi32(vi, 7);

                // convert to 4-bit, 2 consecutive packed into 1 byte
                pp[4*l + 0] = vi_0 | (vi_1 << 4);
                pp[4*l + 1] = vi_2 | (vi_3 << 4);
                pp[4*l + 2] = vi_4 | (vi_5 << 4);
                pp[4*l + 3] = vi_6 | (vi_7 << 4);

                //printf("vi: %7d %7d %7d %7d %7d %7d %7d %7d\n", vi_0, vi_1, vi_2, vi_3, vi_4, vi_5, vi_6, vi_7);
                ////printf("v : %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

                assert(vi_0 >= 0 && vi_0 < 16);
                assert(vi_1 >= 0 && vi_1 < 16);
                assert(vi_2 >= 0 && vi_2 < 16);
                assert(vi_3 >= 0 && vi_3 < 16);

                assert(vi_4 >= 0 && vi_4 < 16);
                assert(vi_5 >= 0 && vi_5 < 16);
                assert(vi_6 >= 0 && vi_6 < 16);
                assert(vi_7 >= 0 && vi_7 < 16);
            }

            memcpy(pb + i*QK/2, pp, sizeof(pp));
        }
#elif defined(__ARM_NEON) && 0
        {
            // TODO
        }
#else
        {
            for (int l = 0; l < QK; l++) {
                const float v = src[i*QK + l];
                amax = MAX(amax, fabsf(v));
            }

            const float d = amax / ((1 << (QB - 1)) - 1);
            const float id = d ? 1.0/d : 0.0;

            pd[i] = GGML_FP32_TO_GQ(d);

            for (int l = 0; l < QK; l++) {
                const float v = src[i*QK + l]*id;
                const int8_t vi = ((int8_t) (round(v))) + 8;
                assert(vi >= 0 && vi < 16);
                pp[l/2] |= (vi & 0xf) << (4*(l & 1));
            }

            memcpy(pb + i*QK/2, pp, sizeof(pp));
        }
#endif
        //printf("min %f max %f\n", min, max);
    }
}

// reimplementation of quantize_5 using quantize_5_row
void quantize_5(const float * restrict src, char * restrict dst, int n, int k) {
    assert(k % QK == 0);

    for (int j = 0; j < n; j++) {
        quantize_5_row(src + j*k, dst, k);
        dst = (char *) dst + quantize_5_row_size(k);
    }
}

void vec_dot_gq_5(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = quantize_5_blocks_per_row(n);

    const gq_scale_t * restrict pd0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pd1 = (const gq_scale_t *) y;

    const uint8_t * restrict pb0 = (const uint8_t *) (pd0 + nb);
    const uint8_t * restrict pb1 = (const uint8_t *) (pd1 + nb);

    float sumf = 0.0;

#if 0
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        for (int j = 0; j < QK/2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*((int8_t) (v0 & 0xf) - 8);
            const float f1 = d0*((int8_t) (v0 >> 4)  - 8);

            const float f2 = d1*((int8_t) (v1 & 0xf) - 8);
            const float f3 = d1*((int8_t) (v1 >> 4)  - 8);

            sumf += f0*f2 + f1*f3;
        }
    }
#else
#if defined(__AVX2__)
#if QK == 64 && 1
    __m256 sum11 = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        const __m256 d0v = _mm256_set1_ps(d0);
        const __m256 d1v = _mm256_set1_ps(d1);

        const __m256 d0d1v = _mm256_mul_ps(d0v, d1v);

        const __m256i m4b = _mm256_set1_epi8(0xf);

        // 64 x 4
        const __m256i v0 = _mm256_loadu_si256((__m256i *) p0);
        const __m256i v1 = _mm256_loadu_si256((__m256i *) p1);

        // 32 x 8
        __m256i v0l = _mm256_and_si256(v0, m4b);
        __m256i v1l = _mm256_and_si256(v1, m4b);

        __m256i v0h = _mm256_and_si256(_mm256_srli_epi16(v0, 4), m4b);
        __m256i v1h = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m4b);

        // sub 8
        v0l = _mm256_sub_epi8(v0l, _mm256_set1_epi8(8));
        v0h = _mm256_sub_epi8(v0h, _mm256_set1_epi8(8));

        v1l = _mm256_sub_epi8(v1l, _mm256_set1_epi8(8));
        v1h = _mm256_sub_epi8(v1h, _mm256_set1_epi8(8));

        // abs
        const __m256i v0la = _mm256_sign_epi8(v0l, v0l);
        const __m256i v0ha = _mm256_sign_epi8(v0h, v0h);

        // sign
        const __m256i v1ls = _mm256_sign_epi8(v1l, v0l);
        const __m256i v1hs = _mm256_sign_epi8(v1h, v0h);

        const __m256i pl = _mm256_maddubs_epi16(v0la, v1ls);
        const __m256i ph = _mm256_maddubs_epi16(v0ha, v1hs);

        const __m256i p16 = _mm256_add_epi16(ph, pl);
        const __m256i p = _mm256_madd_epi16(_mm256_set1_epi16(1), p16);

        sum11 = _mm256_fmadd_ps(d0d1v, _mm256_cvtepi32_ps(p), sum11);
    }

    sumf = _mm256_hadd_ps_gg(sum11);
#endif
#elif defined (__ARM_NEON)
    float sum11 = 0.0f;

    //float32x4_t sum_0 = vdupq_n_f32(0.0f);
    //float32x4_t sum_1 = vdupq_n_f32(0.0f);

    //float16x8_t sum_0 = vdupq_n_f16(0.0f);
    //float16x8_t sum_1 = vdupq_n_f16(0.0f);

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        //float32x4_t d0d1v = vdupq_n_f32(d0*d1);
        //float16x8_t d0d1v = vdupq_n_f16(d0*d1);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        const uint8x16_t m4b = vdupq_n_u8(0xf);
        const int8x16_t s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(p0);
        const uint8x16_t v0_1 = vld1q_u8(p0 + 16);
        const uint8x16_t v1_0 = vld1q_u8(p1);
        const uint8x16_t v1_1 = vld1q_u8(p1 + 16);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v1_0l = vreinterpretq_s8_u8(vandq_u8(v1_0, m4b));
        const int8x16_t v1_1l = vreinterpretq_s8_u8(vandq_u8(v1_1, m4b));

        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));
        const int8x16_t v1_0h = vreinterpretq_s8_u8(vshrq_n_u8(v1_0, 4));
        const int8x16_t v1_1h = vreinterpretq_s8_u8(vshrq_n_u8(v1_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v1_0ls = vsubq_s8(v1_0l, s8b);
        const int8x16_t v1_1ls = vsubq_s8(v1_1l, s8b);

        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);
        const int8x16_t v1_0hs = vsubq_s8(v1_0h, s8b);
        const int8x16_t v1_1hs = vsubq_s8(v1_1h, s8b);

        // dot product into int16x8_t
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));
        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1ls));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1ls));

        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1hs));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1hs));

        const int16x8_t pl0 = vaddq_s16(pl0l, pl0h);
        const int16x8_t pl1 = vaddq_s16(pl1l, pl1h);
        const int16x8_t ph0 = vaddq_s16(ph0l, ph0h);
        const int16x8_t ph1 = vaddq_s16(ph1l, ph1h);

        const int16x8_t pl = vaddq_s16(pl0, pl1);
        const int16x8_t ph = vaddq_s16(ph0, ph1);

        //const int8x16_t pl0 = vmulq_s8(v0_0ls, v1_0ls);
        //const int8x16_t pl1 = vmulq_s8(v0_1ls, v1_1ls);
        //const int8x16_t ph0 = vmulq_s8(v0_0hs, v1_0hs);
        //const int8x16_t ph1 = vmulq_s8(v0_1hs, v1_1hs);

        //const int16x8_t pll = vaddl_s8(vget_low_s8(pl0),  vget_low_s8(pl1));
        //const int16x8_t plh = vaddl_s8(vget_high_s8(pl0), vget_high_s8(pl1));
        //const int16x8_t phl = vaddl_s8(vget_low_s8(ph0),  vget_low_s8(ph1));
        //const int16x8_t phh = vaddl_s8(vget_high_s8(ph0), vget_high_s8(ph1));

        //const int16x8_t pl = vaddq_s16(pll, plh);
        //const int16x8_t ph = vaddq_s16(phl, phh);

        const int16x8_t p = vaddq_s16(pl, ph);

        // convert to float
        //const float32x4_t pf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (p)));
        //const float32x4_t pf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(p)));

        // scalar
        sum11 += d0*d1*vaddvq_s16(p);
        //sum11 += d0*d1*(vaddvq_s16(pl) + vaddvq_s16(ph));
        //sum11 += d0*d1*vaddvq_s16(vaddq_s16(pl, ph));
        //sum11 += d0*d1*(vaddvq_s8(pl0) + vaddvq_s8(pl1) + vaddvq_s8(ph0) + vaddvq_s8(ph1));
        //sum11 += d0*d1*(vaddvq_s16(pll) + vaddvq_s16(plh) + vaddvq_s16(phl) + vaddvq_s16(phh));

        //sum_0 = vfmaq_f16(sum_0, d0d1v, vcvtq_f16_s16(p));
        //sum_0 = vfmaq_f16(sum_0, d0d1v, vcvtq_f16_s16(pl));
        //sum_1 = vfmaq_f16(sum_1, d0d1v, vcvtq_f16_s16(ph));

        // vectorize
        //sum_0 = vmlaq_f32(sum_0, d0d1v, pf0);
        //sum_1 = vmlaq_f32(sum_1, d0d1v, pf1);
    }

    sumf = sum11;
    //sumf = vaddvq_f32(sum_0) + vaddvq_f32(sum_1);
    //sumf = sum_0[0] + sum_0[1] + sum_0[2] + sum_0[3] + sum_0[4] + sum_0[5] + sum_0[6] + sum_0[7];
    //sum_0 = vaddq_f16(sum_0, sum_1);
    //sumf = sum_0[0] + sum_0[1] + sum_0[2] + sum_0[3] + sum_0[4] + sum_0[5] + sum_0[6] + sum_0[7];
#endif
#endif

    *s = sumf;
}

// use vec_dot_gq_5 to compute the dot product of two rows
void mul_mat_gq_5(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

    const int nb = quantize_5_blocks_per_row(k);

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_5(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_5_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_5_row_size(k);
        src1 = (const char *) src1 - n*quantize_5_row_size(k);

        dst = (float *) dst + n;
    }
}

//
// method 6
// same as 5 but with 32 element blocks
//

static inline int quantize_6_blocks_per_row(int k) {
    return k/32;
}

static inline int quantize_6_row_size(int k) {
    const int nb = quantize_6_blocks_per_row(k);

    return nb*(sizeof(gq_scale_t) + 16);
}

void quantize_6_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % 32 == 0);
    assert(QB == 4);

    const int nb = quantize_6_blocks_per_row(k);

    gq_scale_t * restrict pd = (gq_scale_t *) (dst);
    uint8_t    * restrict pb = (uint8_t *)    (pd + nb);

    uint8_t pp[16];

    for (int i = 0; i < nb; i++) {
        memset(pp, 0, sizeof(pp));

        float amax = 0.0f; // absolute max

#if defined(__AVX2__)
        {
            enum { QK8 = 4 };

            __m256 srcv [QK8];
            __m256 asrcv[QK8];
            __m256 amaxv[QK8];

            for (int l = 0; l < QK8; l++) {
                srcv[l]  = _mm256_loadu_ps(src + i*32 + 8*l);
            }

            for (int l = 0; l < QK8; l++) {
                asrcv[l] = _mm256_and_ps(srcv[l], _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
            }

            for (int l = 0; l < QK8/2; l++) {
                amaxv[2*l] = _mm256_max_ps(asrcv[2*l], asrcv[2*l+1]);
            }

            for (int l = 0; l < QK8/4; l++) {
                amaxv[4*l] = _mm256_max_ps(amaxv[4*l], amaxv[4*l+2]);
            }

            const __m256 amaxv0_0 = _mm256_permute2f128_ps(amaxv[0], amaxv[0], 3);
            const __m256 amaxv0_1 = _mm256_max_ps(amaxv[0], amaxv0_0);
            const __m256 amaxv0_2 = _mm256_permute_ps(amaxv0_1, 0x4e);
            const __m256 amaxv0_3 = _mm256_max_ps(amaxv0_1, amaxv0_2);
            const __m256 amaxv0_4 = _mm256_permute_ps(amaxv0_3, 0xb1);
            const __m256 amaxv0_5 = _mm256_max_ps(amaxv0_3, amaxv0_4);

            amax = _mm256_cvtss_f32(amaxv0_5);

            const float d = amax / ((1 << (QB - 1)) - 1);
            const float id = d ? 1.0/d : 0.0;

            pd[i] = GGML_FP32_TO_GQ(d);

            const __m256 idv = _mm256_set1_ps(id);

            for (int l = 0; l < 4; l++) {
                __m256 v = _mm256_mul_ps(srcv[l], idv);

                // convert to int8
                __m256i vi = _mm256_cvtps_epi32(v);
                vi = _mm256_add_epi32(vi, _mm256_set1_epi32(8));

                int32_t vi_0 = _mm256_extract_epi32(vi, 0);
                int32_t vi_1 = _mm256_extract_epi32(vi, 1);
                int32_t vi_2 = _mm256_extract_epi32(vi, 2);
                int32_t vi_3 = _mm256_extract_epi32(vi, 3);

                int32_t vi_4 = _mm256_extract_epi32(vi, 4);
                int32_t vi_5 = _mm256_extract_epi32(vi, 5);
                int32_t vi_6 = _mm256_extract_epi32(vi, 6);
                int32_t vi_7 = _mm256_extract_epi32(vi, 7);

                // convert to 4-bit, 2 consecutive packed into 1 byte
                pp[4*l + 0] = vi_0 | (vi_1 << 4);
                pp[4*l + 1] = vi_2 | (vi_3 << 4);
                pp[4*l + 2] = vi_4 | (vi_5 << 4);
                pp[4*l + 3] = vi_6 | (vi_7 << 4);

                assert(vi_0 >= 0 && vi_0 < 16);
                assert(vi_1 >= 0 && vi_1 < 16);
                assert(vi_2 >= 0 && vi_2 < 16);
                assert(vi_3 >= 0 && vi_3 < 16);

                assert(vi_4 >= 0 && vi_4 < 16);
                assert(vi_5 >= 0 && vi_5 < 16);
                assert(vi_6 >= 0 && vi_6 < 16);
                assert(vi_7 >= 0 && vi_7 < 16);
            }

            memcpy(pb + i*16, pp, sizeof(pp));
        }
#elif defined(__ARM_NEON)
        {
            float32x4_t srcv [8];
            float32x4_t asrcv[8];
            float32x4_t amaxv[8];

            for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(src + i*32 + 4*l);
            for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

            for (int l = 0; l < 4; l++) amaxv[2*l] = vmaxq_f32(asrcv[2*l], asrcv[2*l+1]);
            for (int l = 0; l < 2; l++) amaxv[4*l] = vmaxq_f32(amaxv[4*l], amaxv[4*l+2]);
            for (int l = 0; l < 1; l++) amaxv[8*l] = vmaxq_f32(amaxv[8*l], amaxv[8*l+4]);

            amax = MAX(
                    MAX(vgetq_lane_f32(amaxv[0], 0), vgetq_lane_f32(amaxv[0], 1)),
                    MAX(vgetq_lane_f32(amaxv[0], 2), vgetq_lane_f32(amaxv[0], 3)));

            const float d = amax / ((1 << 3) - 1);
            const float id = d ? 1.0/d : 0.0;

            pd[i] = GGML_FP32_TO_GQ(d);

            for (int l = 0; l < 8; l++) {
                const float32x4_t v = vmulq_n_f32(srcv[l], id);
                const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(8.5f));
                const int32x4_t vi = vcvtq_s32_f32(vf);

                pp[2*l + 0] = vgetq_lane_s32(vi, 0) | (vgetq_lane_s32(vi, 1) << 4);
                pp[2*l + 1] = vgetq_lane_s32(vi, 2) | (vgetq_lane_s32(vi, 3) << 4);
            }

            memcpy(pb + i*16, pp, sizeof(pp));
        }
#else
        {
            for (int l = 0; l < 32; l++) {
                const float v = src[i*32 + l];
                amax = MAX(amax, fabsf(v));
            }

            const float d = amax / ((1 << (QB - 1)) - 1);
            const float id = d ? 1.0/d : 0.0;

            pd[i] = GGML_FP32_TO_GQ(d);

            for (int l = 0; l < 32; l++) {
                const float v = src[i*32 + l]*id;
                const int8_t vi = ((int8_t) (round(v))) + 8;
                assert(vi >= 0 && vi < 16);
                pp[l/2] |= (vi & 0xf) << (4*(l & 1));
            }

            memcpy(pb + i*16, pp, sizeof(pp));
        }
#endif
        //printf("amax = %f\n", amax);
    }
}

// reimplementation of quantize__6using quantize_6_row
void quantize_6(const float * restrict src, char * restrict dst, int n, int k) {
    assert(k % 32 == 0);

    for (int j = 0; j < n; j++) {
        quantize_6_row(src + j*k, dst, k);
        dst = (char *) dst + quantize_6_row_size(k);
    }
}

void vec_dot_gq_6(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = quantize_6_blocks_per_row(n);

    const gq_scale_t * restrict pd0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pd1 = (const gq_scale_t *) y;

    const uint8_t * restrict pb0 = (const uint8_t *) (pd0 + nb);
    const uint8_t * restrict pb1 = (const uint8_t *) (pd1 + nb);

    float sumf = 0.0;

#if 0
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*16;
        const uint8_t * restrict p1 = pb1 + i*16;

        for (int j = 0; j < 16; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*((int8_t) (v0 & 0xf) - 8);
            const float f1 = d0*((int8_t) (v0 >> 4)  - 8);

            const float f2 = d1*((int8_t) (v1 & 0xf) - 8);
            const float f3 = d1*((int8_t) (v1 >> 4)  - 8);

            sumf += f0*f2 + f1*f3;
        }
    }
#else
#if defined(__AVX2__)
    // TODO
#elif defined (__ARM_NEON)
#if 0
    float sum0 = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        //float32x4_t d0d1v = vdupq_n_f32(d0*d1);
        //float16x8_t d0d1v = vdupq_n_f16(d0*d1);

        const uint8_t * restrict p0 = pb0 + i*16;
        const uint8_t * restrict p1 = pb1 + i*16;

        const uint8x16_t m4b = vdupq_n_u8(0xf);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(p0);
        const uint8x16_t v1_0 = vld1q_u8(p1);

        // 4-bit -> 8-bit
        const uint8x16_t v0_0l = vandq_u8(v0_0, m4b);
        const uint8x16_t v1_0l = vandq_u8(v1_0, m4b);

        const uint8x16_t v0_0h = vshrq_n_u8(v0_0, 4);
        const uint8x16_t v1_0h = vshrq_n_u8(v1_0, 4);

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v1_0ls = vsubq_s8(v1_0l, s8b);

        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v1_0hs = vsubq_s8(v1_0h, s8b);

        // dot product into int16x8_t
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));

        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));

        const int16x8_t pl = vaddq_s16(pl0l, pl0h);
        const int16x8_t ph = vaddq_s16(ph0l, ph0h);

        const int16x8_t p = vaddq_s16(pl, ph);

        // scalar
        sum0 += d0*d1*vaddvq_s16(p);
    }

    sumf = sum0;
#elif 1 // this is a bit faster than the above
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int i = 0; i < nb; i += 2) {
        const float d0_0 = GGML_GQ_TO_FP32(pd0[i + 0]);
        const float d1_0 = GGML_GQ_TO_FP32(pd1[i + 0]);
        const float d0_1 = GGML_GQ_TO_FP32(pd0[i + 1]);
        const float d1_1 = GGML_GQ_TO_FP32(pd1[i + 1]);

        const uint8_t * restrict p0 = pb0 + i*16;
        const uint8_t * restrict p1 = pb1 + i*16;

        const uint8x16_t m4b = vdupq_n_u8(0xf);
        const int8x16_t s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(p0);
        const uint8x16_t v0_1 = vld1q_u8(p0 + 16);
        const uint8x16_t v1_0 = vld1q_u8(p1);
        const uint8x16_t v1_1 = vld1q_u8(p1 + 16);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v1_0l = vreinterpretq_s8_u8(vandq_u8(v1_0, m4b));

        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v1_0h = vreinterpretq_s8_u8(vshrq_n_u8(v1_0, 4));

        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        const int8x16_t v1_1l = vreinterpretq_s8_u8(vandq_u8(v1_1, m4b));

        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));
        const int8x16_t v1_1h = vreinterpretq_s8_u8(vshrq_n_u8(v1_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v1_0ls = vsubq_s8(v1_0l, s8b);

        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v1_0hs = vsubq_s8(v1_0h, s8b);

        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v1_1ls = vsubq_s8(v1_1l, s8b);

        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);
        const int8x16_t v1_1hs = vsubq_s8(v1_1h, s8b);

        // dot product into int16x8_t
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));

        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1ls));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1ls));

        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1hs));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1hs));

        const int16x8_t pl_0 = vaddq_s16(pl0l, pl0h);
        const int16x8_t ph_0 = vaddq_s16(ph0l, ph0h);

        const int16x8_t pl_1 = vaddq_s16(pl1l, pl1h);
        const int16x8_t ph_1 = vaddq_s16(ph1l, ph1h);

        const int16x8_t p_0 = vaddq_s16(pl_0, ph_0);
        const int16x8_t p_1 = vaddq_s16(pl_1, ph_1);

        // scalar
        sum0 += d0_0*d1_0*vaddvq_s16(p_0);
        sum1 += d0_1*d1_1*vaddvq_s16(p_1);
    }

    sumf = sum0 + sum1;
#endif
#endif
#endif

    *s = sumf;
}

// use vec_dot_gq_6 to compute the dot product of two rows
void mul_mat_gq_6(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % 32 == 0);

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_6(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_6_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_6_row_size(k);
        src1 = (const char *) src1 - n*quantize_6_row_size(k);

        dst = (float *) dst + n;
    }
}

int main(int argc, const char ** argv) {
    assert(sizeof(gq_quant_t)*8 == gq_t_bits);
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    int method = 0;
    if (argc > 1) {
        method = atoi(argv[1]);
    }

    float * src0 = malloc(sizeof(float)*M*K);
    float * src1 = malloc(sizeof(float)*N*K);
    float * dst  = malloc(sizeof(float)*M*N);

    // allocate aligned memory
    //float * src0 = (float *)aligned_alloc(32, sizeof(float)*M*K);
    //float * src1 = (float *)aligned_alloc(32, sizeof(float)*N*K);
    //float * dst  = (float *)aligned_alloc(32, sizeof(float)*M*N);

    for (int i = 0; i < M*K; i++) {
        src0[i] = 0.8 - rand() / (float)RAND_MAX;
        /*src0[i] = rand() / (float)RAND_MAX;*/
        /*src0[i] = i % 2;*/
    }

    for (int i = 0; i < N*K; i++) {
        src1[i] = 0.8 - rand() / (float)RAND_MAX;
        /*src1[i] = rand() / (float)RAND_MAX;*/
        /*src1[i] = i % 3;*/
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

        if (method == 4) {
            src0_gq = calloc(1, quantize_4_row_size(K)*M);
            src1_gq = calloc(1, quantize_4_row_size(K)*N);

            sizegq  = quantize_4_row_size(K)*M + quantize_4_row_size(K)*N;
        }

        if (method == 5) {
            src0_gq = calloc(1, quantize_5_row_size(K)*M);
            src1_gq = calloc(1, quantize_5_row_size(K)*N);

            sizegq  = quantize_5_row_size(K)*M + quantize_5_row_size(K)*N;
        }

        if (method == 6) {
            src0_gq = calloc(1, quantize_6_row_size(K)*M);
            src1_gq = calloc(1, quantize_6_row_size(K)*N);

            sizegq  = quantize_6_row_size(K)*M + quantize_6_row_size(K)*N;
        }
    }

    const size_t sizef16 = sizeof(ggml_fp16_t)*M*K + sizeof(ggml_fp16_t)*N*K;

    printf("compression: %f\n", (float)sizegq/sizef16);

    // convert fp32 -> gq
    {
        const int64_t t_start = ggml_time_us();

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

        if (method == 4) {
            quantize_4(src0, src0_gq, M, K);
            quantize_4(src1, src1_gq, N, K);
        }

        if (method == 5) {
            quantize_5(src0, src0_gq, M, K);
            quantize_5(src1, src1_gq, N, K);
        }

        if (method == 6) {
            quantize_6(src0, src0_gq, M, K);
            quantize_6(src1, src1_gq, N, K);
        }

        const int64_t t_end = ggml_time_us();
        printf("convert time: %f ms / method = %d\n", (t_end - t_start) / 1000.0, method);
    }

    for (int i = 0; i < 16; ++i) {
        printf("%f %f\n", src0[i], src1[i]);
    }

    const int nIter = 1;

    const int64_t start = ggml_cycles();
    const int64_t start_us = ggml_time_us();

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

        if (method == 4) {
            mul_mat_gq_4(src0_gq, src1_gq, dst, M, N, K);
        }

        if (method == 5) {
            mul_mat_gq_5(src0_gq, src1_gq, dst, M, N, K);
        }

        if (method == 6) {
            mul_mat_gq_6(src0_gq, src1_gq, dst, M, N, K);
        }
    }

    for (int i = 0; i < N; i++) {
        sum += dst[i]*iM;
    }

    {
        const int64_t end = ggml_cycles();
        const int64_t end_us = ggml_time_us();
        printf("%s: elapsed ticks: %" PRIu64 "\n",  __func__, end - start);
        printf("%s: elapsed us:    %d / %f ms\n",  __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0 / nIter);
    }

#if 0
    // print src0
    printf("src0:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%4.1f ", src0[i*K+j]);
        }
        printf("\n");
    }

    // print src1
    printf("src1:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%4.1f ", src1[i*K+j]);
        }
        printf("\n");
    }

    printf("dst:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4.1f ", dst[i*N+j]);
        }
        printf("\n");
    }
#endif

    printf("%f\n", sum);

    free(src0);
    free(src1);
    free(dst);

    if (src0_gq) free(src0_gq);
    if (src1_gq) free(src1_gq);

    return 0;
}
