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
#elif defined(__AVX2__)
#include "immintrin.h"
#endif

#ifndef MIN
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

const int M = 1280;
const int N = 1536;
const int K = 1280;

//const int M = 64;
//const int N = 64;
//const int K = 64;

const int QK = 64;
#define QB 7

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

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK; l++) {
            const float v = src[i*QK + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << QB) - 1);
        const float id = d ? 1.0/d : 0.0;

        pm[i] = GGML_FP32_TO_GQ(min);
        pd[i] = GGML_FP32_TO_GQ(d);

        for (int s = 0; s < nq; ++s) {
            memset(pp, 0, sizeof(pp));

            for (int l = 0; l < gq_t_bits; l++) {
                const   float v = src[i*QK + s*gq_t_bits + l];
                const uint8_t q = (v - min)*id;

                for (int b = 0; b < QB; b++) {
                    pp[b] |= q & (1 << b) ? (1ULL << l) : 0;
                }
            }

            for (int b = 0; b < QB; b++) {
                pb[i*nq*QB + s*QB + b] = pp[b];
            }
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
// method 3 - 4-bit quantization based on Clover
// ref: https://github.com/astojanov/Clover
//

static const uint32_t clover_1st_bit_set_32  = 0x80000000U;
static const uint32_t clover_1st_bit_off_32  = 0x7FFFFFFFU;

static inline float frand() {
    return (float) rand() / (float) RAND_MAX;
}

static inline int quantize_3_blocks_per_row(int k) {
    return k/64;
}

static inline int quantize_3_row_size(int k) {
    const int nb = quantize_3_blocks_per_row(k);

    return (nb + nb%2)*(sizeof(float) + 32);
}

void quantize_3_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % 64 == 0);
    const int nb = quantize_3_blocks_per_row(k);

    float  * dsts =  (float *) (dst);
    int8_t * dstq = (int8_t *) (dsts + nb + nb%2);

    for (int j = 0; j < nb; ++j) {
        const float * srcp = src  + j*64;
             int8_t * dstp = dstq + j*32;

        float amax = srcp[0];
        for (int i = 1; i < 64; ++i) {
            amax = fmaxf(amax, fabsf(srcp[i]));
        }

        dsts[j] = amax;

        const float iscale = 7.0f/amax;

        for (int i = 0; i < 64; i += 2) {
            const float u1 = srcp[i + 0];
            const float u2 = srcp[i + 1];

            const float r1 = frand();
            const float r2 = frand();

            /*const float r1 = 0.0f;*/
            /*const float r2 = 0.0f;*/

            const int8_t u_sgn1  = (int8_t) 1 + ((int8_t) (*(int32_t *) &u1 >> 31) << 1);
            const int8_t u_sgn2  = (int8_t) 1 + ((int8_t) (*(int32_t *) &u2 >> 31) << 1);

            const uint32_t u_abs1 = clover_1st_bit_off_32 & *(uint32_t *) &u1;
            const uint32_t u_abs2 = clover_1st_bit_off_32 & *(uint32_t *) &u2;

            const float v_abs1 = *(float *) &u_abs1;
            const float v_abs2 = *(float *) &u_abs2;

            /*const int8_t q_abs1 = (int8_t) floorf(_mm_fmadd_ss(v_abs1, iscale, r1));*/
            /*const int8_t q_abs2 = (int8_t) floorf(_mm_fmadd_ss(v_abs2, iscale, r2));*/
            const int8_t q_abs1 = (int8_t) floorf(v_abs1*iscale + r1);
            const int8_t q_abs2 = (int8_t) floorf(v_abs2*iscale + r2);

            const int8_t q_1 = (q_abs1 * u_sgn1) << 4;
            const int8_t q_2 = (q_abs2 * u_sgn2) & (int8_t) 0xF;

            //printf("q_1 = %d, q_2 = %d, amax = %f\n", q_1, q_2, amax);

            dstp[i >> 1] = q_1 | q_2;
        }
    }

    //printf("%d %d %d %d %d %d %d %d\n", dstq[0], dstq[1], dstq[2], dstq[3], dstq[4], dstq[5], dstq[6], dstq[7]);
}

void quantize_3(const float * restrict src, char * restrict dst, int n, int k) {
    for (int i = 0; i < n; ++i) {
        quantize_3_row(src + i*k, dst, k);
        dst += quantize_3_row_size(k);
    }
}

void vec_dot_4q(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = quantize_3_blocks_per_row(n);

    float * su = (float *) x;
    float * sv = (float *) y;

    int8_t * u = (int8_t *) (su + nb + nb%2);
    int8_t * v = (int8_t *) (sv + nb + nb%2);

    float result = 0;

    float rcp_49 = 1.0f / 49.0f;

    for (uint64_t b = 0; b < nb; ++b) {
        const uint64_t offset = b * 32;
        int16_t acc = 0;

        for (uint64_t idx = 0; idx < 32; ++idx) {
            const uint64_t i = idx + offset;

            const int8_t qu_p = u[i];
            const int8_t qv_p = v[i];

            const int8_t qu_1 = qu_p >> 4;
            const int8_t qu_2 = ((int8_t)(qu_p << 4)) >> 4;
            const int8_t qv_1 = qv_p >> 4;
            const int8_t qv_2 = ((int8_t)(qv_p << 4)) >> 4;

            acc += (int16_t)(qu_1 * qv_1) + (int16_t)(qu_2 * qv_2);
        }

        const float scaled_rcp_ss = (su[b] / 7.0f) * (sv[b] / 7.0f);
        result += scaled_rcp_ss * (float) acc;
    }

    *s = result;
}

void mul_mat_4q(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

    const int nb = quantize_3_blocks_per_row(k);

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_4q(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_3_row_size(k);
        }

        src0 = (const char *) src0 +   quantize_3_row_size(k);
        src1 = (const char *) src1 - n*quantize_3_row_size(k);

        dst = (float *) dst + n;
    }
}

//
// method 4 - 4-bit SIMD quantization based on Clover
// ref: https://github.com/astojanov/Clover
//

static inline int quantize_4_blocks_per_row(int k) {
    return k/64;
}

static inline int quantize_4_row_size(int k) {
    const int nb = quantize_4_blocks_per_row(k);

    return (nb + nb%2)*(sizeof(float) + 32);
}

static __m256i clover_mm256_1st_bit_off_epi8;
static __m256i clover_mm256_1st_bit_set_epi8;
static __m256  clover_mm256_1st_bit_set_ps;
static __m256  clover_mm256_1st_bit_off_ps;

static __m256i clover_mm256_mask_1st_epi32;

static __m256i clover_mm256_1_epi16;
static __m256  clover_mm256_1_ps;
static __m256  clover_mm256_7_ps;
static __m256  clover_mm256_127_ps;
static __m256  clover_mm256_rcp_7_ps;
static __m256  clover_mm256_rcp_127_ps;
static __m256  clover_mm256_rcp_49_ps;
static __m256  clover_mm256_rcp_2pow31_ps;

static __m256i clover_mm256_8bit_perm_lo;
static __m256i clover_mm256_8bit_perm_hi;

static __m256i clover_mm256_8bit_restore_perm_lo;
static __m256i clover_mm256_8bit_restore_perm_hi;

//
// Calculate the horizontal max in a given AVX vector
//
static inline float _mm256_hmaxf32_ps(const __m256 tmp3)
{
    const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
    const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
    const __m128 tmp6 = _mm_max_ps(tmp4, tmp5);
    const __m128 tmp7 = _mm_shuffle_ps(tmp6, tmp6, 78);
    const __m128 tmp8 = _mm_max_ps(tmp6, tmp7);
    const __m128 tmp9 = _mm_permute_ps(tmp8, 1);
    const __m128 tmp0 = _mm_max_ps(tmp8, tmp9);
    //
    // Return the result stored in the first element
    //
    return _mm_cvtss_f32(tmp0);
}

//
// Calculate the horizontal min in a given AVX vector
//
static inline float _mm256_hminf32_ps(const __m256 tmp3)
{
    const __m128 tmp4 = _mm256_castps256_ps128(tmp3);
    const __m128 tmp5 = _mm256_extractf128_ps(tmp3, 1);
    const __m128 tmp6 = _mm_min_ps(tmp4, tmp5);
    const __m128 tmp7 = _mm_shuffle_ps(tmp6, tmp6, 78);
    const __m128 tmp8 = _mm_min_ps(tmp6, tmp7);
    const __m128 tmp9 = _mm_permute_ps(tmp8, 1);
    const __m128 tmp0 = _mm_min_ps(tmp8, tmp9);
    //
    // Return the result stored in the first element
    //
    return _mm_cvtss_f32(tmp0);
}


//
// For a given vector __m256 of 8 floats, perform reduction
//
static inline float _mm256_haddf32_ps(__m256 acc)
{
    const __m128 left  = _mm256_extractf128_ps(acc, 1);
    const __m128 right = _mm256_castps256_ps128(acc);
    const __m128 x128  = _mm_add_ps(left, right);
    const __m128 x64   = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32   = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return  _mm_cvtss_f32(x32);
}

//
// Transpose 8x8 registers
//
static inline void _mm256_transpose8_epi32(
        __m256i *r0, __m256i *r1, __m256i *r2, __m256i *r3,
        __m256i *r4, __m256i *r5, __m256i *r6, __m256i *r7){
    __m256 u0, u1, u2, u3, u4, u5, u6, u7;
    __m256 s0, s1, s2, s3, s4, s5, s6, s7;

    u0 = (__m256) _mm256_unpacklo_epi32(*r0, *r1);
    u1 = (__m256) _mm256_unpackhi_epi32(*r0, *r1);
    u2 = (__m256) _mm256_unpacklo_epi32(*r2, *r3);
    u3 = (__m256) _mm256_unpackhi_epi32(*r2, *r3);
    u4 = (__m256) _mm256_unpacklo_epi32(*r4, *r5);
    u5 = (__m256) _mm256_unpackhi_epi32(*r4, *r5);
    u6 = (__m256) _mm256_unpacklo_epi32(*r6, *r7);
    u7 = (__m256) _mm256_unpackhi_epi32(*r6, *r7);

    s0 = _mm256_shuffle_ps(u0,u2,_MM_SHUFFLE(1,0,1,0));
    s1 = _mm256_shuffle_ps(u0,u2,_MM_SHUFFLE(3,2,3,2));
    s2 = _mm256_shuffle_ps(u1,u3,_MM_SHUFFLE(1,0,1,0));
    s3 = _mm256_shuffle_ps(u1,u3,_MM_SHUFFLE(3,2,3,2));
    s4 = _mm256_shuffle_ps(u4,u6,_MM_SHUFFLE(1,0,1,0));
    s5 = _mm256_shuffle_ps(u4,u6,_MM_SHUFFLE(3,2,3,2));
    s6 = _mm256_shuffle_ps(u5,u7,_MM_SHUFFLE(1,0,1,0));
    s7 = _mm256_shuffle_ps(u5,u7,_MM_SHUFFLE(3,2,3,2));

    *r0 = (__m256i) _mm256_permute2f128_ps(s0, s4, 0x20);
    *r1 = (__m256i) _mm256_permute2f128_ps(s1, s5, 0x20);
    *r2 = (__m256i) _mm256_permute2f128_ps(s2, s6, 0x20);
    *r3 = (__m256i) _mm256_permute2f128_ps(s3, s7, 0x20);
    *r4 = (__m256i) _mm256_permute2f128_ps(s0, s4, 0x31);
    *r5 = (__m256i) _mm256_permute2f128_ps(s1, s5, 0x31);
    *r6 = (__m256i) _mm256_permute2f128_ps(s2, s6, 0x31);
    *r7 = (__m256i) _mm256_permute2f128_ps(s3, s7, 0x31);
}

static inline __m256 _mm256_hmax_ps(const __m256 hmax_0) {
    const __m256 hmax_1 = _mm256_permute2f128_ps(hmax_0, hmax_0, 3);
    const __m256 hmax_2 = _mm256_max_ps(hmax_0, hmax_1);
    const __m256 hmax_3 = _mm256_permute_ps(hmax_2, 0x4E);
    const __m256 hmax_4 = _mm256_max_ps(hmax_2, hmax_3);
    const __m256 hmax_5 = _mm256_permute_ps(hmax_4, 0xB1);
    const __m256 hmax_6 = _mm256_max_ps(hmax_4, hmax_5);
    return hmax_6;
}

void quantize_4_row(const float * restrict src, void * restrict dst, int k) {
    assert(k % 64 == 0);
    const int nb = quantize_4_blocks_per_row(k);

    float  * dsts =  (float *) (dst);
    int8_t * dstq = (int8_t *) (dsts + nb + nb%2);

    const float * u = src;

    for (uint64_t b = 0; b < nb; b += 1) {
        const uint64_t offset = b * 64;
        const float * u1 = u + offset;
        const float * u2 = u1 + 64;

        const __m256 u_1 = _mm256_loadu_ps(u1 +  0);
        const __m256 u_2 = _mm256_loadu_ps(u1 +  8);
        const __m256 u_3 = _mm256_loadu_ps(u1 + 16);
        const __m256 u_4 = _mm256_loadu_ps(u1 + 24);
        const __m256 u_5 = _mm256_loadu_ps(u1 + 32);
        const __m256 u_6 = _mm256_loadu_ps(u1 + 40);
        const __m256 u_7 = _mm256_loadu_ps(u1 + 48);
        const __m256 u_8 = _mm256_loadu_ps(u1 + 56);
        //
        // Get the absolute values of each
        //
        const __m256 u_abs_1 = _mm256_and_ps(u_1, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_2 = _mm256_and_ps(u_2, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_3 = _mm256_and_ps(u_3, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_4 = _mm256_and_ps(u_4, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_5 = _mm256_and_ps(u_5, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_6 = _mm256_and_ps(u_6, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_7 = _mm256_and_ps(u_7, clover_mm256_1st_bit_off_ps);
        const __m256 u_abs_8 = _mm256_and_ps(u_8, clover_mm256_1st_bit_off_ps);
        //
        // Find the maximum
        //
        const __m256 m1 = _mm256_max_ps(u_abs_1, u_abs_2);
        const __m256 m2 = _mm256_max_ps(u_abs_3, u_abs_4);
        const __m256 m3 = _mm256_max_ps(u_abs_5, u_abs_6);
        const __m256 m4 = _mm256_max_ps(u_abs_7, u_abs_8);
        const __m256 m5 = _mm256_max_ps(m1, m2);
        const __m256 m6 = _mm256_max_ps(m3, m4);
        const __m256 m7 = _mm256_max_ps(m5, m6);

        //
        // Perform horizontal reduction, and make sure that the max is broadcasted in
        // all slots of the 256 bit lane
        //
        const __m256 hmax_5 = _mm256_hmax_ps(m7);

        //
        // Normalize if max is zero
        //
        const __m256i isZero = _mm256_cmpeq_epi32((__m256i) hmax_5, _mm256_setzero_si256());
        const __m256  cndOne = (__m256) _mm256_and_si256((__m256i) clover_mm256_1_ps, isZero);
        const __m256  hmax_6 = _mm256_add_ps(cndOne, hmax_5);

        //
        // Finally we have the scale
        //
        const __m256 scale = _mm256_div_ps(clover_mm256_7_ps, hmax_6);

        //
        // Store the scale to the right place
        //
        _mm256_maskstore_ps(dsts + b, clover_mm256_mask_1st_epi32, hmax_6);

#ifndef CLOVER_STOCHASTIC_ROUNDING_ENABLED
        //const __m256 rnd_1 = _mm256_setzero_ps();
        //const __m256 rnd_2 = _mm256_setzero_ps();
        //const __m256 rnd_3 = _mm256_setzero_ps();
        //const __m256 rnd_4 = _mm256_setzero_ps();
        //const __m256 rnd_5 = _mm256_setzero_ps();
        //const __m256 rnd_6 = _mm256_setzero_ps();
        //const __m256 rnd_7 = _mm256_setzero_ps();
        //const __m256 rnd_8 = _mm256_setzero_ps();

        // TODO: this is slow !!!!!
        const __m256 rnd_1 = _mm256_set1_ps(frand());
        const __m256 rnd_2 = _mm256_set1_ps(frand());
        const __m256 rnd_3 = _mm256_set1_ps(frand());
        const __m256 rnd_4 = _mm256_set1_ps(frand());
        const __m256 rnd_5 = _mm256_set1_ps(frand());
        const __m256 rnd_6 = _mm256_set1_ps(frand());
        const __m256 rnd_7 = _mm256_set1_ps(frand());
        const __m256 rnd_8 = _mm256_set1_ps(frand());
#else
        //
        // Get the first set of 32 random numbers
        //
        const __m256i rnd_xor1 = avx_xorshift128plus(random_key1, random_key2);

        const __m256i rnd_i8_1 = _mm256_and_si256(rnd_xor1, clover_mm256_1st_bit_off_epi8);
        const __m256i rnd_i8_2 = _mm256_slli_epi32(rnd_i8_1,  8);
        const __m256i rnd_i8_3 = _mm256_slli_epi32(rnd_i8_1, 16);
        const __m256i rnd_i8_4 = _mm256_slli_epi32(rnd_i8_1, 24);

        const __m256  rnd_f8_1 = _mm256_cvtepi32_ps(rnd_i8_1);
        const __m256  rnd_f8_2 = _mm256_cvtepi32_ps(rnd_i8_2);
        const __m256  rnd_f8_3 = _mm256_cvtepi32_ps(rnd_i8_3);
        const __m256  rnd_f8_4 = _mm256_cvtepi32_ps(rnd_i8_4);

        const __m256  rnd_1 = _mm256_mul_ps (rnd_f8_1, clover_mm256_rcp_2pow31_ps);
        const __m256  rnd_2 = _mm256_mul_ps (rnd_f8_2, clover_mm256_rcp_2pow31_ps);
        const __m256  rnd_3 = _mm256_mul_ps (rnd_f8_3, clover_mm256_rcp_2pow31_ps);
        const __m256  rnd_4 = _mm256_mul_ps (rnd_f8_4, clover_mm256_rcp_2pow31_ps);

        //
        // Meanwhile, keep busy the pre-fetcher
        //
        _mm_prefetch((char *)(u2 + 16), _MM_HINT_T0);
        _mm_prefetch((char *)(u2 + 32), _MM_HINT_T0);
        _mm_prefetch((char *)(u2 + 48), _MM_HINT_T0);
        _mm_prefetch((char *)(u2 + 64), _MM_HINT_T0);


        //
        // Get the second set of 32 random numbers
        //
        const __m256i rnd_xor2 = avx_xorshift128plus(random_key1, random_key2);

        const __m256i rnd_i8_5 = _mm256_and_si256(rnd_xor2, clover_mm256_1st_bit_off_epi8);
        const __m256i rnd_i8_6 = _mm256_slli_epi32(rnd_i8_5,  8);
        const __m256i rnd_i8_7 = _mm256_slli_epi32(rnd_i8_5, 16);
        const __m256i rnd_i8_8 = _mm256_slli_epi32(rnd_i8_5, 24);

        const __m256  rnd_f8_5 = _mm256_cvtepi32_ps(rnd_i8_5);
        const __m256  rnd_f8_6 = _mm256_cvtepi32_ps(rnd_i8_6);
        const __m256  rnd_f8_7 = _mm256_cvtepi32_ps(rnd_i8_7);
        const __m256  rnd_f8_8 = _mm256_cvtepi32_ps(rnd_i8_8);

        const __m256  rnd_5 = _mm256_mul_ps (rnd_f8_5, clover_mm256_rcp_2pow31_ps);
        const __m256  rnd_6 = _mm256_mul_ps (rnd_f8_6, clover_mm256_rcp_2pow31_ps);
        const __m256  rnd_7 = _mm256_mul_ps (rnd_f8_7, clover_mm256_rcp_2pow31_ps);
        const __m256  rnd_8 = _mm256_mul_ps (rnd_f8_8, clover_mm256_rcp_2pow31_ps);

#endif

        //
        // Calculate the projected values
        //
        const __m256 project_1 = _mm256_fmadd_ps(u_abs_1, scale, rnd_1);
        const __m256 project_2 = _mm256_fmadd_ps(u_abs_2, scale, rnd_2);
        const __m256 project_3 = _mm256_fmadd_ps(u_abs_3, scale, rnd_3);
        const __m256 project_4 = _mm256_fmadd_ps(u_abs_4, scale, rnd_4);
        const __m256 project_5 = _mm256_fmadd_ps(u_abs_5, scale, rnd_5);
        const __m256 project_6 = _mm256_fmadd_ps(u_abs_6, scale, rnd_6);
        const __m256 project_7 = _mm256_fmadd_ps(u_abs_7, scale, rnd_7);
        const __m256 project_8 = _mm256_fmadd_ps(u_abs_8, scale, rnd_8);

        //
        // Truncate
        //
        const __m256i q_abs_1 = _mm256_cvttps_epi32(project_1);
        const __m256i q_abs_2 = _mm256_cvttps_epi32(project_2);
        const __m256i q_abs_3 = _mm256_cvttps_epi32(project_3);
        const __m256i q_abs_4 = _mm256_cvttps_epi32(project_4);
        const __m256i q_abs_5 = _mm256_cvttps_epi32(project_5);
        const __m256i q_abs_6 = _mm256_cvttps_epi32(project_6);
        const __m256i q_abs_7 = _mm256_cvttps_epi32(project_7);
        const __m256i q_abs_8 = _mm256_cvttps_epi32(project_8);

        //
        // Reassemble the signs
        //
        __m256i q_1 = _mm256_sign_epi32(q_abs_1, (__m256i) u_1);
        __m256i q_2 = _mm256_sign_epi32(q_abs_2, (__m256i) u_2);
        __m256i q_3 = _mm256_sign_epi32(q_abs_3, (__m256i) u_3);
        __m256i q_4 = _mm256_sign_epi32(q_abs_4, (__m256i) u_4);
        __m256i q_5 = _mm256_sign_epi32(q_abs_5, (__m256i) u_5);
        __m256i q_6 = _mm256_sign_epi32(q_abs_6, (__m256i) u_6);
        __m256i q_7 = _mm256_sign_epi32(q_abs_7, (__m256i) u_7);
        __m256i q_8 = _mm256_sign_epi32(q_abs_8, (__m256i) u_8);

        //
        // Transpose the 8x8 registers (this might actually run faster if done right)
        //
        _mm256_transpose8_epi32(&q_1, &q_2, &q_3, &q_4, &q_5, &q_6, &q_7, &q_8);

        q_1 = _mm256_slli_epi32(q_1, 28);
        q_2 = _mm256_slli_epi32(q_2, 28);
        q_3 = _mm256_slli_epi32(q_3, 28);
        q_4 = _mm256_slli_epi32(q_4, 28);
        q_5 = _mm256_slli_epi32(q_5, 28);
        q_6 = _mm256_slli_epi32(q_6, 28);
        q_7 = _mm256_slli_epi32(q_7, 28);
        q_8 = _mm256_slli_epi32(q_8, 28);

        q_1 = _mm256_srli_epi32(q_1, 6 * 4);
        q_2 = _mm256_srli_epi32(q_2, 7 * 4);
        q_3 = _mm256_srli_epi32(q_3, 4 * 4);
        q_4 = _mm256_srli_epi32(q_4, 5 * 4);
        q_5 = _mm256_srli_epi32(q_5, 2 * 4);
        q_6 = _mm256_srli_epi32(q_6, 3 * 4);
        q_7 = _mm256_srli_epi32(q_7, 0 * 4);
        q_8 = _mm256_srli_epi32(q_8, 1 * 4);

        const __m256i t1 = _mm256_or_si256(q_1, q_2);
        const __m256i t2 = _mm256_or_si256(q_3, q_4);
        const __m256i t3 = _mm256_or_si256(q_5, q_6);
        const __m256i t4 = _mm256_or_si256(q_7, q_8);
        const __m256i t5 = _mm256_or_si256(t1, t2);
        const __m256i t6 = _mm256_or_si256(t3, t4);
        const __m256i t7 = _mm256_or_si256(t5, t6);

        _mm256_storeu_si256((__m256i *)(dstq + (offset >> 1)), t7);
    }

    //printf("%d %d %d %d %d %d %d %d\n", dstq[0], dstq[1], dstq[2], dstq[3], dstq[4], dstq[5], dstq[6], dstq[7]);
}

void quantize_4(const float * restrict src, char * restrict dst, int n, int k) {
    for (int i = 0; i < n; ++i) {
        quantize_4_row(src + i*k, dst, k);
        dst += quantize_4_row_size(k);
    }
}

void vec_dot_4q_2(const int n, float * restrict s, const void * restrict x, const void * restrict y) {
    const int nb = quantize_4_blocks_per_row(n);

    float * su = (float *) x;
    float * sv = (float *) y;

    int8_t * u = (int8_t *) (su + nb + nb%2);
    int8_t * v = (int8_t *) (sv + nb + nb%2);

    __m256 dot_product_acc_1 = _mm256_setzero_ps();
    __m256 dot_product_acc_2 = _mm256_setzero_ps();

    for (uint64_t b = 0; b < nb; b += 2) {
        const uint64_t offset_1 = b * 32;
        const uint64_t b1       = b + 1;
        const uint64_t b2       = b + 2; // ???????????????
        const uint64_t offset_2 = offset_1 + 32;
        const uint64_t offset_3 = offset_1 + 64;

        const __m256i qu_1 = _mm256_loadu_si256( (__m256i *) (u + offset_1) );
        const __m256i qu_2 = _mm256_loadu_si256( (__m256i *) (u + offset_2) );
        const __m256i qv_1 = _mm256_loadu_si256( (__m256i *) (v + offset_1) );
        const __m256i qv_2 = _mm256_loadu_si256( (__m256i *) (v + offset_2) );

        const __m256 su_1 = _mm256_broadcast_ss(su + b);
        const __m256 su_2 = _mm256_broadcast_ss(su + b1);
        const __m256 sv_1 = _mm256_broadcast_ss(sv + b);
        const __m256 sv_2 = _mm256_broadcast_ss(sv + b1);

        const __m256 su_scaled_1  = _mm256_mul_ps(su_1, clover_mm256_rcp_49_ps);
        const __m256 su_scaled_2  = _mm256_mul_ps(su_2, clover_mm256_rcp_49_ps);
        const __m256 scaled_rcp_1 = _mm256_mul_ps(su_scaled_1, sv_1);
        const __m256 scaled_rcp_2 = _mm256_mul_ps(su_scaled_2, sv_2);

        _mm_prefetch((char *)(u + offset_3), _MM_HINT_T0);
        _mm_prefetch((char *)(v + offset_3), _MM_HINT_T0);
        _mm_prefetch((char *)(su + b2), _MM_HINT_T0);
        _mm_prefetch((char *)(sv + b2), _MM_HINT_T0);

        const __m256i qu_lo_shift_1 = _mm256_slli_epi16(qu_1, 4);
        const __m256i qv_lo_shift_1 = _mm256_slli_epi16(qv_1, 4);
        const __m256i qu_lo_shift_2 = _mm256_slli_epi16(qu_2, 4);
        const __m256i qv_lo_shift_2 = _mm256_slli_epi16(qv_2, 4);

        const __m256i qu_hi_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_1);
        const __m256i qv_hi_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_1);
        const __m256i qu_lo_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_1);
        const __m256i qv_lo_1 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_lo_shift_1);
        const __m256i qu_hi_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_2);
        const __m256i qv_hi_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_2);
        const __m256i qu_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qu_lo_shift_2);
        const __m256i qv_lo_2 = _mm256_and_si256(clover_mm256_1st_bit_set_epi8, qv_lo_shift_2);
        //
        // Get absolute values of u vectors
        //
        const __m256i au_hi_1 = _mm256_sign_epi8(qu_hi_1, qu_hi_1);
        const __m256i au_lo_1 = _mm256_sign_epi8(qu_lo_1, qu_lo_1);
        const __m256i au_hi_2 = _mm256_sign_epi8(qu_hi_2, qu_hi_2);
        const __m256i au_lo_2 = _mm256_sign_epi8(qu_lo_2, qu_lo_2);
        //
        // Sign the values of the v vectors
        //
        const __m256i sv_hi_1 = _mm256_sign_epi8(qv_hi_1, qu_hi_1);
        const __m256i sv_lo_1 = _mm256_sign_epi8(qv_lo_1, qu_lo_1);
        const __m256i sv_hi_2 = _mm256_sign_epi8(qv_hi_2, qu_hi_2);
        const __m256i sv_lo_2 = _mm256_sign_epi8(qv_lo_2, qu_lo_2);
        //
        // Perform multiplication and create 16-bit values
        //
        const __m256i dot_hi_1 = _mm256_maddubs_epi16 (au_hi_1, sv_hi_1);
        const __m256i dot_lo_1 = _mm256_maddubs_epi16 (au_lo_1, sv_lo_1);
        const __m256i dot_hi_2 = _mm256_maddubs_epi16 (au_hi_2, sv_hi_2);
        const __m256i dot_lo_2 = _mm256_maddubs_epi16 (au_lo_2, sv_lo_2);

        const __m256i dot_hi_shift_1 = _mm256_srai_epi16 (dot_hi_1, 8);
        const __m256i dot_lo_shift_1 = _mm256_srai_epi16 (dot_lo_1, 8);
        const __m256i dot_hi_shift_2 = _mm256_srai_epi16 (dot_hi_2, 8);
        const __m256i dot_lo_shift_2 = _mm256_srai_epi16 (dot_lo_2, 8);

        const __m256i dot_16_1 = _mm256_add_epi16(dot_hi_shift_1, dot_lo_shift_1);
        const __m256i dot_16_2 = _mm256_add_epi16(dot_hi_shift_2, dot_lo_shift_2);

        const __m256i dot_32_1 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_1);
        const __m256i dot_32_2 = _mm256_madd_epi16(clover_mm256_1_epi16, dot_16_2);

        const __m256  dot_f_1  = _mm256_cvtepi32_ps(dot_32_1);
        const __m256  dot_f_2  = _mm256_cvtepi32_ps(dot_32_2);

        //
        // Perform dot product on the block
        //
        dot_product_acc_1 = _mm256_fmadd_ps(scaled_rcp_1, dot_f_1, dot_product_acc_1);
        dot_product_acc_2 = _mm256_fmadd_ps(scaled_rcp_2, dot_f_2, dot_product_acc_2);
    }

    const __m256 vacc = _mm256_add_ps(dot_product_acc_1, dot_product_acc_2);
    *s = _mm256_haddf32_ps(vacc);
}

void mul_mat_4q_2(
    const void * src0,
    const void * src1, // transposed
         float * dst,
    int m, int n, int k) {
    assert(k % QK == 0);

    const int nb = quantize_4_blocks_per_row(k);

    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_4q_2(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_4_row_size(k);
        }

        src0 = (const char *) src0 +   quantize_4_row_size(k);
        src1 = (const char *) src1 - n*quantize_4_row_size(k);

        dst = (float *) dst + n;
    }
}

int main(int argc, const char ** argv) {
    // AVX constants init

    clover_mm256_1st_bit_off_epi8 = _mm256_set1_epi32 (0x7F7F7F7FU);
    clover_mm256_1st_bit_set_epi8 = _mm256_set1_epi8  (-16);
    clover_mm256_1st_bit_set_ps   = (__m256) _mm256_set1_epi32 (clover_1st_bit_set_32);
    clover_mm256_1st_bit_off_ps   = (__m256) _mm256_set1_epi32 (clover_1st_bit_off_32);

    clover_mm256_mask_1st_epi32    = _mm256_setr_epi32(0xFFFFFFFFU, 0, 0, 0, 0, 0, 0, 0);

    clover_mm256_1_epi16           = _mm256_set1_epi16(1);
    clover_mm256_1_ps              = _mm256_set1_ps(1.0f);
    clover_mm256_7_ps              = _mm256_set1_ps(7.0f);
    clover_mm256_127_ps            = _mm256_set1_ps(127.0f);
    clover_mm256_rcp_7_ps          = _mm256_set1_ps(1.0f / 7.0f);
    clover_mm256_rcp_127_ps        = _mm256_set1_ps(1.0f / 127.0f);
    clover_mm256_rcp_49_ps         = _mm256_set1_ps(1.0f / 49.0f);
    clover_mm256_rcp_2pow31_ps     = _mm256_set1_ps(1.0f / 2147483648.0f);

    clover_mm256_8bit_perm_lo = _mm256_setr_epi8 (
            0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15,
            0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15
            );
    clover_mm256_8bit_perm_hi = _mm256_setr_epi8 (
            2, 6, 10, 14, 0, 4, 8, 12, 3, 7, 11, 15, 1, 5, 9, 13,
            2, 6, 10, 14, 0, 4, 8, 12, 3, 7, 11, 15, 1, 5, 9, 13
            );

    clover_mm256_8bit_restore_perm_lo = _mm256_setr_epi8(
            0, 8, -128, -128, 1, 9, -128, -128, 2, 10, -128, -128, 3, 11, -128, -128,
            -128, -128, 4, 12, -128, -128, 5, 13, -128, -128, 6, 14, -128, -128, 7, 15
            );
    clover_mm256_8bit_restore_perm_hi = _mm256_setr_epi8 (
            -128, -128, 0, 8, -128, -128, 1, 9, -128, -128, 2, 10, -128, -128, 3, 11,
            4, 12, -128, -128, 5, 13, -128, -128, 6, 14, -128, -128, 7, 15, -128, -128
            );

    ///////////////////////////////

    assert(sizeof(gq_quant_t)*8 == gq_t_bits);

    float * src0 = (float *)malloc(sizeof(float)*M*K);
    float * src1 = (float *)malloc(sizeof(float)*N*K);
    float * dst  = (float *)malloc(sizeof(float)*M*N);

    for (int i = 0; i < M*K; i++) {
        /*src0[i] = rand() / (float)RAND_MAX;*/
        /*src0[i] = i%100;*/
        src0[i] = 1;
    }

    for (int i = 0; i < N*K; i++) {
        //src1[i] = rand() / (float)RAND_MAX;
        /*src1[i] = i%100;*/
        src1[i] = i%4;
    }

    void * src0_gq = calloc(1, quantize_2_row_size(K)*M);
    void * src1_gq = calloc(1, quantize_2_row_size(K)*N);

    void * src0_4q = calloc(1, quantize_3_row_size(K)*M);
    void * src1_4q = calloc(1, quantize_3_row_size(K)*N);

    const size_t sizef16 = sizeof(ggml_fp16_t)*M*K + sizeof(ggml_fp16_t)*N*K;
    const size_t sizegq  = quantize_2_row_size(K)*M + quantize_2_row_size(K)*N;
    const size_t size4q  = quantize_3_row_size(K)*M + quantize_3_row_size(K)*N;

    printf("compression: %f\n", (float)sizegq/sizef16);
    printf("compression: %f\n", (float)size4q/sizef16);

    int method = 0;
    if (argc > 1) {
        method = atoi(argv[1]);
    }

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
            quantize_3(src0, src0_4q, M, K);
            quantize_3(src1, src1_4q, N, K);
        }

        if (method == 4) {
            quantize_4(src0, src0_4q, M, K);
            quantize_4(src1, src1_4q, N, K);
        }

        const uint64_t t_end = get_time_us();
        printf("convert time: %f ms / method = %d\n", (t_end - t_start) / 1000.0, method);
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
            mul_mat_4q(src0_4q, src1_4q, dst, M, N, K);
        }

        if (method == 4) {
            mul_mat_4q_2(src0_4q, src1_4q, dst, M, N, K);
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

    free(src0_gq);
    free(src1_gq);

    return 0;
}
