#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>

#include <immintrin.h>

const int N = 1 << 14;
const int M = 768;

//
// naive implementation
//

void mul_mat_vec_f32_0(
    const float * restrict src0,
    const float * restrict src1,
    float * dst,
    int nrows,
    int ncols) {
    for (int i = 0; i < nrows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < ncols; j++) {
            sum += src0[i*ncols + j]*src1[j];
        }
        dst[i] = sum;
    }
}

//
// SIMD with 8 32-bit floats
//

float reduce_vector8_0(__m256 v) {
    __m128 v1 = _mm256_extractf128_ps(v, 0);
    __m128 v2 = _mm256_extractf128_ps(v, 1);
    __m128 v3 = _mm_add_ps(v1, v2);
    __m128 v4 = _mm_shuffle_ps(v3, v3, 0x4e);
    __m128 v5 = _mm_add_ps(v3, v4);
    __m128 v6 = _mm_shuffle_ps(v5, v5, 0x11);
    __m128 v7 = _mm_add_ps(v5, v6);
    return _mm_cvtss_f32(v7);
}

// vectorized implementation using AVX
void mul_mat_vec_f32_1(
    const float * restrict src0,
    const float * restrict src1,
    float * dst,
    int nrows,
    int ncols) {

    const int ncols8 = ncols & ~7;

    for (int i = 0; i < nrows; i++) {
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < ncols8; j += 8) {
            __m256 a = _mm256_loadu_ps(src0 + i*ncols + j);
            __m256 b = _mm256_loadu_ps(src1 + j);
            __m256 c = _mm256_mul_ps(a, b);
            sum = _mm256_add_ps(sum, c);
        }
        dst[i] = reduce_vector8_0(sum);

        for (int j = ncols8; j < ncols; j++) {
            dst[i] += src0[i*ncols + j]*src1[j];
        }
    }
}

void mul_mat_vec_f32_2(
    const float * restrict src0,
    const float * restrict src1,
    float * dst,
    int nrows,
    int ncols) {

    const int ncols32 = ncols & ~31;

    for (int i = 0; i < nrows; i++) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        const float * restrict src0_row = src0 + i*ncols;
        for (int j = 0; j < ncols32; j += 32) {
            __m256 a0 = _mm256_loadu_ps(src0_row + j + 0);
            __m256 a1 = _mm256_loadu_ps(src0_row + j + 8);
            __m256 a2 = _mm256_loadu_ps(src0_row + j + 16);
            __m256 a3 = _mm256_loadu_ps(src0_row + j + 24);
            __m256 b0 = _mm256_loadu_ps(src1 + j + 0);
            __m256 b1 = _mm256_loadu_ps(src1 + j + 8);
            __m256 b2 = _mm256_loadu_ps(src1 + j + 16);
            __m256 b3 = _mm256_loadu_ps(src1 + j + 24);
#if defined(__FMA__)
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
#else
            sum0 = _mm256_add_ps(_mm256_mul_ps(a0, b0), sum0);
            sum1 = _mm256_add_ps(_mm256_mul_ps(a1, b1), sum1);
            sum2 = _mm256_add_ps(_mm256_mul_ps(a2, b2), sum2);
            sum3 = _mm256_add_ps(_mm256_mul_ps(a3, b3), sum3);
#endif
        }
        dst[i] = reduce_vector8_0(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));

        for (int j = ncols32; j < ncols; j++) {
            dst[i] += src0[i*ncols + j]*src1[j];
        }
    }
}

//
// SIMD with 8 16-bit floats
//

static inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
    return as_float(w);
#elif defined(__CUDA_ARCH__)
    return __uint_as_float((unsigned int) w);
#elif defined(__INTEL_COMPILER)
    return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    return _CopyFloatFromInt32((__int32) w);
#else
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = { w };
    return fp32.as_value;
#endif
}

static inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
	return as_uint(f);
#elif defined(__CUDA_ARCH__)
	return (uint32_t) __float_as_uint(f);
#elif defined(__INTEL_COMPILER)
	return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return (uint32_t) _CopyInt32FromFloat(f);
#else
	union {
		float as_value;
		uint32_t as_bits;
	} fp32 = { f };
	return fp32.as_bits;
#endif
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit representation, to
 * a 32-bit floating-point number in IEEE single-precision format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding mode and no operations on denormals)
 * floating-point operations and bitcasts between integer and floating-point variables.
 */
static inline float fp16_ieee_to_fp32_value(uint16_t h) {
    /*
     * Extend the half-precision floating-point number to 32 bits and shift to the upper part of the 32-bit word:
     *      +---+-----+------------+-------------------+
     *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
     *      +---+-----+------------+-------------------+
     * Bits  31  26-30    16-25            0-15
     *
     * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0 - zero bits.
     */
    const uint32_t w = (uint32_t) h << 16;
    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = w & UINT32_C(0x80000000);
    /*
     * Extract mantissa and biased exponent of the input number into the high bits of the 32-bit word:
     *
     *      +-----+------------+---------------------+
     *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
     *      +-----+------------+---------------------+
     * Bits  27-31    17-26            0-16
     */
    const uint32_t two_w = w + w;

    /*
     * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become mantissa and exponent
     * of a single-precision floating-point number:
     *
     *       S|Exponent |          Mantissa
     *      +-+---+-----+------------+----------------+
     *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
     *      +-+---+-----+------------+----------------+
     * Bits   | 23-31   |           0-22
     *
     * Next, there are some adjustments to the exponent:
     * - The exponent needs to be corrected by the difference in exponent bias between single-precision and half-precision
     *   formats (0x7F - 0xF = 0x70)
     * - Inf and NaN values in the inputs should become Inf and NaN values after conversion to the single-precision number.
     *   Therefore, if the biased exponent of the half-precision input was 0x1F (max possible value), the biased exponent
     *   of the single-precision output must be 0xFF (max possible value). We do this correction in two steps:
     *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset below) rather than by 0x70 suggested
     *     by the difference in the exponent bias (see above).
     *   - Then we multiply the single-precision result of exponent adjustment by 2**(-112) to reverse the effect of
     *     exponent adjustment by 0xE0 less the necessary exponent adjustment by 0x70 due to difference in exponent bias.
     *     The floating-point multiplication hardware would ensure than Inf and NaN would retain their value on at least
     *     partially IEEE754-compliant implementations.
     *
     * Note that the above operations do not handle denormal inputs (where biased exponent == 0). However, they also do not
     * operate on denormal inputs, and do not produce denormal results.
     */
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    /*
     * Convert denormalized half-precision inputs into single-precision results (always normalized).
     * Zero inputs are also handled here.
     *
     * In a denormalized number the biased exponent is zero, and mantissa has on-zero bits.
     * First, we shift mantissa into bits 0-9 of the 32-bit word.
     *
     *                  zeros           |  mantissa
     *      +---------------------------+------------+
     *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
     *      +---------------------------+------------+
     * Bits             10-31                0-9
     *
     * Now, remember that denormalized half-precision numbers are represented as:
     *    FP16 = mantissa * 2**(-24).
     * The trick is to construct a normalized single-precision number with the same mantissa and thehalf-precision input
     * and with an exponent which would scale the corresponding mantissa bits to 2**(-24).
     * A normalized single-precision floating-point number is represented as:
     *    FP32 = (1 + mantissa * 2**(-23)) * 2**(exponent - 127)
     * Therefore, when the biased exponent is 126, a unit change in the mantissa of the input denormalized half-precision
     * number causes a change of the constructud single-precision number by 2**(-24), i.e. the same ammount.
     *
     * The last step is to adjust the bias of the constructed single-precision number. When the input half-precision number
     * is zero, the constructed single-precision number has the value of
     *    FP32 = 1 * 2**(126 - 127) = 2**(-1) = 0.5
     * Therefore, we need to subtract 0.5 from the constructed single-precision number to get the numerical equivalent of
     * the input half-precision number.
     */
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    /*
     * - Choose either results of conversion of input as a normalized number, or as a denormalized number, depending on the
     *   input exponent. The variable two_w contains input exponent in bits 27-31, therefore if its smaller than 2**27, the
     *   input is either a denormal number, or zero.
     * - Combine the result of conversion of exponent and mantissa with the sign of the input number.
     */
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a 16-bit floating-point number in
 * IEEE half-precision format, in bit representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding mode and no operations on denormals)
 * floating-point operations and bitcasts between integer and floating-point variables.
 */
static inline uint16_t fp16_ieee_from_fp32_value(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

void mul_mat_vec_f16_0(
    const uint16_t * src0,
    const uint16_t * src1,
             float * dst,
    int nrows,
    int ncols) {

    const int ncols8 = ncols & ~7;

    for (int i = 0; i < nrows; i++) {
        __m256 sum = _mm256_setzero_ps();

        const uint16_t * src0_row = src0 + i * ncols;
        for (int j = 0; j < ncols8; j += 8) {
            __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j)));
            __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j)));
#if defined(__FMA__)
            sum = _mm256_fmadd_ps(a, b, sum);
#else
            sum = _mm256_add_ps(_mm256_mul_ps(a, b), sum);
#endif
        }
        dst[i] = reduce_vector8_0(sum);

        for (int j = ncols8; j < ncols; j++) {
            dst[i] += fp16_ieee_to_fp32_value(src0_row[j]) * fp16_ieee_to_fp32_value(src1[j]);
        }
    }
}

void mul_mat_vec_f16_1(
    const uint16_t * src0,
    const uint16_t * src1,
             float * dst,
    int nrows,
    int ncols) {

    const int ncols16 = ncols & ~15;

    for (int i = 0; i < nrows; i++) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        const uint16_t * src0_row = src0 + i * ncols;
        for (int j = 0; j < ncols16; j += 16) {
            __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 0)));
            __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 8)));
            __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j)));
            __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j + 8)));
#if defined(__FMA__)
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
#else
            sum0 = _mm256_add_ps(_mm256_mul_ps(a0, b0), sum0);
            sum1 = _mm256_add_ps(_mm256_mul_ps(a1, b1), sum1);
#endif
        }
        dst[i] = reduce_vector8_0(sum0) + reduce_vector8_0(sum1);

        for (int j = ncols16; j < ncols; j++) {
            dst[i] += fp16_ieee_to_fp32_value(src0_row[j]) * fp16_ieee_to_fp32_value(src1[j]);
        }
    }
}

void mul_mat_vec_f16_2(
    const uint16_t * src0,
    const uint16_t * src1,
             float * dst,
    int nrows,
    int ncols) {

    const int ncols32 = ncols & ~31;

    for (int i = 0; i < nrows; i++) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        const uint16_t * src0_row = src0 + i * ncols;
        for (int j = 0; j < ncols32; j += 32) {
            __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 0)));
            __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 8)));
            __m256 a2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 16)));
            __m256 a3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 24)));
            __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j)));
            __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j + 8)));
            __m256 b2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j + 16)));
            __m256 b3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src1 + j + 24)));
#if defined(__FMA__)
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
#else
            sum0 = _mm256_add_ps(_mm256_mul_ps(a0, b0), sum0);
            sum1 = _mm256_add_ps(_mm256_mul_ps(a1, b1), sum1);
            sum2 = _mm256_add_ps(_mm256_mul_ps(a2, b2), sum2);
            sum3 = _mm256_add_ps(_mm256_mul_ps(a3, b3), sum3);
#endif
        }
        dst[i] = reduce_vector8_0(sum0) + reduce_vector8_0(sum1) + reduce_vector8_0(sum2) + reduce_vector8_0(sum3);

        for (int j = ncols32; j < ncols; j++) {
            dst[i] += fp16_ieee_to_fp32_value(src0_row[j]) * fp16_ieee_to_fp32_value(src1[j]);
        }
    }
}

void mul_mat_vec_f16_3(
    const uint16_t * src0,
    const    float * src1,
             float * dst,
    int nrows,
    int ncols) {

    const int ncols32 = ncols & ~31;

    for (int i = 0; i < nrows; i++) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        const uint16_t * src0_row = src0 + i * ncols;
        for (int j = 0; j < ncols32; j += 32) {
            __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 0)));
            __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 8)));
            __m256 a2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 16)));
            __m256 a3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src0_row + j + 24)));
            __m256 b0 = _mm256_loadu_ps(src1 + j);
            __m256 b1 = _mm256_loadu_ps(src1 + j + 8);
            __m256 b2 = _mm256_loadu_ps(src1 + j + 16);
            __m256 b3 = _mm256_loadu_ps(src1 + j + 24);
#if defined(__FMA__)
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);
#else
            sum0 = _mm256_add_ps(_mm256_mul_ps(a0, b0), sum0);
            sum1 = _mm256_add_ps(_mm256_mul_ps(a1, b1), sum1);
            sum2 = _mm256_add_ps(_mm256_mul_ps(a2, b2), sum2);
            sum3 = _mm256_add_ps(_mm256_mul_ps(a3, b3), sum3);
#endif
        }
        dst[i] = reduce_vector8_0(sum0) + reduce_vector8_0(sum1) + reduce_vector8_0(sum2) + reduce_vector8_0(sum3);

        for (int j = ncols32; j < ncols; j++) {
            dst[i] += fp16_ieee_to_fp32_value(src0_row[j]) * fp16_ieee_to_fp32_value(src1[j]);
        }
    }
}

uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, const char ** argv) {
    float * src0 = malloc(sizeof(float)*N*M);
    float * src1 = malloc(sizeof(float)*M);
    float * dst  = malloc(sizeof(float)*N);

    //float * src0 = (float *)(aligned_alloc(64, sizeof(float)*N*M));
    //float * src1 = (float *)(aligned_alloc(64, sizeof(float)*M));
    //float * dst  = (float *)(aligned_alloc(64, sizeof(float)*N));

    for (int i = 0; i < N*M; i++) {
        src0[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < M; i++) {
        src1[i] = rand() / (float)RAND_MAX;
    }

    // convert src0 and src1 to __fp16
    uint16_t * src0_fp16 = (uint16_t *)(malloc(sizeof(uint16_t)*N*M));
    uint16_t * src1_fp16 = (uint16_t *)(malloc(sizeof(uint16_t)*M));
    //uint16_t * src0_fp16 = (uint16_t *)(aligned_alloc(64, sizeof(uint16_t)*N*M));
    //uint16_t * src1_fp16 = (uint16_t *)(aligned_alloc(64, sizeof(uint16_t)*M));

    {
        const uint64_t t_start = get_time_us();

        for (int i = 0; i < N*M; i++) {
            src0_fp16[i] = fp16_ieee_from_fp32_value(src0[i]);
            //printf("%f %f\n", src0[i], fp16_ieee_to_fp32_value(src0_fp16[i]));
            //assert(!isnan(fp16_ieee_to_fp32_value(src0_fp16[i])));
        }

        for (int i = 0; i < M; i++) {
            src1_fp16[i] = fp16_ieee_from_fp32_value(src1[i]);
        }

        const uint64_t t_end = get_time_us();
        printf("convert time: %f ms\n", (t_end - t_start) / 1000.0);
    }

    for (int i = 0; i < 16; ++i) {
        printf("%f %f\n", src0[i], fp16_ieee_to_fp32_value(src0_fp16[i]));
    }

    int method = 0;
    if (argc > 1) {
        method = atoi(argv[1]);
    }

    const int nIter = 1000;

    const clock_t start = clock();
    const uint64_t start_us = get_time_us();

    double iM = 1.0/M;
    double sum = 0.0f;
    for (int i = 0; i < nIter; i++) {
        if (method == 0) {
            mul_mat_vec_f32_0(src0, src1, dst, N, M);
        }

        if (method == 1) {
            mul_mat_vec_f32_1(src0, src1, dst, N, M);
        }

        if (method == 2) {
            mul_mat_vec_f32_2(src0, src1, dst, N, M);
        }

        if (method == 3) {
            mul_mat_vec_f16_0(src0_fp16, src1_fp16, dst, N, M);
        }

        if (method == 4) {
            mul_mat_vec_f16_1(src0_fp16, src1_fp16, dst, N, M);
        }

        if (method == 5) {
            mul_mat_vec_f16_2(src0_fp16, src1_fp16, dst, N, M);
        }

        if (method == 6) {
            mul_mat_vec_f16_3(src0_fp16, src1, dst, N, M);
        }
    }

    for (int i = 0; i < N; i++) {
        sum += dst[i]*iM;
    }

    {
        const clock_t end = clock();
        const uint64_t end_us = get_time_us();
        printf("%s: elapsed ticks: %ld\n", __func__, end - start);
        printf("%s: elapsed us: %ld\n", __func__, end_us - start_us);
    }

    printf("%f\n", sum);

    free(src0);
    free(src1);
    free(dst);

    free(src0_fp16);
    free(src1_fp16);

    return 0;
}
