#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>

#include <arm_neon.h>

#include <Accelerate/Accelerate.h>

const int M = 1280;
const int N = 1536;
const int K = 1280;

uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

//
// naive implementation
//

void mul_mat_f32_0(
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

void mul_mat_f16_0(
    const __fp16 * src0,
    const __fp16 * src1,
           float * dst,
    int m, int n, int k) {
    const int k32 = k & ~31;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sumf = 0.0;

            float16x8_t sum0 = vdupq_n_f16(0.0f);
            float16x8_t sum1 = vdupq_n_f16(0.0f);
            float16x8_t sum2 = vdupq_n_f16(0.0f);
            float16x8_t sum3 = vdupq_n_f16(0.0f);

            float16x8_t x0, x1, x2, x3;
            float16x8_t y0, y1, y2, y3;

            const __fp16 * restrict p0 = src0 + i*k;
            const __fp16 * restrict p1 = src1 + j*k;

            for (int l = 0; l < k32; l += 32) {
                x0 = vld1q_f16(p0 + l + 0 );
                x1 = vld1q_f16(p0 + l + 8 );
                x2 = vld1q_f16(p0 + l + 16);
                x3 = vld1q_f16(p0 + l + 24);

                y0 = vld1q_f16(p1 + l + 0 );
                y1 = vld1q_f16(p1 + l + 8 );
                y2 = vld1q_f16(p1 + l + 16);
                y3 = vld1q_f16(p1 + l + 24);

                sum0 = vfmaq_f16(sum0, x0, y0);
                sum1 = vfmaq_f16(sum1, x1, y1);
                sum2 = vfmaq_f16(sum2, x2, y2);
                sum3 = vfmaq_f16(sum3, x3, y3);
            }

            // reduce sum0..sum3 to sum0
            sum0 = vaddq_f16(sum0, sum1);
            sum2 = vaddq_f16(sum2, sum3);
            sum0 = vaddq_f16(sum0, sum2);

            // load sum0 into 2 float32x4_t
            float32x4_t sum0f32 = vcvt_f32_f16(vget_low_f16(sum0));
            float32x4_t sum1f32 = vcvt_f32_f16(vget_high_f16(sum0));

            // reduce sum0f32 and sum1f32 to sumf
            sum0f32 = vaddq_f32(sum0f32, sum1f32);

            float32x2_t sumf32 = vadd_f32(vget_low_f32(sum0f32), vget_high_f32(sum0f32));
            sumf = vget_lane_f32(sumf32, 0) + vget_lane_f32(sumf32, 1);

            //sumf = sum0[0] + sum0[1] + sum0[2] + sum0[3] + sum0[4] + sum0[5] + sum0[6] + sum0[7];

            for (int l = k32; l < k32; l++) {
                sumf += p0[l]*p1[l];
            }

            dst[i*n + j] = sumf;
        }
    }
}

// blocking with block size 32
void mul_mat_f16_1(
    const __fp16 * src0,
    const __fp16 * src1,
           float * dst,
    int m, int n, int k) {

    const int k32 = k & ~31;
    const int bs  = 32;

    memset(dst, 0, m*n*sizeof(float));

    for (int i = 0; i < m; i += bs) {
        for (int j = 0; j < n; j += bs) {
            for (int l = 0; l < k; l += bs) {
                for (int ii = i; ii < i + bs; ii++) {
                    const __fp16 * restrict p0 = src0 + ii*k;

                    float16x8_t x0, x1, x2, x3;

                    x0 = vld1q_f16(p0 + l + 0 );
                    x1 = vld1q_f16(p0 + l + 8 );
                    x2 = vld1q_f16(p0 + l + 16);
                    x3 = vld1q_f16(p0 + l + 24);

                    for (int jj = j; jj < j + bs; jj++) {
                        float sumf = 0.0;

                        float16x8_t sum0 = vdupq_n_f16(0.0f);
                        float16x8_t sum1 = vdupq_n_f16(0.0f);
                        float16x8_t sum2 = vdupq_n_f16(0.0f);
                        float16x8_t sum3 = vdupq_n_f16(0.0f);

                        float16x8_t y0, y1, y2, y3;

                        const __fp16 * restrict p1 = src1 + jj*k;

                        y0 = vld1q_f16(p1 + l + 0 );
                        y1 = vld1q_f16(p1 + l + 8 );
                        y2 = vld1q_f16(p1 + l + 16);
                        y3 = vld1q_f16(p1 + l + 24);

                        sum0 = vfmaq_f16(sum0, x0, y0);
                        sum1 = vfmaq_f16(sum1, x1, y1);
                        sum2 = vfmaq_f16(sum2, x2, y2);
                        sum3 = vfmaq_f16(sum3, x3, y3);

                        // reduce sum0..sum3 to sum0
                        sum0 = vaddq_f16(sum0, sum1);
                        sum2 = vaddq_f16(sum2, sum3);
                        sum0 = vaddq_f16(sum0, sum2);

                        // load sum0 into 2 float32x4_t
                        float32x4_t sum0f32 = vcvt_f32_f16(vget_low_f16(sum0));
                        float32x4_t sum1f32 = vcvt_f32_f16(vget_high_f16(sum0));

                        // reduce sum0f32 and sum1f32 to sumf
                        sum0f32 = vaddq_f32(sum0f32, sum1f32);

                        float32x2_t sumf32 = vadd_f32(vget_low_f32(sum0f32), vget_high_f32(sum0f32));
                        sumf = vget_lane_f32(sumf32, 0) + vget_lane_f32(sumf32, 1);

                        //sumf = sum0[0] + sum0[1] + sum0[2] + sum0[3] + sum0[4] + sum0[5] + sum0[6] + sum0[7];

                        dst[ii*n + jj] += sumf;
                    }
                }
            }
        }
    }

}

void mul_mat_f8_0(
    const uint8_t * src0,
    const uint8_t * src1,
           float * dst,
    int m, int n, int k) {
    const int k32 = k & ~31;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sumf = 0.0;

            const uint8_t * restrict p0 = src0 + i*k;
            const uint8_t * restrict p1 = src1 + j*k;

            for (int l = 0; l < k32; l += 32) {
                uint8x16_t x0 = vld1q_u8(p0 + l + 0 );
                uint8x16_t x1 = vld1q_u8(p0 + l + 16);

                uint8x16_t y0 = vld1q_u8(p1 + l + 0 );
                uint8x16_t y1 = vld1q_u8(p1 + l + 16);

                x0 = vmulq_u8(x0, y0);
                x1 = vmulq_u8(x1, y1);

                sumf += vaddvq_u8(x0) + vaddvq_u8(x1);
            }

            dst[i*n + j] = sumf;
        }
    }
}

int main(int argc, const char ** argv) {
    float * src0 = malloc(sizeof(float)*M*K);
    float * src1 = malloc(sizeof(float)*N*K);
    float * dst  = malloc(sizeof(float)*M*N);

    for (int i = 0; i < M*K; i++) {
        src0[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < N*K; i++) {
        src1[i] = rand() / (float)RAND_MAX;
    }

    // convert src0 and src1 to __fp16
    __fp16 * src0_fp16 = (__fp16 *)(malloc(sizeof(__fp16)*M*K));
    __fp16 * src1_fp16 = (__fp16 *)(malloc(sizeof(__fp16)*N*K));

    uint8_t * src0_fp8 = (uint8_t *)(malloc(sizeof(__fp16)*M*K));
    uint8_t * src1_fp8 = (uint8_t *)(malloc(sizeof(__fp16)*N*K));

    {
        const uint64_t t_start = get_time_us();

        for (int i = 0; i < M*K; i++) {
            src0_fp16[i] = src0[i];
            //printf("%f %f\n", src0[i], src0_fp16[i]);
            //assert(!isnan(src0_fp16[i]));
        }

        for (int i = 0; i < N*K; i++) {
            src1_fp16[i] = src1[i];
        }

        const uint64_t t_end = get_time_us();
        printf("convert time: %f ms\n", (t_end - t_start) / 1000.0);
    }

    for (int i = 0; i < 16; ++i) {
        printf("%f %f\n", src0[i], src0_fp16[i]);
    }

    int method = 0;
    if (argc > 1) {
        method = atoi(argv[1]);
    }

    const int nIter = 1;

    const clock_t start = clock();
    const uint64_t start_us = get_time_us();

    double iM = 1.0/M;
    double sum = 0.0f;
    for (int i = 0; i < nIter; i++) {
        if (method == 0) {
            mul_mat_f32_0(src0, src1, dst, M, N, K);
        }

        if (method == 1) {
            mul_mat_f16_0(src0_fp16, src1_fp16, dst, M, N, K);
        }

        if (method == 2) {
            mul_mat_f16_1(src0_fp16, src1_fp16, dst, M, N, K);
        }

        if (method == 3) {
            mul_mat_f8_0(src0_fp8, src1_fp8, dst, M, N, K);
        }

        if (method == 4) {
            // Use BLAS sgemm from Accelerate framework
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, src0, K, src1, K, 0.0f, dst, N);
        }
    }

    for (int i = 0; i < N; i++) {
        sum += dst[i]*iM;
    }

    {
        const clock_t end = clock();
        const uint64_t end_us = get_time_us();
        printf("%s: elapsed ticks: %ld\n",  __func__, end - start);
        printf("%s: elapsed us:    %llu / %f ms\n",  __func__, end_us - start_us, (end_us - start_us) / 1000.0 / nIter);
    }

    printf("%f\n", sum);

    free(src0);
    free(src1);
    free(dst);

    free(src0_fp16);
    free(src1_fp16);

    return 0;
}
