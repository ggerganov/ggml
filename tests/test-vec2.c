#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>

#include <arm_neon.h>

const int N = 1 << 12;
const int M = 1 << 12;

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

void mul_mat_vec_f16_0(
    const __fp16 * src0,
    const __fp16 * src1,
           float * dst,
    int nrows,
    int ncols) {

    const int n64 = ncols & ~63;

    for (int r = 0; r < nrows; r++) {
        float sumf = 0.0;

        float16x8_t sum0 = vdupq_n_f16(0.0f);
        float16x8_t sum1 = vdupq_n_f16(0.0f);
        float16x8_t sum2 = vdupq_n_f16(0.0f);
        float16x8_t sum3 = vdupq_n_f16(0.0f);
        float16x8_t sum4 = vdupq_n_f16(0.0f);
        float16x8_t sum5 = vdupq_n_f16(0.0f);
        float16x8_t sum6 = vdupq_n_f16(0.0f);
        float16x8_t sum7 = vdupq_n_f16(0.0f);

        float16x8_t x0, x1, x2, x3, x4, x5, x6, x7;
        float16x8_t y0, y1, y2, y3, y4, y5, y6, y7;

        const __fp16 * restrict p0 = src0 + r*ncols;

        for (int i = 0; i < n64; i += 64) {
            x0 = vld1q_f16(p0 + i + 0 );
            x1 = vld1q_f16(p0 + i + 8 );
            x2 = vld1q_f16(p0 + i + 16);
            x3 = vld1q_f16(p0 + i + 24);
            x4 = vld1q_f16(p0 + i + 32);
            x5 = vld1q_f16(p0 + i + 40);
            x6 = vld1q_f16(p0 + i + 48);
            x7 = vld1q_f16(p0 + i + 56);

            y0 = vld1q_f16(src1 + i + 0 );
            y1 = vld1q_f16(src1 + i + 8 );
            y2 = vld1q_f16(src1 + i + 16);
            y3 = vld1q_f16(src1 + i + 24);
            y4 = vld1q_f16(src1 + i + 32);
            y5 = vld1q_f16(src1 + i + 40);
            y6 = vld1q_f16(src1 + i + 48);
            y7 = vld1q_f16(src1 + i + 56);

            sum0 = vfmaq_f16(sum0, x0, y0);
            sum1 = vfmaq_f16(sum1, x1, y1);
            sum2 = vfmaq_f16(sum2, x2, y2);
            sum3 = vfmaq_f16(sum3, x3, y3);
            sum4 = vfmaq_f16(sum4, x4, y4);
            sum5 = vfmaq_f16(sum5, x5, y5);
            sum6 = vfmaq_f16(sum6, x6, y6);
            sum7 = vfmaq_f16(sum7, x7, y7);
        }

        // TODO: F16 - better way to reduce this ?
        float16x8_t sum = vaddq_f16(sum0, sum1);

        sum = vaddq_f16(sum, sum2);
        sum = vaddq_f16(sum, sum3);
        sum = vaddq_f16(sum, sum4);
        sum = vaddq_f16(sum, sum5);
        sum = vaddq_f16(sum, sum6);
        sum = vaddq_f16(sum, sum7);

        sumf += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

        for (int j = n64; j < n64; j++) {
            sumf += src0[r*ncols + j]*src1[j];
        }

        dst[r] = sumf;
    }
}

void mul_mat_vec_f16_1(
    const __fp16 * src0,
    const __fp16 * src1,
           float * dst,
    int nrows,
    int ncols) {

    const int n32 = ncols & ~31;

    for (int r = 0; r < nrows; r++) {
        float sumf = 0.0;

        float16x8_t sum0 = vdupq_n_f16(0.0f);
        float16x8_t sum1 = vdupq_n_f16(0.0f);
        float16x8_t sum2 = vdupq_n_f16(0.0f);
        float16x8_t sum3 = vdupq_n_f16(0.0f);

        float16x8_t x0, x1, x2, x3;
        float16x8_t y0, y1, y2, y3;

        const __fp16 * restrict p0 = src0 + r*ncols;

        for (int i = 0; i < n32; i += 32) {
            x0 = vld1q_f16(p0 + i + 0 );
            x1 = vld1q_f16(p0 + i + 8 );
            x2 = vld1q_f16(p0 + i + 16);
            x3 = vld1q_f16(p0 + i + 24);

            y0 = vld1q_f16(src1 + i + 0 );
            y1 = vld1q_f16(src1 + i + 8 );
            y2 = vld1q_f16(src1 + i + 16);
            y3 = vld1q_f16(src1 + i + 24);

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

        for (int j = n32; j < n32; j++) {
            sumf += src0[r*ncols + j]*src1[j];
        }

        dst[r] = sumf;
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
    __fp16 * src0_fp16 = (__fp16 *)(malloc(sizeof(__fp16)*N*M));
    __fp16 * src1_fp16 = (__fp16 *)(malloc(sizeof(__fp16)*M));

    {
        const uint64_t t_start = get_time_us();

        for (int i = 0; i < N*M; i++) {
            src0_fp16[i] = src0[i];
            //printf("%f %f\n", src0[i], src0_fp16[i]);
            //assert(!isnan(src0_fp16[i]));
        }

        for (int i = 0; i < M; i++) {
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
            mul_mat_vec_f16_0(src0_fp16, src1_fp16, dst, N, M);
        }

        if (method == 2) {
            mul_mat_vec_f16_1(src0_fp16, src1_fp16, dst, N, M);
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
