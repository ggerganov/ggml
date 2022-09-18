#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

const int N = 1 << 14;
const int M = 1 << 14;

void mul_mat_vec_f32_0(
    const float * src0,
    const float * src1,
    float * dst,
    unsigned nrows,
    unsigned ncols) {
    for (unsigned i = 0; i < nrows; i++) {
        float sum = 0.0f;
        for (unsigned j = 0; j < ncols; j++) {
            sum += src0[i*ncols + j]*src1[j];
        }
        dst[i] = sum;
    }
}

typedef float afloat __attribute__ ((__aligned__(32)));
void mul_mat_vec_f32_1(
    const afloat *restrict src0,
    const afloat *restrict src1,
    afloat *restrict dst,
    unsigned nrows,
    unsigned ncols) {
    for (unsigned i = 0; i < nrows; i++) {
        const afloat * restrict row = src0 + i*ncols;
        const afloat * restrict col = src1;

        float sum = 0.0f;

        for (unsigned j = 0; j < ncols; j++) {
            sum += *row++ * *col++;
        }

        dst[i] = sum;

        //float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        //for (unsigned j = 0; j < ncols; j += 8) {
        //    sum[0] += row[0]*col[0];
        //    sum[1] += row[1]*col[1];
        //    sum[2] += row[2]*col[2];
        //    sum[3] += row[3]*col[3];
        //    sum[4] += row[4]*col[4];
        //    sum[5] += row[5]*col[5];
        //    sum[6] += row[6]*col[6];
        //    sum[7] += row[7]*col[7];

        //    row += 8;
        //    col += 8;
        //}

        //dst[i] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
    }
}

void mul_mat_vec_f32_2(
    const void * src0,
    const void * src1,
    void * dst,
    unsigned nrows,
    unsigned ncols) {
    void * d = dst;
    for (unsigned i = 0; i < nrows; i++) {
        float sum = 0.0f;

        const void * row = src0 + i*ncols*sizeof(float);
        const void * col = src1;
        for (unsigned j = 0; j < ncols; j++) {
            sum += (*(float *)row) * (*(float *)col);
            row += sizeof(float);
            col += sizeof(float);
        }
        *(float *)d = sum;
        d += sizeof(float);
    }
}

int main(int argc, const char ** argv) {
    //float * src0 = (float *)malloc(sizeof(float)*N*M);
    //float * src1 = (float *)malloc(sizeof(float)*M);
    //float * dst  = (float *)malloc(sizeof(float)*N);

    afloat * src0 = (float *)(aligned_alloc(32, sizeof(float)*N*M));
    afloat * src1 = (float *)(aligned_alloc(32, sizeof(float)*M));
    afloat * dst  = (float *)(aligned_alloc(32, sizeof(float)*N));

    for (unsigned i = 0; i < N*M; i++) {
        src0[i] = i;
    }

    for (unsigned i = 0; i < M; i++) {
        src1[i] = i;
    }

    const int nIter = 10;

    const clock_t start = clock();

    double sum = 0.0f;
    for (int i = 0; i < nIter; i++) {
        //mul_mat_vec_f32_0(src0, src1, dst, N, M);
        mul_mat_vec_f32_1(src0, src1, dst, N, M);
        //mul_mat_vec_f32_2(src0, src1, dst, N, M);
        for (unsigned i = 0; i < N; i++) {
            sum += dst[i];
        }
    }

    {
        const clock_t end = clock();
        printf("%s: elapsed ticks: %ld\n", __func__, end - start);
    }

    printf("%f\n", sum);

    return 0;
}
