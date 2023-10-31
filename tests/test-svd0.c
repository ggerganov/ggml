// SVD dimensionality reduction

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>

#ifdef GGML_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

float frand(void) {
    return (float) rand() / (float) RAND_MAX;
}

//int sgesvd_(char *__jobu, char *__jobvt, __CLPK_integer *__m,
//        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
//        __CLPK_real *__s, __CLPK_real *__u, __CLPK_integer *__ldu,
//        __CLPK_real *__vt, __CLPK_integer *__ldvt, __CLPK_real *__work,
//        __CLPK_integer *__lwork,
//        __CLPK_integer *__info)

int main(int argc, const char ** argv) {
    int m = 10;
    int n = 5;

    float * A  = malloc(n * m * sizeof(float));
    float * A0 = malloc(n * m * sizeof(float));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            A[i * m + j] = (float) (10.0f*(i + 1) + 1.0f * frand());
            //A[i * m + j] = (float) (10.0f*(i%2 + 1) + 0.1f * frand());
            //if (i == 2) {
            //    A[i * m + j] += 20*frand();
            //}
            if ((i == 1 || i == 3) && j > m/2) {
                A[i * m + j] = -A[i * m + j];
            }
        }
    }

    // average vector
    //float * M = malloc(m * sizeof(float));

    //{
    //    for (int j = 0; j < m; ++j) {
    //        M[j] = 0.0f;
    //    }
    //    for (int i = 0; i < n; ++i) {
    //        for (int j = 0; j < m; ++j) {
    //            M[j] += A[i * m + j];
    //        }
    //    }
    //    for (int j = 0; j < m; ++j) {
    //        M[j] /= (float) n;
    //    }
    //}

    //// subtract average vector
    //for (int i = 0; i < n; ++i) {
    //    for (int j = 0; j < m; ++j) {
    //        A[i * m + j] -= M[j];
    //    }
    //}

    memcpy(A0, A, n * m * sizeof(float));

    // print A
    printf("A:\n");
    for (int i = 0; i < n; ++i) {
        printf("col %d : ", i);
        for (int j = 0; j < m; ++j) {
            printf("%9.5f ", A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    // SVD
    // A = U * S * V^T

    float * U = malloc(n * m * sizeof(float));
    float * S = malloc(n * sizeof(float));
    float * V = malloc(n * n * sizeof(float));

    int lda = m;
    int ldu = m;
    int ldvt = n;

    float work_size;
    int lwork = -1;
    int info = 0;

    sgesvd_("S", "S", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, &work_size, &lwork, &info);

    lwork = (int) work_size;

    printf("work_size = %f, info = %d, lwork = %d\n", work_size, info, lwork);

    float * work = malloc(lwork * sizeof(float));

    sgesvd_("S", "S", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, work, &lwork, &info);

    // print U
    printf("U:\n");
    for (int i = 0; i < n; ++i) {
        printf("col %d : ", i);
        for (int j = 0; j < m; ++j) {
            printf("%9.5f ", U[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    // normalize S
    {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += S[i];
        }
        sum *= sqrt((double) m);
        for (int i = 0; i < n; ++i) {
            S[i] /= sum;
        }
    }

    // print S
    printf("S:\n");
    for (int i = 0; i < n; ++i) {
        printf("- %d = %9.5f\n", i, S[i]);
    }
    printf("\n");

    // print V
    printf("V:\n");
    for (int i = 0; i < n; ++i) {
        printf("col %d : ", i);
        for (int j = 0; j < n; ++j) {
            printf("%9.5f ", V[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    // print A
    printf("A:\n");
    for (int i = 0; i < n; ++i) {
        printf("col %d : ", i);
        for (int j = 0; j < m; ++j) {
            printf("%9.5f ", A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    // compute singular vectors in U
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            U[i * m + j] *= S[i];
        }
    }

    // normalize U
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += U[i * m + j] * U[i * m + j];
        }
        sum = sqrt(sum);
        for (int j = 0; j < m; ++j) {
            U[i * m + j] /= sum*sqrt((double) m);
        }
    }

    // print U
    printf("U:\n");
    for (int i = 0; i < n; ++i) {
        printf("col %d : ", i);
        for (int j = 0; j < m; ++j) {
            printf("%9.5f ", U[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");


    // project A0 onto U
    float * A1 = malloc(n * n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A1[i * n + j] = 0.0f;
            for (int k = 0; k < m; ++k) {
                A1[i * n + j] += A0[i * m + k] * U[j * m + k];
            }
        }
    }

    // print A1
    printf("A1:\n");
    for (int i = 0; i < n; ++i) {
        printf("col %d : ", i);
        for (int j = 0; j < n; ++j) {
            printf("%9.5f ", A1[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
