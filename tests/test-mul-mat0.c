#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows
#include "ggml/ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_NARGS 2

float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

int irand(int n) {
    return rand()%n;
}

void get_random_dims(int64_t * dims, int ndims) {
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = 1 + irand(4);
    }
}

struct ggml_tensor * get_random_tensor(
        struct ggml_context * ctx0,
        int ndims,
        int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return result;
}

float get_element(const struct ggml_tensor * t, int idx) {
    return ((float *)t->data)[idx];
}

void set_element(struct ggml_tensor * t, int idx, float value) {
    ((float *)t->data)[idx] = value;
}

bool check_gradient(
        const char * op_name,
        struct ggml_context * ctx0,
        struct ggml_tensor * x[],
        struct ggml_tensor * f,
        int ndims,
        int nargs,
        float eps,
        float max_error_abs,
        float max_error_rel) {
    const int n_threads = 1;

    struct ggml_cgraph gf = ggml_build_forward (f);
    struct ggml_cgraph gb = ggml_build_backward(ctx0, &gf, false);

    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);
    ggml_graph_reset  (&gf);
    ggml_set_f32      (f->grad, 1.0f);
    ggml_graph_compute_with_ctx(ctx0, &gb, n_threads);

    ggml_graph_dump_dot(&gf, NULL, "test-grad0-forward.dot");
    ggml_graph_dump_dot(&gb, &gf,  "test-grad0-backward.dot");

    for (int i = 0; i < nargs; ++i) {
        const int64_t nelements = ggml_nelements(x[i]);
        for (int64_t k = 0; k < nelements; ++k) {
            // compute gradient using finite differences
            const float x0 = get_element(x[i], k);

            set_element(x[i], k, x0 + eps);
            ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

            const float f0 = ggml_get_f32_1d(f, 0);

            set_element(x[i], k, x0 - eps);
            ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

            const float f1 = ggml_get_f32_1d(f, 0);

            const float g0 = (f0 - f1)/(2.0f*eps);

            set_element(x[i], k, x0);

            // compute gradient using backward graph
            ggml_graph_reset  (&gf);
            ggml_set_f32      (f->grad, 1.0f);
            ggml_graph_compute_with_ctx(ctx0, &gb, n_threads);

            const float g1 = get_element(x[i]->grad, k);

            const float error_abs = fabsf(g0 - g1);
            const float error_rel = g0 != 0 ? fabsf(g0 - g1)/fabs(g0) : 0;

            if (error_abs > max_error_abs || error_rel > max_error_rel) {
                printf("%s: ndims=%d, i=%d, k=%" PRId64 ", g0=%f, g1=%f, error_abs=%f, error_rel=%f\n", op_name, ndims, i, k, g0, g1, error_abs, error_rel);
                assert(false);
            }
        }
    }

    return true;
}


float mat_get(const struct ggml_tensor * t, int i0, int i1, int i2, int i3) {
    const size_t nb0 = t->nb[0];
    const size_t nb1 = t->nb[1];
    const size_t nb2 = t->nb[2];
    const size_t nb3 = t->nb[3];

    return
        *((float*) ((char*)t->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3));
}

bool check_mat_mul(
        const struct ggml_tensor * y,
        const struct ggml_tensor * x0,
        const struct ggml_tensor * x1) {
    const int64_t n00 = x0->ne[0];
    const int64_t n10 = x0->ne[1];
    const int64_t n20 = x0->ne[2];
    const int64_t n30 = x0->ne[3];

    const int64_t n01 = x1->ne[0];
    const int64_t n11 = x1->ne[1];
    const int64_t n21 = x1->ne[2];
    const int64_t n31 = x1->ne[3];

    const int64_t n02 = y->ne[0];
    const int64_t n12 = y->ne[1];
    const int64_t n22 = y->ne[2];
    const int64_t n32 = y->ne[3];

    printf("x0: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n00, n10, n20, n30);
    for (int j = 0; j < n10; ++j) {
        for (int i = 0; i < n00; ++i) {
            printf("%6.3f ", mat_get(x0, i, j, 0, 0));
        }
        printf("\n");
    }
    printf("\n");

    printf("x1: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n01, n11, n21, n31);
    for (int j = 0; j < n11; ++j) {
        for (int i = 0; i < n01; ++i) {
            printf("%6.3f ", mat_get(x1, i, j, 0, 0));
        }
        printf("\n");
    }
    printf("\n");

    printf("y: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n02, n12, n22, n32);
    for (int j = 0; j < n12; ++j) {
        for (int i = 0; i < n02; ++i) {
            printf("%6.3f ", mat_get(y, i, j, 0, 0));
        }
        printf("\n");
    }

    for (int i3 = 0; i3 < n32; ++i3) {
        for (int i2 = 0; i2 < n22; ++i2) {
            for (int i1 = 0; i1 < n12; ++i1) {
                for (int i0 = 0; i0 < n02; ++i0) {
                    float sum = 0.0f;
                    for (int k = 0; k < n00; ++k) {
                        sum += mat_get(x0, k, i0, i2, i3) * mat_get(x1, k, i1, i2, i3);
                    }
                    if (fabsf(sum - mat_get(y, i0, i1, i2, i3)) > 1e-5) {
                        printf("error: i0=%d, i1=%d, i2=%d, i3=%d, sum=%f, y=%f\n",
                                i0, i1, i2, i3, sum, mat_get(y, i0, i1, i2, i3));
                        assert(false);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    int64_t ne[4];

    // original loop: 500
    int niter = 500;
    const char *env = getenv("GGML_NLOOP");
    if (env != NULL) {
        niter = atoi(env);
    }
    if (argc > 1) {
        niter = atoi(argv[1]);
    }

    int n_threads = 1;

    for (int iter = 0; iter < niter; ++iter) {
        printf("test-mul-mat0: iter:%d/%d\n", iter, niter);
        struct ggml_context * ctx0 = ggml_init(params);

        get_random_dims(ne, 4);

        struct ggml_tensor * x[MAX_NARGS];

        // mul_mat
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                x[0] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                ne[1] = rand()%4 + 1;
                x[1] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * m = ggml_mul_mat(ctx0, x[1], x[0]);
                struct ggml_tensor * f = ggml_sum(ctx0, m);

                printf("testing: mul_mat, [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] = [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] * [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                           m->ne[0],    m->ne[1],    m->ne[2],    m->ne[3],
                        x[1]->ne[0], x[1]->ne[1], x[1]->ne[2], x[1]->ne[3],
                        x[0]->ne[0], x[0]->ne[1], x[0]->ne[2], x[0]->ne[3]);

                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);

                if (ndims <= 2) {
                    check_gradient("mul_mat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                } else {
                    struct ggml_cgraph gf = ggml_build_forward(m);
                    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);
                }

                check_mat_mul(m, x[1], x[0]);
            }
        }

        // mul_mat (transposed)
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                x[0] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                ne[1] = ne[0];
                ne[0] = rand()%4 + 1;
                x[1] = ggml_cont(ctx0, ggml_transpose(ctx0, get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f)));

                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * m = ggml_mul_mat(ctx0, x[1], x[0]);
                struct ggml_tensor * f = ggml_sum(ctx0, m);

                printf("testing: mul_mat, [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] = [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] * [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                           m->ne[0],    m->ne[1],    m->ne[2],    m->ne[3],
                        x[1]->ne[0], x[1]->ne[1], x[1]->ne[2], x[1]->ne[3],
                        x[0]->ne[0], x[0]->ne[1], x[0]->ne[2], x[0]->ne[3]);

                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);

                if (ndims <= 2) {
                    check_gradient("mul_mat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                } else {
                    struct ggml_cgraph gf = ggml_build_forward(m);
                    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);
                }

                check_mat_mul(m, x[1], x[0]);
            }
        }
        ggml_free(ctx0);
    }

    return 0;
}
