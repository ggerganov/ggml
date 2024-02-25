#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows
#include "ggml/ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

bool is_close(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    //struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    //opt_params.adam.alpha = 0.01f;

    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_TYPE_LBFGS);

    // original threads: 8
    int nthreads = 8;
    const char *env = getenv("GGML_NTHREADS");
    if (env != NULL) {
        nthreads = atoi(env);
    }
    if (argc > 1) {
        nthreads = atoi(argv[1]);
    }
    opt_params.n_threads = nthreads;
    printf("test2: n_threads:%d\n", opt_params.n_threads);

    const float xi[] = {  1.0f,  2.0f,  3.0f,  4.0f,  5.0f , 6.0f,  7.0f,  8.0f,  9.0f,  10.0f, };
          float yi[] = { 15.0f, 25.0f, 35.0f, 45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, };

    const int n = sizeof(xi)/sizeof(xi[0]);

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n);
    struct ggml_tensor * y = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n);

    for (int i = 0; i < n; i++) {
        ((float *) x->data)[i] = xi[i];
        ((float *) y->data)[i] = yi[i];
    }

    {
        struct ggml_tensor * t0 = ggml_new_f32(ctx0, 0.0f);
        struct ggml_tensor * t1 = ggml_new_f32(ctx0, 0.0f);

        // initialize auto-diff parameters:
        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = sum_i[(t0 + t1*x_i - y_i)^2]/(2n)
        struct ggml_tensor * f =
            ggml_div(ctx0,
                    ggml_sum(ctx0,
                        ggml_sqr(ctx0,
                            ggml_sub(ctx0,
                                ggml_add(ctx0,
                                    ggml_mul(ctx0, x, ggml_repeat(ctx0, t1, x)),
                                    ggml_repeat(ctx0, t0, x)),
                                y)
                            )
                        ),
                    ggml_new_f32(ctx0, 2.0f*n));

        enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

        printf("t0 = %f\n", ggml_get_f32_1d(t0, 0));
        printf("t1 = %f\n", ggml_get_f32_1d(t1, 0));

        GGML_ASSERT(res == GGML_OPT_RESULT_OK);

        GGML_ASSERT(is_close(ggml_get_f32_1d(t0, 0),  5.0f, 1e-3f));
        GGML_ASSERT(is_close(ggml_get_f32_1d(t1, 0), 10.0f, 1e-3f));
    }

    {
        struct ggml_tensor * t0 = ggml_new_f32(ctx0, -1.0f);
        struct ggml_tensor * t1 = ggml_new_f32(ctx0,  9.0f);

        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = 0.5*sum_i[abs(t0 + t1*x_i - y_i)]/n
        struct ggml_tensor * f =
            ggml_mul(ctx0,
                    ggml_new_f32(ctx0, 1.0/(2*n)),
                    ggml_sum(ctx0,
                        ggml_abs(ctx0,
                            ggml_sub(ctx0,
                                ggml_add(ctx0,
                                    ggml_mul(ctx0, x, ggml_repeat(ctx0, t1, x)),
                                    ggml_repeat(ctx0, t0, x)),
                                y)
                            )
                        )
                    );


        enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

        GGML_ASSERT(res == GGML_OPT_RESULT_OK);
        GGML_ASSERT(is_close(ggml_get_f32_1d(t0, 0),  5.0f, 1e-2f));
        GGML_ASSERT(is_close(ggml_get_f32_1d(t1, 0), 10.0f, 1e-2f));
    }

    {
        struct ggml_tensor * t0 = ggml_new_f32(ctx0,  5.0f);
        struct ggml_tensor * t1 = ggml_new_f32(ctx0, -4.0f);

        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = t0^2 + t1^2
        struct ggml_tensor * f =
            ggml_add(ctx0,
                    ggml_sqr(ctx0, t0),
                    ggml_sqr(ctx0, t1)
                    );

        enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

        GGML_ASSERT(res == GGML_OPT_RESULT_OK);
        GGML_ASSERT(is_close(ggml_get_f32_1d(f,  0), 0.0f, 1e-3f));
        GGML_ASSERT(is_close(ggml_get_f32_1d(t0, 0), 0.0f, 1e-3f));
        GGML_ASSERT(is_close(ggml_get_f32_1d(t1, 0), 0.0f, 1e-3f));
    }

    /////////////////////////////////////////

    {
        struct ggml_tensor * t0 = ggml_new_f32(ctx0, -7.0f);
        struct ggml_tensor * t1 = ggml_new_f32(ctx0,  8.0f);

        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = (t0 + 2*t1 - 7)^2 + (2*t0 + t1 - 5)^2
        struct ggml_tensor * f =
            ggml_add(ctx0,
                    ggml_sqr(ctx0,
                        ggml_sub(ctx0,
                            ggml_add(ctx0,
                                t0,
                                ggml_mul(ctx0, t1, ggml_new_f32(ctx0, 2.0f))),
                            ggml_new_f32(ctx0, 7.0f)
                            )
                        ),
                    ggml_sqr(ctx0,
                        ggml_sub(ctx0,
                            ggml_add(ctx0,
                                ggml_mul(ctx0, t0, ggml_new_f32(ctx0, 2.0f)),
                                t1),
                            ggml_new_f32(ctx0, 5.0f)
                            )
                        )
                    );

        enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

        GGML_ASSERT(res == GGML_OPT_RESULT_OK);
        GGML_ASSERT(is_close(ggml_get_f32_1d(f,  0), 0.0f, 1e-3f));
        GGML_ASSERT(is_close(ggml_get_f32_1d(t0, 0), 1.0f, 1e-3f));
        GGML_ASSERT(is_close(ggml_get_f32_1d(t1, 0), 3.0f, 1e-3f));
    }

    ggml_free(ctx0);

    return 0;
}
