#include "ggml/ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

bool is_close(float a, float b, float epsilon) {
    return fabs(a - b) < epsilon;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 1024*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    //struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_TYPE_LBFGS);

    opt_params.n_threads = (argc > 1) ? atoi(argv[1]) : 8;

    const int NP = 1 << 12;
    const int NF = 1 << 8;

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * F = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, NF, NP);
    struct ggml_tensor * l = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, NP);

    // regularization weight
    struct ggml_tensor * lambda = ggml_new_f32(ctx0, 1e-5f);

    srand(0);

    for (int j = 0; j < NP; j++) {
        const float ll = j < NP/2 ? 1.0f : -1.0f;
        ((float *)l->data)[j] = ll;

        for (int i = 0; i < NF; i++) {
            ((float *)F->data)[j*NF + i] = ((ll > 0 && i < NF/2 ? 1.0f : ll < 0 && i >= NF/2 ? 1.0f : 0.0f) + ((float)rand()/(float)RAND_MAX - 0.5f)*0.1f)/(0.5f*NF);
        }
    }

    {
        // initial guess
        struct ggml_tensor * x = ggml_set_f32(ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, NF), 0.0f);

        ggml_set_param(ctx0, x);

        // f = sum[(fj*x - l)^2]/n + lambda*|x^2|
        struct ggml_tensor * f =
            ggml_add(ctx0,
                    ggml_div(ctx0,
                        ggml_sum(ctx0,
                            ggml_sqr(ctx0,
                                ggml_sub(ctx0,
                                    ggml_mul_mat(ctx0, F, x),
                                    l)
                                )
                            ),
                        ggml_new_f32(ctx0, (float)NP)
                        ),
                    ggml_mul(ctx0,
                        ggml_sum(ctx0, ggml_sqr(ctx0, x)),
                        lambda)
                    );

        enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

        GGML_ASSERT(res == GGML_OPT_RESULT_OK);

        // print results
        for (int i = 0; i < 16; i++) {
            printf("x[%3d] = %g\n", i, ((float *)x->data)[i]);
        }
        printf("...\n");
        for (int i = NF - 16; i < NF; i++) {
            printf("x[%3d] = %g\n", i, ((float *)x->data)[i]);
        }
        printf("\n");

        for (int i = 0; i < NF; ++i) {
            if (i < NF/2) {
                GGML_ASSERT(is_close(((float *)x->data)[i],  1.0f, 1e-2f));
            } else {
                GGML_ASSERT(is_close(((float *)x->data)[i], -1.0f, 1e-2f));
            }
        }
    }

    ggml_free(ctx0);

    return 0;
}
