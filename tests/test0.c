#include "ggml/ggml.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * t1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 10);
    struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_I16, 10, 20);
    struct ggml_tensor * t3 = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 10, 20, 30);

    GGML_ASSERT(t1->n_dims == 1);
    GGML_ASSERT(t1->ne[0]  == 10);
    GGML_ASSERT(t1->nb[1]  == 10*sizeof(float));

    GGML_ASSERT(t2->n_dims == 2);
    GGML_ASSERT(t2->ne[0]  == 10);
    GGML_ASSERT(t2->ne[1]  == 20);
    GGML_ASSERT(t2->nb[1]  == 10*sizeof(int16_t));
    GGML_ASSERT(t2->nb[2]  == 10*20*sizeof(int16_t));

    GGML_ASSERT(t3->n_dims == 3);
    GGML_ASSERT(t3->ne[0]  == 10);
    GGML_ASSERT(t3->ne[1]  == 20);
    GGML_ASSERT(t3->ne[2]  == 30);
    GGML_ASSERT(t3->nb[1]  == 10*sizeof(int32_t));
    GGML_ASSERT(t3->nb[2]  == 10*20*sizeof(int32_t));
    GGML_ASSERT(t3->nb[3]  == 10*20*30*sizeof(int32_t));

    ggml_print_objects(ctx0);

    ggml_free(ctx0);

    return 0;
}
