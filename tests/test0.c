#include "ggml/ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * t1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 10);
    struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_I16, 10, 20);
    struct ggml_tensor * t3 = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 10, 20, 30);

    assert(t1->n_dims == 1);
    assert(t1->ne[0]  == 10);
    assert(t1->nb[1]  == 10*sizeof(float));

    assert(t2->n_dims == 2);
    assert(t2->ne[0]  == 10);
    assert(t2->ne[1]  == 20);
    assert(t2->nb[1]  == 10*sizeof(int16_t));
    assert(t2->nb[2]  == 10*20*sizeof(int16_t));

    assert(t3->n_dims == 3);
    assert(t3->ne[0]  == 10);
    assert(t3->ne[1]  == 20);
    assert(t3->ne[2]  == 30);
    assert(t3->nb[1]  == 10*sizeof(int32_t));
    assert(t3->nb[2]  == 10*20*sizeof(int32_t));
    assert(t3->nb[3]  == 10*20*30*sizeof(int32_t));

    ggml_print_objects(ctx0);

    ggml_free(ctx0);

    return 0;
}
