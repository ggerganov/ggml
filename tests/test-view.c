#include "ggml/ggml.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
      .mem_size   = 16*1024*1024,
      .mem_buffer = NULL,
      .no_alloc   = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    int len = 10;
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, len);
    for (int i = 0; i < len; i++) {
        ggml_set_f32_1d(x, i, i);
    }

    struct ggml_tensor* view = ggml_view_1d(ctx, x, 5, 0);
    GGML_ASSERT(ggml_get_f32_1d(view, 0) == 0);
    GGML_ASSERT(ggml_get_f32_1d(view, 1) == 1);
    GGML_ASSERT(ggml_get_f32_1d(view, 2) == 2);
    GGML_ASSERT(ggml_get_f32_1d(view, 3) == 3);
    GGML_ASSERT(ggml_get_f32_1d(view, 4) == 4);

    view = ggml_view_1d(ctx, x, 5, 5);
    GGML_ASSERT(ggml_get_f32_1d(view, 0) == 5);
    GGML_ASSERT(ggml_get_f32_1d(view, 1) == 6);
    GGML_ASSERT(ggml_get_f32_1d(view, 2) == 7);
    GGML_ASSERT(ggml_get_f32_1d(view, 3) == 8);
    GGML_ASSERT(ggml_get_f32_1d(view, 4) == 9);

    ggml_free(ctx);

    return 0;
}
