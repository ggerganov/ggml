#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-backend-impl.h>
#include <stdio.h>
#include <stdlib.h>


bool is_pow2(size_t x) {
    return (x & (x - 1)) == 0;
}

void test_buffer(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_get_default_buffer_type(backend) == buft);

    //ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, 1024);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 1024);

    GGML_ASSERT(buffer != NULL);

    GGML_ASSERT(is_pow2(ggml_backend_buffer_get_alignment(buffer)));

    GGML_ASSERT(ggml_backend_buffer_get_base(buffer) != NULL);

    GGML_ASSERT(ggml_backend_buffer_get_size(buffer) >= 1024);

    struct ggml_init_params params = {
        /* .mem_size = */ 1024,
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);

    GGML_ASSERT(ggml_backend_buffer_get_alloc_size(buffer, tensor) >= 10 * sizeof(float));

    ggml_tallocr_t allocr = ggml_tallocr_new_from_buffer(buffer);
    ggml_tallocr_alloc(allocr, tensor);

    GGML_ASSERT(tensor->data != NULL);

    GGML_ASSERT(tensor->data >= ggml_backend_buffer_get_base(buffer));

    // TODO:
    // supports_backend
    // get/set tensor
    // cpy tensor from/to

    ggml_tallocr_free(allocr);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    // enumerate backends
    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        printf("Backend %zu/%zu (%s)\n", i + 1, ggml_backend_reg_get_count(), ggml_backend_reg_get_name(i));
        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);
        printf("  Backend name: %s\n", ggml_backend_name(backend));
        test_buffer(backend, ggml_backend_reg_get_default_buffer_type(i));
        ggml_backend_free(backend);
        printf("  OK\n\n");
    }
}
