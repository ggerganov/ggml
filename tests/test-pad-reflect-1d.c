#include "ggml.h"
#include "ggml-cpu.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct ggml_context* make_ctx(void) {
    struct ggml_init_params params = {
        .mem_size = 2 * 1024 * 1024,
    };
    return ggml_init(params);
}

void check_tensor(struct ggml_tensor * t, float * expected_t_d, int ne0, int ne1, int ne2) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(t->ne[0] == ne0);
    GGML_ASSERT(t->ne[1] == ne1);
    GGML_ASSERT(t->ne[2] == ne2);
    for (int i2 = 0; i2 < ne2; ++i2) {
        for (int i1 = 0; i1 < ne1; ++i1) {
            for (int i0 = 0; i0 < ne0; ++i0) {
                float expected = *(expected_t_d + i2 * ne1 * ne0 + i1 * ne0 + i0);
                float actual = ggml_get_data_f32(t)[i2 * ne1 * ne0 + i1 * ne0 + i0];
                if (expected != actual) {
                    printf("expected %.1f, got %.1f at (%d,%d,%d)\n", expected, actual, i0, i1, i2);
                }
                GGML_ASSERT(expected == actual);
            }
        }
    }
}

void test_pad_reflect_1d(void) {
    // Test cases for different padding configurations
    {
        struct ggml_context * ctx = make_ctx();

        // Create a simple 1D input tensor [1, 2, 3, 4]
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        memcpy(t->data, input_data, ggml_nbytes(t));

        // Test case 1: pad left=1, right=1
        // Expected: [2, 1, 2, 3, 4, 3]
        float expected_1[] = {2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f};
        struct ggml_tensor * out_1 = ggml_pad_reflect_1d(ctx, t, 1, 1);

        // Test case 2: pad left=2, right=1
        // Expected: [3, 2, 1, 2, 3, 4, 3]
        float expected_2[] = {3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f};
        struct ggml_tensor * out_2 = ggml_pad_reflect_1d(ctx, t, 2, 1);

        // Test case 3: pad left=1, right=2
        // Expected: [2, 1, 2, 3, 4, 3, 2]
        float expected_3[] = {2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f};
        struct ggml_tensor * out_3 = ggml_pad_reflect_1d(ctx, t, 1, 2);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_1);
        ggml_build_forward_expand(gf, out_2);
        ggml_build_forward_expand(gf, out_3);

        ggml_graph_compute_with_ctx(ctx, gf, 1);

        check_tensor(out_1, expected_1, 6, 1, 1);
        check_tensor(out_2, expected_2, 7, 1, 1);
        check_tensor(out_3, expected_3, 7, 1, 1);

        ggml_free(ctx);
    }

    {
        struct ggml_context * ctx = make_ctx();

        // Create a 2D input tensor (5 columns × 4 rows)
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 4);
        float input_data[] = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // row 1
            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, // row 2
            11.0f, 12.0f, 13.0f, 14.0f, 15.0f, // row 3
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f  // row 4
        };
        memcpy(t->data, input_data, ggml_nbytes(t));

        // Test case 4: pad left=3, right=2 on a 2D tensor
        // Each row should be padded independently
        float expected_4[] = {
            4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f,  // row 1
            9.0f, 8.0f, 7.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 9.0f, 8.0f, // row 2
            14.0f, 13.0f, 12.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 14.0f, 13.0f, // row 3
            19.0f, 18.0f, 17.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 19.0f, 18.0f  // row 4
        };
        struct ggml_tensor * out_4 = ggml_pad_reflect_1d(ctx, t, 3, 2);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_4);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        check_tensor(out_4, expected_4, 10, 4, 1);

        ggml_free(ctx);
    }
}

int main(int argc, const char * argv[]) {
    test_pad_reflect_1d();
    return 0;
}
