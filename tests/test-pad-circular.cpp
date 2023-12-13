#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
//This is a very crude test, and not testing anything other than fp32, and not testing cuda or other optimizations
//I wrote it for my sanity of making sure this was working correctly
struct ggml_context* make_ctx(void) {
    struct ggml_init_params params;
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc=false;
    params.mem_buffer=NULL;

    return ggml_init(params);
}
int main(void) {
    bool debug = false;
    const int pad_amount=2;
    const int base_tensor_size=3;
    struct ggml_context * ctx = make_ctx();
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3);
    //Base tensor values is hardcoded
    float test_data[base_tensor_size][base_tensor_size] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
        };
    for (int i = 0; i < base_tensor_size; ++i) {
        for (int j = 0; j < base_tensor_size; ++j) {
            ggml_set_f32_1d(a, i * base_tensor_size + j, test_data[i][j]);
        }
    }
    if (debug) {
        printf("Base Tensor Data:\n");
        for (int i = 0; i < base_tensor_size; ++i) {
            for (int j = 0; j < base_tensor_size; ++j) {
                float val = ggml_get_f32_1d(a, i * base_tensor_size + j);
                printf("%4.1f ", val);
            }
            printf("\n");
        }
    }
    struct ggml_tensor * padded_a = ggml_pad_circular(ctx, a, pad_amount);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, padded_a);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    //Expected Result is also hardcoded
    float expected_pad1[base_tensor_size+pad_amount*2][base_tensor_size+pad_amount*2] = {
        {9, 7, 8, 9, 7},
        {3, 1, 2, 3, 1},
        {6, 4, 5, 6, 4},
        {9, 7, 8, 9, 7},
        {3, 1, 2, 3, 1}
    };
    float expected_pad2[base_tensor_size+pad_amount*2][base_tensor_size+pad_amount*2] = {
        {5, 6, 4, 5, 6, 4, 5},
        {8, 9, 7, 8, 9, 7, 8},
        {2, 3, 1, 2, 3, 1, 2},
        {5, 6, 4, 5, 6, 4, 5},
        {8, 9, 7, 8, 9, 7, 8},
        {2, 3, 1, 2, 3, 1, 2},
        {5, 6, 4, 5, 6, 4, 5}
    };
    bool passed = true;
    if(debug){
        printf("\nResult:\n");
    }
    for (int i = 0; i < base_tensor_size+pad_amount*2; ++i) {
        for (int j = 0; j < base_tensor_size+pad_amount*2; ++j) {
            float val = ggml_get_f32_1d(padded_a, i * (base_tensor_size+pad_amount*2) + j);

            if(debug){
                printf("%4.1f ", val);
            }
            if (val != expected_pad2[i][j]) {
                passed = false;
            }
        }
        printf("\n");
    }
    if (debug && !passed) {
        printf("Expected Result:\n");
        for (int i = 0; i < base_tensor_size+pad_amount*2; ++i) {
            for (int j = 0; j < base_tensor_size+pad_amount*2; ++j) {
                printf("%4.1f ", expected_pad2[i][j]);
            }
            printf("\n");
        }
    }
    if(debug){
        printf("\nPass: %s", passed ? "true" : "false");
    }
    ggml_free(ctx);
    return passed ? 0 : 1;
}
