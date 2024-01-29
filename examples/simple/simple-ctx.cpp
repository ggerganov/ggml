#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float* a, float* b) {
    size_t ctx_size = 0;
    {
        ctx_size += 2 * 2 * ggml_type_size(GGML_TYPE_F32); // tensor a
        ctx_size += 2 * 2 * ggml_type_size(GGML_TYPE_F32); // tensor b
        ctx_size += 2 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
    }

    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 2, 2);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 2, 2);

    memcpy(model.a->data, a, ggml_nbytes(model.a));
    memcpy(model.b->data, b, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    // result = a*b^T
    struct ggml_tensor* result = ggml_mul_mat(model.ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// compute with backend
struct ggml_tensor* compute(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(void)
{
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication

    float matrix_A[4] = {
        2, 8,
        5, 1
    };

    // transpose of [10, 5, 9, 9]
    float matrix_B[4] = {
        10, 9,
        5, 9
    };

    simple_model model;
    load_model(model, matrix_A, matrix_B);

    // perform computation in cpu
    struct ggml_tensor * result = compute(model);

    // get the result data pointer as a float array to print
    float* out_data = (float*)result->data;

    // expected result:
    // [92.00, 59.00, 82.00, 34.00]
    printf("mult mat:\n[%.2f, %.2f, %.2f, %.2f]\n",
                out_data[0], out_data[1], out_data[2], out_data[3]);

    // free memory
    ggml_free(model.ctx);
    return 0;
}
