#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

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

    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float* a, float* b) {
    // initialize the backend
#ifdef GGML_USE_CUBLAS
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if(!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 2;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 2, 2);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 2, 2);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model, struct ggml_allocr * allocr) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result = a*b^T
    struct ggml_tensor* result = ggml_mul_mat(ctx0, model.a, model.b);

    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor* compute(const simple_model & model, struct ggml_allocr * allocr) {
    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = build_graph(model, allocr);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

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

    // calculate the temporaly memory required to compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model, allocr);
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        ggml_allocr_free(allocr);

        // compute the required memory
        allocr = ggml_allocr_new_from_backend(model.backend, mem_size);
        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    // perform computation
    struct ggml_tensor * result = compute(model, allocr);

    // create a array to print result
    float* out_data = new float[2 * 2];

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data, 0, ggml_nbytes(result));

    // expected result:
    // [92.00, 59.00, 82.00, 34.00]
    printf("mult mat:\n[%.2f, %.2f, %.2f, %.2f]\n",
                out_data[0], out_data[1], out_data[2], out_data[3]);

    // release backend memory used for computation
    ggml_allocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
