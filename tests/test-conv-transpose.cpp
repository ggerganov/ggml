#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#define GGML_USE_CUBLAS

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    
    
    float adata[] = {1,2,3};
    float bdata[] = {1,2};
 
    size_t buffer_size = 0;
    {
        buffer_size += 3* ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += 2* ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024;
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 3);
    model.b = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 2);

    // create a allocator
    ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, adata, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, adata, 0, ggml_nbytes(model.a));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, bdata, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, bdata, 0, ggml_nbytes(model.b));
    }
}

struct ggml_cgraph * build_graph(const test_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    int s0 = 1;
    int p0 = 0;
    int d0 = 1;

    struct ggml_tensor* conv1d_transpose_res = ggml_conv_transpose_1d(ctx0, model.a, model.b, s0, p0, d0);
    ggml_set_name(conv1d_transpose_res, "conv1d_transpose_res");
    ggml_build_forward_expand(gf, conv1d_transpose_res);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph* compute_graph(const test_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    return gf;
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, true);

    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * conv1d_transpose_res = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
       if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv1d_transpose_res") == 0) {
            conv1d_transpose_res = gf_res->nodes[i];
        }
    }

    float* conv1d_transpose_data = new float[ggml_nelements(conv1d_transpose_res)];

    ggml_backend_tensor_get(conv1d_transpose_res, conv1d_transpose_data, 0, ggml_nbytes(conv1d_transpose_res));

    const int n_conv_transpose_1d_test = 4;

    float expected_conv1d[n_conv_transpose_1d_test] = {
       1.00f,4.00f,7.00f,6.00f
    };
   
    

    printf("\nPerforming test:\n");

    bool passed = true;
    for(int i = 0; i < n_conv_transpose_1d_test; i++) {
        if(
            conv1d_transpose_data[i] != expected_conv1d[i]) {
            std::cout << "index: " << i << std::endl;
            std::cout << "expected: " << expected_conv1d[i] << std::endl;
            std::cout << "actual: " << conv1d_transpose_data[i] << std::endl;
            passed = false;
            break;
        }
    }

    printf("ggml_conv_1d_transpose (%d): %s\n", (int) ggml_nelements(conv1d_transpose_res), passed && (ggml_nelements(conv1d_transpose_res) == n_conv_transpose_1d_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
