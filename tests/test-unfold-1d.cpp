#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

//#define GGML_USE_CUBLAS


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
#include <cmath>
#include <iostream>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * a_0;
    struct ggml_tensor * a_1;
    struct ggml_tensor * a_2;
    struct ggml_tensor * a_3;
    struct ggml_tensor * a_4;
    struct ggml_tensor * a_5;




    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    
    


    float data[1024];
    for (int i = 0; i < 1024; ++i) {
        data[i] = (float)i;
    }



 
    size_t buffer_size = 0;
    {
        buffer_size += 2 * 6 * ggml_type_size(GGML_TYPE_F32); // tensor a_0
    
        buffer_size += 1024;
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 1;
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
    model.a_0 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 6,2);


    // create a allocator
    ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a_0);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a_0->data, data, ggml_nbytes(model.a_0));
    } else {
        ggml_backend_tensor_set(model.a_0, data, 0, ggml_nbytes(model.a_0));
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

    int k = 3;
    int s = 3;

    struct ggml_tensor* pad_res_0 = ggml_unfold_1d(ctx0, model.a_0, k, s);
    ggml_set_name(pad_res_0, "pad_res_0");
    ggml_build_forward_expand(gf, pad_res_0);


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

    struct ggml_tensor * pad_res_0 = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
       if(strcmp(ggml_get_name(gf_res->nodes[i]), "pad_res_0") == 0) {
            pad_res_0 = gf_res->nodes[i];
        }
    }

    float* pad_data_0 = new float[ggml_nelements(pad_res_0)];

    ggml_backend_tensor_get(pad_res_0, pad_data_0, 0, ggml_nbytes(pad_res_0));

    const int n_pad_test_0 = 2 *2 * 3;

    float expected_pad_reflect_0[n_pad_test_0] = {7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0};

    printf("\nPerforming test:\n");

    bool passed = true;
    for(int i = 0; i < n_pad_test_0; i++) {
        if(
            pad_data_0[i] != expected_pad_reflect_0[i]) {
            std::cout << "index: " << i << std::endl;
            std::cout << "expected: " << expected_pad_reflect_0[i] << std::endl;
            std::cout << "actual: " << pad_data_0[i] << std::endl;
            passed = false;
            break;
        }
    }

    std::cout << ggml_nelements(pad_res_0) << std::endl;

    printf("ggml_pad_ext (%d): %s\n", (int) ggml_nelements(pad_res_0), passed && (ggml_nelements(pad_res_0) == n_pad_test_0) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");


    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
