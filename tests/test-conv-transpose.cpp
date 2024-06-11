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
    struct ggml_tensor * a_0;
    struct ggml_tensor * b_0;

    struct ggml_tensor * a_1;
    struct ggml_tensor * b_1;

    struct ggml_tensor * a_2;
    struct ggml_tensor * b_2;



    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    
    
    float adata_0[] = {1,2,3};
    float bdata_0[] = {1,2};

    float adata_1[] = {1,2,3,3,2,1};
    float bdata_1[] = {2,3,1,1,3,2};

    float adata_2[] =  {3,2,1,1,2,3,1,2,3,3,2,1};
    float bdata_2[] =  {2,3,1,1,3,2};



 
    size_t buffer_size = 0;
    {
        buffer_size += 3* ggml_type_size(GGML_TYPE_F32); // tensor a_0
        buffer_size += 2* ggml_type_size(GGML_TYPE_F32); // tensor b_0

        buffer_size += 6* ggml_type_size(GGML_TYPE_F32); // tensor a_0
        buffer_size += 6* ggml_type_size(GGML_TYPE_F32); // tensor b_0

        buffer_size += 12* ggml_type_size(GGML_TYPE_F32); // tensor a_0
        buffer_size += 6* ggml_type_size(GGML_TYPE_F32); // tensor b_0

    
        buffer_size += 1024;
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 6;
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
    model.a_0 = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 3);
    model.b_0 = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 2);


    model.a_1 = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 3,1,2);
    model.b_1 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3,2);

    model.a_2 = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 3,2,2);
    model.b_2 = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3,2);



    // create a allocator
    ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a_0);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a_0->data, adata_0, ggml_nbytes(model.a_0));
    } else {
        ggml_backend_tensor_set(model.a_0, adata_0, 0, ggml_nbytes(model.a_0));
    }


    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a_1);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a_1->data, adata_1, ggml_nbytes(model.a_1));
    } else {
        ggml_backend_tensor_set(model.a_1, adata_1, 0, ggml_nbytes(model.a_1));
    }

     // alloc memory
    ggml_tallocr_alloc(&alloc, model.a_2);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a_2->data, adata_2, ggml_nbytes(model.a_2));
    } else {
        ggml_backend_tensor_set(model.a_2, adata_2, 0, ggml_nbytes(model.a_2));
    }



    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b_0);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b_0->data, bdata_0, ggml_nbytes(model.b_0));
    } else {
        ggml_backend_tensor_set(model.b_0, bdata_0, 0, ggml_nbytes(model.b_0));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b_1);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b_1->data, bdata_1, ggml_nbytes(model.b_1));
    } else {
        ggml_backend_tensor_set(model.b_1, bdata_1, 0, ggml_nbytes(model.b_1));
    }

     // alloc memory
    ggml_tallocr_alloc(&alloc, model.b_2);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b_2->data, bdata_2, ggml_nbytes(model.b_2));
    } else {
        ggml_backend_tensor_set(model.b_2, bdata_2, 0, ggml_nbytes(model.b_2));
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

    struct ggml_tensor* conv1d_transpose_res_0 = ggml_conv_transpose_1d(ctx0, model.a_0, model.b_0, s0, p0, d0);
    ggml_set_name(conv1d_transpose_res_0, "conv1d_transpose_res_0");
    ggml_build_forward_expand(gf, conv1d_transpose_res_0);


    s0 = 1;
    p0 = 0;
    d0 = 1;

    struct ggml_tensor* conv1d_transpose_res_1 = ggml_conv_transpose_1d(ctx0, model.a_1, model.b_1, s0, p0, d0);
    ggml_set_name(conv1d_transpose_res_1, "conv1d_transpose_res_1");
    ggml_build_forward_expand(gf, conv1d_transpose_res_1);

    s0 = 1;
    p0 = 0;
    d0 = 1;

    struct ggml_tensor* conv1d_transpose_res_2 = ggml_conv_transpose_1d(ctx0, model.a_2, model.b_2, s0, p0, d0);
    ggml_set_name(conv1d_transpose_res_2, "conv1d_transpose_res_2");
    ggml_build_forward_expand(gf, conv1d_transpose_res_2);

    s0 = 2;
    p0 = 0;
    d0 = 1;

    struct ggml_tensor* conv1d_transpose_res_3 = ggml_conv_transpose_1d(ctx0, model.a_2, model.b_2, s0, p0, d0);
    ggml_set_name(conv1d_transpose_res_3, "conv1d_transpose_res_3");
    ggml_build_forward_expand(gf, conv1d_transpose_res_3);



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

    struct ggml_tensor * conv1d_transpose_res_0 = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
       if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv1d_transpose_res_0") == 0) {
            conv1d_transpose_res_0 = gf_res->nodes[i];
        }
    }

    float* conv1d_transpose_data_0 = new float[ggml_nelements(conv1d_transpose_res_0)];

    ggml_backend_tensor_get(conv1d_transpose_res_0, conv1d_transpose_data_0, 0, ggml_nbytes(conv1d_transpose_res_0));

    const int n_conv_transpose_1d_test_0 = 4;

    float expected_conv1d_0[n_conv_transpose_1d_test_0] = {
       1.00f,4.00f,7.00f,6.00f
    };


    struct ggml_tensor * conv1d_transpose_res_1 = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
       if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv1d_transpose_res_1") == 0) {
            conv1d_transpose_res_1 = gf_res->nodes[i];
        }
    }

    float* conv1d_transpose_data_1 = new float[ggml_nelements(conv1d_transpose_res_1)];

    ggml_backend_tensor_get(conv1d_transpose_res_1, conv1d_transpose_data_1, 0, ggml_nbytes(conv1d_transpose_res_1));





    const int n_conv_transpose_1d_test_1 = 5;

    float expected_conv1d_1[n_conv_transpose_1d_test_1] =
       {5.0f, 18.0f, 26.0f, 18.0f,  5.0f};


    struct ggml_tensor * conv1d_transpose_res_2 = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
       if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv1d_transpose_res_2") == 0) {
            conv1d_transpose_res_2 = gf_res->nodes[i];
        }
    }

    float* conv1d_transpose_data_2 = new float[ggml_nelements(conv1d_transpose_res_2)];

    ggml_backend_tensor_get(conv1d_transpose_res_2, conv1d_transpose_data_2, 0, ggml_nbytes(conv1d_transpose_res_2));


    const int n_conv_transpose_1d_test_2 = 10;

    float expected_conv1d_2[n_conv_transpose_1d_test_2] =
       {7.0f, 18.0f, 22.0f, 18.0f,  7.0f,
         5.0f, 18.0f, 26.0f, 18.0f,  5.0f};
   
    struct ggml_tensor * conv1d_transpose_res_3 = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
       if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv1d_transpose_res_3") == 0) {
            conv1d_transpose_res_3 = gf_res->nodes[i];
        }
    }

    float* conv1d_transpose_data_3 = new float[ggml_nelements(conv1d_transpose_res_3)];

    ggml_backend_tensor_get(conv1d_transpose_res_3, conv1d_transpose_data_3, 0, ggml_nbytes(conv1d_transpose_res_3));


    const int n_conv_transpose_1d_test_3 = 14;

    float expected_conv1d_3[n_conv_transpose_1d_test_3] =
       {7.0f,  6.0f, 17.0f, 12.0f, 17.0f,  6.0f,  7.0f
         ,5.0f,  6.0f, 19.0f, 12.0f, 19.0f,  6.0f,  5.0f};
   
    
   
    
   
    

    printf("\nPerforming test:\n");

    bool passed = true;
    for(int i = 0; i < n_conv_transpose_1d_test_0; i++) {
        if(
            conv1d_transpose_data_0[i] != expected_conv1d_0[i]) {
            std::cout << "index: " << i << std::endl;
            std::cout << "expected: " << expected_conv1d_0[i] << std::endl;
            std::cout << "actual: " << conv1d_transpose_data_0[i] << std::endl;
            passed = false;
            break;
        }
    }

    printf("ggml_conv_1d_transpose (%d): %s\n", (int) ggml_nelements(conv1d_transpose_res_0), passed && (ggml_nelements(conv1d_transpose_res_0) == n_conv_transpose_1d_test_0) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    
    for(int i = 0; i < n_conv_transpose_1d_test_1; i++) {
        if(
            conv1d_transpose_data_1[i] != expected_conv1d_1[i]) {
            std::cout << "index: " << i << std::endl;
            std::cout << "expected: " << expected_conv1d_1[i] << std::endl;
            std::cout << "actual: " << conv1d_transpose_data_1[i] << std::endl;
            passed = false;
        }
    }

    printf("ggml_conv_1d_transpose (%d): %s\n", (int) ggml_nelements(conv1d_transpose_res_1), passed && (ggml_nelements(conv1d_transpose_res_1) == n_conv_transpose_1d_test_1) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");


    for(int i = 0; i < n_conv_transpose_1d_test_2; i++) {
        if(
            conv1d_transpose_data_2[i] != expected_conv1d_2[i]) {
            std::cout << "index: " << i << std::endl;
            std::cout << "expected: " << expected_conv1d_2[i] << std::endl;
            std::cout << "actual: " << conv1d_transpose_data_2[i] << std::endl;
            passed = false;
        }
    }

    printf("ggml_conv_1d_transpose (%d): %s\n", (int) ggml_nelements(conv1d_transpose_res_2), passed && (ggml_nelements(conv1d_transpose_res_2) == n_conv_transpose_1d_test_2) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");



    for(int i = 0; i < n_conv_transpose_1d_test_3; i++) {
        if(
            conv1d_transpose_data_3[i] != expected_conv1d_3[i]) {
            std::cout << "index: " << i << std::endl;
            std::cout << "expected: " << expected_conv1d_3[i] << std::endl;
            std::cout << "actual: " << conv1d_transpose_data_3[i] << std::endl;
            passed = false;
        }
    }

    printf("ggml_conv_1d_transpose (%d): %s\n", (int) ggml_nelements(conv1d_transpose_res_3), passed && (ggml_nelements(conv1d_transpose_res_3) == n_conv_transpose_1d_test_3) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");


    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
