#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

// #define GGML_USE_CUBLAS

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

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * image;
    struct ggml_tensor * kernel_w;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    const int KW = 3, KH = 3, IC = 1, OC = 1;
    const int IW = 8, IH = 8, N = 1;

    // Initialize kernel data
    const int kernel_elements = KW * KH * IC * OC;
    float kernel_d[kernel_elements] = {
        2.f, 1.0f, 4.f,
        2.5f, 1.5f, 2.f,
        2.f, 1.5f, 4.f,
    };

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> kernel_d_f16(kernel_elements);
    ggml_fp32_to_fp16_row(kernel_d, kernel_d_f16.data(), kernel_elements);

    // Initialize image data
    const int image_elements = IW * IH * IC * N;
    float image_d[image_elements] = {
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 2.f, 3.f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 1.f, 4.f,
        2.5f, 3.f, 1.f, 1.f, 2.f, 2.f, 3.f, 2.f,
        2.5f, 3.f, 4.f, 3.f, 1.f, 2.f, 2.f, 2.5f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        2.5f, 3.f, 1.f, 4.f, 2.f, 2.f, 2.f, 2.f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 1.f, 2.f
    };

    size_t buffer_size = 0;
    {
        buffer_size += kernel_elements * ggml_type_sizef(GGML_TYPE_F16); // tensor a
        buffer_size += image_elements * ggml_type_sizef(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f KB\n", __func__, (buffer_size / 1024.f));

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
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
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
    model.kernel_w = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  KW, KH, IC, OC);
    model.image = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, IC, N);

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.kernel_w);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.kernel_w->data, kernel_d_f16.data(), ggml_nbytes(model.kernel_w));
    } else {
        ggml_backend_tensor_set(model.kernel_w, kernel_d_f16.data(), 0, ggml_nbytes(model.kernel_w));
    }

    // alloc memory
    ggml_allocr_alloc(alloc, model.image);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.image->data, image_d, ggml_nbytes(model.image));
    } else {
        ggml_backend_tensor_set(model.image, image_d, 0, ggml_nbytes(model.image));
    }

    ggml_allocr_free(alloc);
}

struct ggml_cgraph * build_graph(const test_model& model, struct ggml_allocr * allocr) {
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

    int s0 = 1;
    int s1 = 1;
    int p0 = 1;
    int p1 = 1;
    int d0 = 1;
    int d1 = 1;

    // split conv2d in fundamental methods for test unit
    struct ggml_tensor* im2col_0 = ggml_im2col(ctx0, model.kernel_w, model.image, s0, s1, p0, p1, d0, d1, true);
    ggml_set_name(im2col_0, "im2col_res");
    ggml_build_forward_expand(gf, im2col_0);

    // recalculate for avoid fragmentation
    struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.kernel_w, model.image, s0, s1, p0, p1, d0, d1);
    ggml_set_name(conv2d_res, "conv2d_res");
    ggml_build_forward_expand(gf, conv2d_res);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * compute_graph(const test_model & model, struct ggml_allocr * allocr) {
    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = build_graph(model, allocr);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
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

    ggml_backend_buffer_t buf_compute; // for compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model, allocr);
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        ggml_allocr_free(allocr);

        // compute the required memory
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);
        fprintf(stderr, "%s: compute buffer size: %.2f KB\n", __func__, mem_size / 1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * im2col_res = NULL;
    struct ggml_tensor * conv2d_res = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf_res->nodes[i]), "im2col_res") == 0) {
            im2col_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv2d_res") == 0) {
            conv2d_res = gf_res->nodes[i];
        }
    }

    uint16_t* im2col_data = new uint16_t[ggml_nelements(im2col_res)];
    float* conv2d_data = new float[ggml_nelements(conv2d_res)];

    ggml_backend_tensor_get(im2col_res, im2col_data, 0, ggml_nbytes(im2col_res));
    ggml_backend_tensor_get(conv2d_res, conv2d_data, 0, ggml_nbytes(conv2d_res));

    const int n_conv2d_test = 64;
    const int n_im2col_test = 576;

    float expected_conv2d [n_conv2d_test] = {
        25.500f, 44.250f, 31.500f, 33.000f, 22.500f, 27.000f, 31.000f, 16.500f,
        40.000f, 68.250f, 45.500f, 50.000f, 34.500f, 37.000f, 49.500f, 23.500f,
        40.000f, 56.250f, 41.000f, 44.000f, 34.500f, 43.000f, 49.000f, 24.500f,
        40.000f, 62.250f, 49.000f, 41.500f, 37.000f, 37.000f, 51.500f, 24.250f,
        40.000f, 56.250f, 46.500f, 45.000f, 38.000f, 42.500f, 43.000f, 23.750f,
        40.000f, 56.250f, 61.000f, 46.500f, 43.500f, 39.000f, 43.000f, 21.500f,
        40.000f, 62.250f, 47.000f, 47.000f, 42.000f, 37.000f, 39.500f, 19.000f,
        24.250f, 30.750f, 38.500f, 29.500f, 27.500f, 24.000f, 24.500f, 11.500f };

    uint16_t expected_im2col[n_im2col_test] = {
        0, 0, 0, 0, 16640, 16896, 0, 16640, 16896,
        0, 0, 0, 16640, 16896, 17408, 16640, 16896, 17408,
        0, 0, 0, 16896, 17408, 15360, 16896, 17408, 15360,
        0, 0, 0, 17408, 15360, 16384, 17408, 15360, 16384,
        0, 0, 0, 15360, 16384, 16384, 15360, 16384, 16384,
        0, 0, 0, 16384, 16384, 16384, 16384, 16384, 16384,
        0, 0, 0, 16384, 16384, 16384, 16384, 16384, 16896,
        0, 0, 0, 16384, 16384, 0, 16384, 16896, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 17408, 16640, 16896, 17408,
        16896, 17408, 15360, 16896, 17408, 15360, 16896, 17408, 15360,
        17408, 15360, 16384, 17408, 15360, 16384, 17408, 15360, 16384,
        15360, 16384, 16384, 15360, 16384, 16384, 15360, 16384, 16384,
        16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 15360,
        16384, 16384, 16384, 16384, 16384, 16896, 16384, 15360, 17408,
        16384, 16384, 0, 16384, 16896, 0, 15360, 17408, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 17408, 16640, 16896, 15360,
        16896, 17408, 15360, 16896, 17408, 15360, 16896, 15360, 15360,
        17408, 15360, 16384, 17408, 15360, 16384, 15360, 15360, 16384,
        15360, 16384, 16384, 15360, 16384, 16384, 15360, 16384, 16384,
        16384, 16384, 16384, 16384, 16384, 15360, 16384, 16384, 16896,
        16384, 16384, 16896, 16384, 15360, 17408, 16384, 16896, 16384,
        16384, 16896, 0, 15360, 17408, 0, 16896, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 15360, 16640, 16896, 17408,
        16896, 17408, 15360, 16896, 15360, 15360, 16896, 17408, 16896,
        17408, 15360, 16384, 15360, 15360, 16384, 17408, 16896, 15360,
        15360, 16384, 16384, 15360, 16384, 16384, 16896, 15360, 16384,
        16384, 16384, 15360, 16384, 16384, 16896, 15360, 16384, 16384,
        16384, 15360, 17408, 16384, 16896, 16384, 16384, 16384, 16640,
        15360, 17408, 0, 16896, 16384, 0, 16384, 16640, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 15360, 16640, 16896, 17408, 16640, 16896, 17408,
        16896, 15360, 15360, 16896, 17408, 16896, 16896, 17408, 15360,
        15360, 15360, 16384, 17408, 16896, 15360, 17408, 15360, 16384,
        15360, 16384, 16384, 16896, 15360, 16384, 15360, 16384, 16384,
        16384, 16384, 16896, 15360, 16384, 16384, 16384, 16384, 16384,
        16384, 16896, 16384, 16384, 16384, 16640, 16384, 16384, 16384,
        16896, 16384, 0, 16384, 16640, 0, 16384, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 17408, 16640, 16896, 15360,
        16896, 17408, 16896, 16896, 17408, 15360, 16896, 15360, 17408,
        17408, 16896, 15360, 17408, 15360, 16384, 15360, 17408, 16384,
        16896, 15360, 16384, 15360, 16384, 16384, 17408, 16384, 16384,
        15360, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
        16384, 16384, 16640, 16384, 16384, 16384, 16384, 16384, 16384,
        16384, 16640, 0, 16384, 16384, 0, 16384, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 15360, 16640, 16896, 17408,
        16896, 17408, 15360, 16896, 15360, 17408, 16896, 17408, 15360,
        17408, 15360, 16384, 15360, 17408, 16384, 17408, 15360, 16384,
        15360, 16384, 16384, 17408, 16384, 16384, 15360, 16384, 16384,
        16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 15360,
        16384, 16384, 16384, 16384, 16384, 16384, 16384, 15360, 16384,
        16384, 16384, 0, 16384, 16384, 0, 15360, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 0, 0,
        16640, 16896, 15360, 16640, 16896, 17408, 0, 0, 0,
        16896, 15360, 17408, 16896, 17408, 15360, 0, 0, 0,
        15360, 17408, 16384, 17408, 15360, 16384, 0, 0, 0,
        17408, 16384, 16384, 15360, 16384, 16384, 0, 0, 0,
        16384, 16384, 16384, 16384, 16384, 15360, 0, 0, 0,
        16384, 16384, 16384, 16384, 15360, 16384, 0, 0, 0,
        16384, 16384, 0, 15360, 16384, 0, 0, 0, 0
    };

    printf("\nConvolution with im2col\n");

    for(int i = 0; i < ggml_nelements(conv2d_res);i++) {
        if(i> 0 && i % conv2d_res->ne[0] == 0) {
            printf("\n");
        }
        printf("%.3ff, ", conv2d_data[i]);
    }

    printf("\nPerforming test:\n");

    bool passed = true;
    for(int i = 0; i < n_im2col_test; i++) {
        if(
            im2col_data[i] != expected_im2col[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_im2col (%d): %s\n", (int) ggml_nelements(im2col_res), passed && (ggml_nelements(im2col_res) == n_im2col_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    passed = true;
    for(int i = 0; i < n_conv2d_test; i++) {
        if(conv2d_data[i] != expected_conv2d[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_conv2d (%d): %s\n", (int) ggml_nelements(conv2d_res), passed && (ggml_nelements(conv2d_res) == n_conv2d_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}
