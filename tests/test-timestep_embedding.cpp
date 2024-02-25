#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

void set_timestep_embedding(struct ggml_tensor* timesteps, struct ggml_tensor* embedding, int dim, int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [dim, N]
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < timesteps->ne[0]; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = ggml_get_f32_1d(timesteps, i) * freqs[j];
            ggml_tensor_set_f32(embedding, std::cos(arg), j, i);
            ggml_tensor_set_f32(embedding, std::sin(arg), j + half, i);
        }
        if (dim % 2 != 0) {
            *(float*)((char*)embedding->data + i * embedding->nb[1] + dim * embedding->nb[0]) = 0;
        }
    }
}

struct ggml_tensor* new_timestep_embedding(struct ggml_context* ctx,
                                           struct ggml_tensor* timesteps,
                                           int dim,
                                           int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [dim, N]
    int acutual_dim = dim;
    if (dim % 2 != 0) {
        acutual_dim = dim + 1;
    }
    struct ggml_tensor* embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, acutual_dim, timesteps->ne[0]);
    set_timestep_embedding(timesteps, embedding, dim, max_period);
    return embedding;
}

int main(int argc, const char** argv) {
    std::vector<float> ts = {12, 24};
    int dim = 15;
    int max_period = 10000;
    {
        struct ggml_init_params params;
        params.mem_size = 16 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc = false;
        // memory allocation happens here
        struct ggml_context* ctx = ggml_init(params);
        
        struct ggml_tensor* timesteps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ts.size());
        memcpy(timesteps->data, ts.data(), ggml_nbytes(timesteps));
        struct ggml_tensor* embedding = new_timestep_embedding(ctx, timesteps, dim, max_period);
        
        float* vec1 = ggml_get_data_f32(embedding);
        for (int i = 0; i < ggml_nelements(embedding); i++) {
            printf("%.4f ", *(vec1 + i));
        }
        printf("\n");
    }
    printf("-----------------------------------\n");
    {
        bool use_gpu = true;
        ggml_backend_t backend = NULL;
        ggml_backend_buffer_t params_buffer = NULL;

        #ifdef GGML_USE_CUBLAS
        if (use_gpu) {
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
            backend = ggml_backend_cuda_init(0);
            if (!backend) {
                fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
            }
        }
        #endif

        int num_tensors = 2;
        int buffer_size = 1024;
        struct ggml_init_params params = {
                /*.mem_size   =*/ggml_tensor_overhead() * num_tensors + 2 * 1024 * 1024,
                /*.mem_size   =*/ NULL,
                /*.mem_size   =*/ true,
        };

        if(!backend) {
            // fallback to CPU backend
            backend = ggml_backend_cpu_init();
        }

        struct ggml_context* ctx = ggml_init(params);


        struct ggml_tensor* timesteps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ts.size());

        params_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // load data to buffer
        if(ggml_backend_is_cpu(backend)) {
            memcpy(timesteps->data, ts.data(), ggml_nbytes(timesteps));
        } else {
            ggml_backend_tensor_set(timesteps, ts.data(), 0, ggml_nbytes(timesteps));
        }


        struct ggml_tensor * t = ggml_timestep_embedding(ctx, timesteps, dim, max_period);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t);

        ggml_gallocr_alloc_graph(galloc, graph);

        int n_threads = 4;

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

        ggml_backend_graph_compute(backend, graph);

        float* output = new float[ggml_nelements(t)];
        ggml_backend_tensor_get(t, output, 0, ggml_nbytes(t));

        for (int i = 0; i < ggml_nelements(t); i++) {
            printf("%.4f ", output[i]);
        }
        printf("\n");

        free(output);
        ggml_free(ctx);
        ggml_backend_buffer_free(params_buffer);
        ggml_backend_free(backend);
        ggml_gallocr_free(galloc);
    }

    return 0;
}
