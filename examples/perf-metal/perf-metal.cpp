// basic tool to experiment with the Metal backend
//
// 1. Get GPU trace of a dummy graph:
//
//   rm -rf /tmp/perf-metal.gputrace
//   make -j perf-metal && METAL_CAPTURE_ENABLED=1 ./bin/perf-metal
//   open /tmp/perf-metal.gputrace
//
//   https://github.com/ggerganov/llama.cpp/issues/9507
//

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cstdio>
#include <vector>
#include <thread>

int main(int argc, char ** argv) {
    int n_op = 1024;
    int n_iter = 128;

    if (argc > 1) {
        n_op = std::atoi(argv[1]);
    }

    if (argc > 2) {
        n_iter = std::atoi(argv[2]);
    }

    printf("%s: n_op = %d, n_iter = %d\n", __func__, n_op, n_iter);

    const int ne00 = 8;
    const int ne01 = 8;
    const int ne11 = 8;

    std::vector<float> data0(ne00*ne01, 1.0f);
    std::vector<float> data1(ne00*ne01, 1.0f/ne00);

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        return 1;
    }

    const size_t ctx_size = 2 * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne01);
    struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, ne11);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(t0, data0.data(), 0, ggml_nbytes(t0));
    ggml_backend_tensor_set(t1, data1.data(), 0, ggml_nbytes(t1));

    struct ggml_cgraph * gf = NULL;

    struct ggml_context * ctx_cgraph = NULL;

    // create a dummy compute graph:
    //
    // x = mul_mat(t0, t1)
    // x = x * 1.0f
    // x = mul_mat(x, t1)
    // x = x * 1.0f
    // ... repeat n_op times ...
    //
    {
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ 4*n_op*ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx_cgraph = ggml_init(params0);

        gf = ggml_new_graph_custom(ctx_cgraph, 4*n_op, false);

        struct ggml_tensor * cur = ggml_mul_mat(ctx_cgraph, t0, t1);
        cur = ggml_scale(ctx_cgraph, cur, 1.0f);

        for (int i = 0; i < n_op - 1; i++) {
            cur = ggml_mul_mat(ctx_cgraph, cur, t1);
            cur = ggml_scale(ctx_cgraph, cur, 1.0f);
        }

        cur = ggml_scale(ctx_cgraph, cur, 42.0f);

        ggml_build_forward_expand(gf, cur);
    }

    printf("%s: graph nodes = %d\n", __func__, ggml_graph_n_nodes(gf));

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    {
        // warm-up
        ggml_backend_graph_compute(backend, gf);

        const int64_t t_start = ggml_time_us();

        for (int iter = 0; iter < n_iter; iter++) {
            ggml_backend_graph_compute(backend, gf);
        }

        const int64_t t_end = ggml_time_us();

        // actual trace
        ggml_backend_metal_capture_next_compute(backend);
        ggml_backend_graph_compute(backend, gf);
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // NOTE: these intervals do not appear in the XCode trace!
        ggml_backend_metal_capture_next_compute(backend);
        ggml_backend_graph_compute(backend, gf);
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // NOTE: these intervals do not appear in the XCode trace!
        ggml_backend_metal_capture_next_compute(backend);
        ggml_backend_graph_compute(backend, gf);

        printf("%s: time = %f ms\n", __func__, (t_end - t_start) / 1000.0 / n_iter);
    }

    {
        struct ggml_tensor * res = ggml_graph_node(gf, -1);

        std::vector<float> data(res->ne[0] * res->ne[1], 0.0f);

        ggml_backend_tensor_get(res, data.data(), 0, ggml_nbytes(res));

        for (int i1 = 0; i1 < res->ne[1]; i1++) {
            for (int i0 = 0; i0 < res->ne[0]; i0++) {
                printf("%f ", data[i1*res->ne[0] + i0]);
            }
            printf("\n");
        }
    }

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return 0;
}
