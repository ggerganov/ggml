#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <thread>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a {};
    struct ggml_tensor * b {};

    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend {};
    ggml_backend_t cpu_backend {};
    ggml_backend_sched_t sched {};

    // storage for the graph and tensors
    std::vector<uint8_t> buf;
};

// initialize data of matrices to perform matrix multiplication
const int rows_A = 4, cols_A = 2;

float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};

const int rows_B = 3, cols_B = 2;
/* Transpose([
    10, 9, 5,
    5, 9, 4
]) 2 rows, 3 cols */
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9, 9,
    5, 4
};


// initialize the backends and scheduler
void init_model(simple_model & model) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_load_all();

    model.backend = ggml_backend_init_best();
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
    model.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(simple_model& model) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later
    };

    // create a context to build the graph
    struct ggml_context * ctx = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    // create tensors
    model.a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    // result = a*b^T
    struct ggml_tensor * result = ggml_mul_mat(ctx, model.a, model.b);

    // build operations nodes
    ggml_build_forward_expand(gf, result);

    ggml_free(ctx);

    return gf;
}

// compute with backend
struct ggml_tensor * compute(simple_model & model, struct ggml_cgraph * gf) {
    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_alloc_graph(model.sched, gf);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));

    // compute the graph
    ggml_backend_sched_graph_compute(model.sched, gf);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

int main(void) {
    ggml_time_init();

    std::thread worker([]() {
    simple_model model;
    init_model(model);

    struct ggml_cgraph * gf = build_graph(model);

    // perform computation
    struct ggml_tensor * result = compute(model, gf);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:
    // [ 60.00 55.00 50.00 110.00
    //  90.00 54.00 54.00 126.00
    //  42.00 29.00 28.00 64.00 ]

    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // release backend memory and free backend
    ggml_backend_sched_free(model.sched);
    ggml_backend_free(model.backend);
    ggml_backend_free(model.cpu_backend);
    });
    worker.join();
    return 0;
}
