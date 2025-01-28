#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

struct test_model {
    struct ggml_tensor * a = nullptr;
    struct ggml_tensor * b = nullptr;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer = NULL;
    struct ggml_context * ctx = nullptr;
    int M =0, N=0, K=1;
};

// in MB
size_t getCudaFreeMem() {
    size_t cudafree = 0;
    size_t cudatotal = 0;
    ggml_backend_cuda_get_device_memory(0, &cudafree, &cudatotal);
    return cudafree/1024/1024;
}

ggml_status load_model(test_model & model, unsigned S) {
    size_t totalFreeMem = getCudaFreeMem();
    printf("%s: cuda free: %ld MB \n", __func__, totalFreeMem);

    // for a 2d matrix multiplication: K = shared dim, M=num rows for the left tensor A, N=num cols for the right tensor B
    model.M = S;
    model.N = S;
    model.K = S;
    printf("%s: M=%d N=%d K=%d \n", __func__, model.M, model.N, model.K);

    size_t buffer_size = 0;
    {
        buffer_size += (model.M * model.K) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (model.K * model.N) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += (model.M * model.N) * ggml_type_size(GGML_TYPE_F32); // output tensor
        buffer_size += 1024; // overhead
    }
    printf("%s: backend buffer size = %ld KB\n", __func__, buffer_size/1024);

    int num_tensors = 3;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, //
            };

    // initialize the backend
    printf("%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0);
    if (!model.backend) {
        printf("%s: ggml_backend_cuda_init() failed\n", __func__);
        return GGML_STATUS_FAILED;
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);
    if (!model.buffer) {
        printf("%s: model.buffer null \n", __func__);
        return GGML_STATUS_ALLOC_FAILED;
    }

    printf("%s: buffer allocated. cuda free: %ld MB \n", __func__, getCudaFreeMem());

    // create context
    model.ctx = ggml_init(params);
    printf("%s: ctx created. cuda free: %ld MB \n", __func__, getCudaFreeMem());

    // create tensors
    printf("%s: creating input tensors...\n", __func__);
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.K, model.M);
    if (!model.a) {
        printf("%s: cannot create tensor a \n", __func__);
        abort();
    }
    model.a->name[0] = 'A';
    //printf("Matrix A: [%i, %i]\n", K, M);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.K, model.N);
    if(!model.b)
        abort();
    model.b->name[0] = 'B';
    //printf("Matrix B: [%i, %i]\n", K, N);
    printf("%s: tensors (a&b) created. cuda free: %ld MB \n", __func__, getCudaFreeMem());

    // create an allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory for a
    ggml_tallocr_alloc(&alloc, model.a);

    // alloc memory for b
    ggml_tallocr_alloc(&alloc, model.b);
    return GGML_STATUS_SUCCESS;
}


struct ggml_cgraph * build_graph(const test_model& model, ggml_tensor* a, ggml_tensor *b, unsigned repeat) {
    printf("build_graph %d...\n", repeat);
    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    printf("%s: graph buf size: %ld KB\n", __func__, buf_size/1024);
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);
    if (!ctx0) {
        printf("error: ggml_init returned null\n");
        return nullptr;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    if (!gf)
        return nullptr;

    // zT = x @ yT
    struct ggml_tensor * result = ggml_mul_mat(ctx0, a, ggml_cont(ctx0, b));
    if (!result) {
        printf("error:  ggml_mul_mat returned null\n");
        return nullptr;
    }

    // z = (zT)T
    struct ggml_tensor* T = ggml_transpose(ctx0, result);
    if (!T) {
        fprintf(stderr, "error: ggml_transpose returned null\n");
        return nullptr;
    }

    struct ggml_tensor* c = ggml_cont(ctx0, T);
    if (!c) {
        fprintf(stderr, "error: ggml_cont returned null\n");
        return nullptr;
    }

    std::vector<struct ggml_tensor *> outTensors;
    outTensors.push_back(c);
    for (unsigned i=0; i < repeat; i++) {
        struct ggml_tensor * d = ggml_mul_mat(ctx0, outTensors.back(), ggml_cont(ctx0, outTensors.back()));
        if (!d) {
            printf("error:  ggml_mul_mat returned null\n");
            return nullptr;
        }
        //printf("%s: matmul out: %s %ld %ld \n", __func__, d->name, d->ne[0], d->ne[1]);
        outTensors.push_back(d);
        c = ggml_concat(ctx0, c, d, 0);
    }

    ggml_build_forward_expand(gf, c);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

ggml_status compute(const test_model & model, ggml_gallocr_t allocr, unsigned repeat) {
    printf("compute ...\n");
    printf("compute: free device mem: %ld MB\n", getCudaFreeMem());

    ggml_tensor* ot = NULL;
    ggml_tensor* left = model.a;
    ggml_tensor* right = model.b;

    struct ggml_cgraph * gf = build_graph(model, left, right, repeat);
    printf("conpute: graph built. free cuda mem: %ld MB\n", getCudaFreeMem());

    // allocate tensors
    if (!ggml_gallocr_alloc_graph(allocr, gf))
        return GGML_STATUS_ALLOC_FAILED;

    printf("%s: graph buf allocated. free device mem: %ld MB\n", __func__, getCudaFreeMem());

    ggml_status status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS)
        return status;

    ggml_graph_print(gf);
    printf("compute: graph computed. free device mem: %ld MB\n", getCudaFreeMem());
    // in this case, the output tensor is the last one in the graph
    ot = ggml_graph_node(gf, -1);
    if (!ot)
        return GGML_STATUS_FAILED;
    printf("%s: output tensor shape: %ld x %ld name: %s\n", __func__, ot->ne[0], ot->ne[1], ot->name);

    return GGML_STATUS_SUCCESS;
}


int main(void) {
#ifndef GGML_USE_CUDA
    fprintf(stderr, "note: test-oom ony implemented for the cuda backend at the moment");
    return 0;
#endif

    const char* GGML_CUDA_NO_ABORT = getenv("GGML_CUDA_NO_ABORT");
    if (!GGML_CUDA_NO_ABORT) {
      fprintf(stderr, "warning: skipping: test-oom requires the GGML_CUDA_NO_ABORT envvar to be set\n");
      return 0;
    }

    test_model model;

    ggml_status status = load_model(model, 8192); // will also init the backend
    if (status != GGML_STATUS_SUCCESS) {
        printf("failed to load model");
        return GGML_EXIT_ABORTED;
    }

    ggml_gallocr_t allocr = NULL;
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) {
        printf("Cannot ggml_gallocr_new\n");
        return GGML_EXIT_ABORTED;
    }

    // will run multiple matmul in a lopp accumulating big output tensors. Should oom.
    status = compute(model, allocr, 160);
    if (status == GGML_STATUS_SUCCESS) {
        printf("main: compute failed to oom (matmul too small to oom the GPU? for loop too smal ?)\n");
        return GGML_EXIT_ABORTED;
    }
    printf("main: compute correctly OOM: ggml status=%d expected: %d \n", status, GGML_STATUS_ALLOC_FAILED);
    return GGML_EXIT_SUCCESS;
}
