#include "model.hpp"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cassert>

model::model(ggml_backend_t be, int64_t s, ggml_type t, void* weights_data)
    : backend(be)
    , size(s)
    , type(t)
{
    assert(weights_data);
    static constexpr size_t numWeightTensors = sizeof(weights_t) / sizeof(ggml_tensor*);
    wctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * numWeightTensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
        });
    weights.w = ggml_new_tensor_1d(wctx, type, size);
    wbuf = ggml_backend_alloc_buffer(backend, 0);
    auto wallocr = ggml_allocr_new_from_buffer(wbuf);
    ggml_allocr_set_tensor_external_data(wallocr, weights.w, weights_data, 0);
    ggml_allocr_free(wallocr);

    cbuf = ggml_backend_alloc_buffer(backend, 0);
    callocr = ggml_allocr_new_from_buffer(cbuf);
}

model::~model() {
    ggml_free(wctx);
    ggml_backend_buffer_free(wbuf);
    ggml_allocr_free(callocr);
    ggml_backend_buffer_free(cbuf);
}

struct io_tensors {
    ggml_tensor* input = nullptr;
    ggml_tensor* output = nullptr;
};

void model::compute(void* output, void* input) {
    assert(input);
    assert(output);

    static constexpr size_t num_io_tensors = sizeof(io_tensors) / sizeof(ggml_tensor*);
    auto cctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * num_io_tensors + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
        });

    io_tensors io = {};
    io.input = ggml_new_tensor_1d(cctx, type, size);
    io.output = ggml_add(cctx, io.input, weights.w);

    ggml_allocr_set_tensor_external_data(callocr, io.input, input, 0);
    ggml_allocr_set_tensor_external_data(callocr, io.output, output, 0);

    auto graph = ggml_new_graph(cctx);
    ggml_build_forward_expand(graph, io.output);

    ggml_backend_graph_compute(backend, graph);

    ggml_allocr_reset(callocr);
    ggml_free(cctx);
}
