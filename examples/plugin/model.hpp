#pragma once
#include <cstdint>

struct ggml_tensor;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_context;
enum ggml_type;
struct ggml_backend_buffer;
struct ggml_allocr;

struct model {
    struct weights_t {
        ggml_tensor* w = nullptr;
    } weights;

    ggml_backend_t backend = nullptr;

    ggml_context* wctx = nullptr;
    ggml_backend_buffer* wbuf = nullptr; // weights buffer

    ggml_backend_buffer* cbuf = nullptr; // compute buffer
    ggml_allocr* callocr = nullptr; // compute allocator

    const int64_t size;
    const ggml_type type;

    model(ggml_backend_t be, int64_t s, ggml_type t, void* weights_data);
    ~model();

    void compute(void* output, void* input);
};

// util
template <typename Vec>
size_t data_size(const Vec& vec) {
    return vec.size() * sizeof(typename Vec::value_type);
}
