#pragma once

struct ggml_context;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

int mnist_mtl_eval(
    struct ggml_context * ctx_data,
    struct ggml_context * ctx_eval,
    struct ggml_context * ctx_work,
    struct ggml_cgraph  * gf);

#ifdef __cplusplus
}
#endif
