#pragma once

struct ggml_context;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_mtl_context;

struct ggml_mtl_context * mnist_mtl_init(
        struct ggml_context * ctx_data,
        struct ggml_context * ctx_eval,
        struct ggml_context * ctx_work,
        struct ggml_cgraph  * gf);

void mnist_mtl_free(struct ggml_mtl_context * ctx);

int mnist_mtl_eval(
        struct ggml_mtl_context * ctx,
        struct ggml_cgraph      * gf);

#ifdef __cplusplus
}
#endif
