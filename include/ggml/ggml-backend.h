#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif
    struct ggml_backend;
    struct ggml_backend_buffer;

    // type-erased backend-specific types / wrappers
    typedef void * ggml_backend_context_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef void * ggml_backend_buffer_context_t;

    //
    // backend buffer
    //

    struct ggml_backend_buffer_i {
        void   (*free_buffer)   (struct ggml_backend_buffer * buffer);
        void * (*get_base)      (struct ggml_backend_buffer * buffer); // get base pointer
        size_t (*get_alloc_size)(struct ggml_backend_buffer * buffer, struct ggml_tensor * tensor); // pre-allocation callback
        void   (*init_tensor)   (struct ggml_backend_buffer * buffer, struct ggml_tensor * tensor); // post-allocation callback
        void   (*free_tensor)   (struct ggml_backend_buffer * buffer, struct ggml_tensor * tensor); // pre-free callback
    };

    struct ggml_backend_buffer {
        struct ggml_backend * backend;

        struct ggml_backend_buffer_i interface;

        ggml_backend_buffer_context_t context;

        size_t size; // GG: can we absorb the size inside the context?
    };

    // backend buffer functions
    GGML_API struct ggml_backend_buffer * ggml_backend_buffer_init(
            struct ggml_backend                  * backend,
            struct ggml_backend_buffer_i           interface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size);

    GGML_API void   ggml_backend_buffer_free          (struct ggml_backend_buffer * buffer);
    GGML_API size_t ggml_backend_buffer_get_alignment (struct ggml_backend_buffer * buffer);
    GGML_API void * ggml_backend_buffer_get_base      (struct ggml_backend_buffer * buffer);
    GGML_API size_t ggml_backend_buffer_get_size      (struct ggml_backend_buffer * buffer);
    GGML_API size_t ggml_backend_buffer_get_alloc_size(struct ggml_backend_buffer * buffer, struct ggml_tensor * tensor);
    GGML_API void   ggml_backend_buffer_init_tensor   (struct ggml_backend_buffer * buffer, struct ggml_tensor * tensor);
    GGML_API void   ggml_backend_buffer_free_tensor   (struct ggml_backend_buffer * buffer, struct ggml_tensor * tensor);

    //
    // backend
    //

    struct ggml_backend_i {
        const char * (*get_name)(struct ggml_backend * backend);

        void (*free)(struct ggml_backend * backend);

        // buffer allocation
        struct ggml_backend_buffer * (*alloc_buffer)(struct ggml_backend * backend, size_t size);

        // get buffer alignment
        size_t (*get_alignment)(struct ggml_backend * backend);

        // tensor data access
        // these functions can be asynchronous, helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(struct ggml_backend * backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(struct ggml_backend * backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*synchronize)     (struct ggml_backend * backend);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(struct ggml_backend * backend, struct ggml_tensor * src, struct ggml_tensor * dst);
        void (*cpy_tensor_to)  (struct ggml_backend * backend, struct ggml_tensor * src, struct ggml_tensor * dst);

        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (struct ggml_backend * backend, struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (struct ggml_backend * backend, ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_compute)(struct ggml_backend * backend, ggml_backend_graph_plan_t plan);

        // compute graph without a plan
        void (*graph_compute)(struct ggml_backend * backend, struct ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*supports_op)(struct ggml_backend * backend, const struct ggml_tensor * op);
    };

    struct ggml_backend {
        struct ggml_backend_i interface;

        ggml_backend_context_t context;
    };

    // backend helper functions
    GGML_API struct ggml_backend * ggml_get_backend(const struct ggml_tensor * tensor);

    GGML_API const char * ggml_backend_name(struct ggml_backend * backend);
    GGML_API void         ggml_backend_free(struct ggml_backend * backend);

    GGML_API struct ggml_backend_buffer * ggml_backend_alloc_buffer(struct ggml_backend * backend, size_t size);

    GGML_API size_t ggml_backend_get_alignment(struct ggml_backend * backend);

    GGML_API void ggml_backend_tensor_set_async(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(struct ggml_backend * backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create (struct ggml_backend * backend, struct ggml_cgraph * cgraph);

    GGML_API void ggml_backend_graph_plan_free   (struct ggml_backend * backend, ggml_backend_graph_plan_t plan);
    GGML_API void ggml_backend_graph_plan_compute(struct ggml_backend * backend, ggml_backend_graph_plan_t plan);
    GGML_API void ggml_backend_graph_compute     (struct ggml_backend * backend, struct ggml_cgraph * cgraph);
    GGML_API bool ggml_backend_supports_op       (struct ggml_backend * backend, const struct ggml_tensor * op);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

    //
    // CPU backend
    //

    GGML_API struct ggml_backend * ggml_backend_cpu_init(void);

    GGML_API void ggml_backend_cpu_set_n_threads(struct ggml_backend * backend_cpu, int n_threads);

    GGML_API struct ggml_backend_buffer * ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

    ///////////////////////////

#if 0
    // graph splitting
    #define GGML_MAX_SPLITS 200
    #define GGML_MAX_SPLIT_INPUTS 4

    struct ggml_graph_split {
        char name[GGML_MAX_NAME];
        struct ggml_context * ctx;
        struct ggml_tensor  * src_inputs[GGML_MAX_SPLIT_INPUTS + 1];
        struct ggml_tensor  * dst_inputs[GGML_MAX_SPLIT_INPUTS + 1];
        struct ggml_cgraph  * graph;
    };

    // TODO: this shouldn't be fixed size, allocate from ggml_context
    struct ggml_graph_splits {
        int n_splits;
        struct ggml_graph_split splits[GGML_MAX_SPLITS];
    };

    // TODO: allocate in ggml_context
    GGML_API struct ggml_graph_splits ggml_graph_split_init(void);

    // this won't be needed once we can allocate graphs from a ggml_context
    GGML_API void ggml_graph_splits_free(struct ggml_graph_splits * splits);

    // add a split to the graph - single and multiple inputs versions
    GGML_API void ggml_graph_splits_add(struct ggml_graph_splits * splits, struct ggml_tensor ** input, struct ggml_context * ctx, const char * fmt, ...);
    GGML_API void ggml_graph_splits_add_n(struct ggml_graph_splits * splits, struct ggml_tensor *** inputs, struct ggml_context * ctx, const char * fmt, ...);

    // build graphs for all splits
    GGML_API void ggml_graph_splits_build_forward(struct ggml_graph_splits * splits, struct ggml_tensor * output);

    // compute
    GGML_API void ggml_graph_splits_compute(struct ggml_graph_splits * splits);

    // graph tensor allocator
    GGML_API void ggml_graph_allocate_tensors(struct ggml_cgraph * graph, struct ggml_context * ctx);
    GGML_API void ggml_graph_splits_allocate_tensors(struct ggml_graph_splits * splits);

    // automatically split a graph into multiple graphs based on the location of the tensors
    GGML_API struct ggml_graph_splits ggml_graph_split(struct ggml_cgraph * graph, struct ggml_context * ctx);
#endif

#ifdef  __cplusplus
}
#endif
