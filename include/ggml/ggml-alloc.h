#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_backend;
struct ggml_backend_buffer;
typedef struct ggml_allocr * ggml_allocr_t;

// initialize allocator for use with CPU backend only
GGML_API ggml_allocr_t ggml_allocr_new(void * data, size_t size, size_t alignment);
GGML_API ggml_allocr_t ggml_allocr_new_measure(size_t alignment);

// initialize allocator for use with ggml-backend
GGML_API ggml_allocr_t ggml_allocr_new_from_buffer(struct ggml_backend_buffer * buffer);
GGML_API ggml_allocr_t ggml_allocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
GGML_API ggml_allocr_t ggml_allocr_new_measure_from_backend(struct ggml_backend * backend);

GGML_API struct ggml_backend_buffer * ggml_allocr_get_buffer(ggml_allocr_t alloc);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_API void   ggml_allocr_set_parse_seq(ggml_allocr_t alloc, const int * list, int n);

GGML_API void   ggml_allocr_free       (ggml_allocr_t alloc);
GGML_API bool   ggml_allocr_is_measure (ggml_allocr_t alloc);
GGML_API void   ggml_allocr_reset      (ggml_allocr_t alloc);
GGML_API void   ggml_allocr_alloc      (ggml_allocr_t alloc, struct ggml_tensor * tensor);
GGML_API size_t ggml_allocr_max_size   (ggml_allocr_t alloc);

GGML_API size_t ggml_allocr_alloc_graph(ggml_allocr_t alloc, struct ggml_cgraph * graph);

// Allocate tensors from the allocators given by the hash table
GGML_API void ggml_allocr_alloc_graph_n(
                    struct ggml_cgraph * graph,
                    const struct ggml_tensor * hash_keys[GGML_GRAPH_HASHTABLE_SIZE],
                    ggml_allocr_t hash_node_alloct[GGML_GRAPH_HASHTABLE_SIZE]);


#ifdef  __cplusplus
}
#endif
