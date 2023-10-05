#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED GGML_UNUSED

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
        struct ggml_backend                  * backend,
        struct ggml_backend_buffer_i           interface,
               ggml_backend_buffer_context_t   context,
               size_t                          size) {
    ggml_backend_buffer_t buffer = malloc(sizeof(struct ggml_backend_buffer));

    GGML_ASSERT(interface.get_base != NULL);

    (*buffer) = (struct ggml_backend_buffer) {
        /* .interface = */ interface,
        /* .backend   = */ backend,
        /* .context   = */ context,
        /* .size      = */ size,
    };

    return buffer;
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer->interface.free_buffer != NULL) {
        buffer->interface.free_buffer(buffer);
    }
    free(buffer);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_get_alignment(buffer->backend);
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->interface.get_base(buffer);
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    return buffer->size;
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (buffer->interface.get_alloc_size) {
        return buffer->interface.get_alloc_size(buffer, tensor);
    }
    return ggml_nbytes(tensor);
}

void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (buffer->interface.init_tensor) {
        buffer->interface.init_tensor(buffer, tensor);
    }
}

void ggml_backend_buffer_free_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (buffer->interface.free_tensor) {
        buffer->interface.free_tensor(buffer, tensor);
    }
}

// backend

ggml_backend_t ggml_get_backend(const struct ggml_tensor * tensor) {
    return tensor->buffer->backend;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    return backend->interface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    backend->interface.free(backend);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return backend->interface.alloc_buffer(backend, size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return backend->interface.get_alignment(backend);
}

void ggml_backend_tensor_set_async(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_get_backend(tensor)->interface.set_tensor_async(ggml_get_backend(tensor), tensor, data, offset, size);
}

void ggml_backend_tensor_get_async(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_get_backend(tensor)->interface.get_tensor_async(ggml_get_backend(tensor), tensor, data, offset, size);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_get_backend(tensor)->interface.set_tensor_async(ggml_get_backend(tensor), tensor, data, offset, size);
    ggml_get_backend(tensor)->interface.synchronize(ggml_get_backend(tensor));
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_get_backend(tensor)->interface.get_tensor_async(ggml_get_backend(tensor), tensor, data, offset, size);
    ggml_get_backend(tensor)->interface.synchronize(ggml_get_backend(tensor));
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    backend->interface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->interface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    backend->interface.graph_plan_free(backend, plan);
}

void ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    backend->interface.graph_plan_compute(backend, plan);
}

void ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    backend->interface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return backend->interface.supports_op(backend, op);
}

// backend copy

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    //printf("src: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", src->name, (int)src->ne[0], (int)src->ne[1], (int)src->ne[2], (int)src->ne[3], (int)src->nb[0], (int)src->nb[1], (int)src->nb[2], (int)src->nb[3]);
    //printf("dst: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3], (int)dst->nb[0], (int)dst->nb[1], (int)dst->nb[2], (int)dst->nb[3]);
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    // printf("cpy tensor %s from %s to %s (%lu bytes)\n", src->name, ggml_backend_name(src->backend), ggml_backend_name(dst->backend), ggml_nbytes(src));

    if (src == dst) {
        return;
    }

    // TODO: allow backends to support copy to/from same backend

    if (ggml_get_backend(dst)->interface.cpy_tensor_from != NULL) {
        ggml_get_backend(dst)->interface.cpy_tensor_from(ggml_get_backend(dst)->context, src, dst);
    } else if (ggml_get_backend(src)->interface.cpy_tensor_to != NULL) {
        ggml_get_backend(src)->interface.cpy_tensor_to(ggml_get_backend(src)->context, src, dst);
    } else {
        // shouldn't be hit when copying from/to CPU
        #ifndef NDEBUG
        fprintf(stderr, "ggml_backend_tensor_copy: neither cpy_tensor_from nor cpy_tensor_to are implemented for backends %s and %s, falling back to get/set\n", ggml_backend_name(src->buffer->backend), ggml_backend_name(dst->buffer->backend));
        #endif
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

// backend CPU

struct ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;
};

static const char * ggml_backend_cpu_name(ggml_backend_t backend) {
    return "CPU";

    UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
    UNUSED(buffer);
}

static struct ggml_backend_buffer_i cpu_backend_buffer_i = {
    /* .free_buffer    = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_cpu_buffer_get_base,
    /* .get_alloc_size = */ NULL, // defaults to ggml_nbytes
    /* .init_tensor    = */ NULL, // no initialization required
    /* .free_tensor    = */ NULL, // no cleanup required
};

// for buffers from ptr, free is not called
static struct ggml_backend_buffer_i cpu_backend_buffer_i_from_ptr = {
    /* .free_buffer    = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base       = */ ggml_backend_cpu_buffer_get_base,
    /* .get_alloc_size = */ NULL, // defaults to ggml_nbytes
    /* .init_tensor    = */ NULL,
    /* .free_tensor    = */ NULL,
};

static const size_t TENSOR_ALIGNMENT = 64; // should be enough for AVX 512

static ggml_backend_buffer_t ggml_backend_cpu_alloc_buffer(ggml_backend_t backend, size_t size) {
    void * data = malloc(size + TENSOR_ALIGNMENT); // malloc may return an address that is not aligned
                                                   // TODO: maybe use GGML_ALIGNED_MALLOC?
    return ggml_backend_buffer_init(backend, cpu_backend_buffer_i, data, size);
}

static size_t ggml_backend_cpu_get_alignment(ggml_backend_t backend) {
    return TENSOR_ALIGNMENT;
    UNUSED(backend);
}

static void ggml_backend_cpu_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(backend);
}

static void ggml_backend_cpu_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy(data, (const char *)tensor->data + offset, size);

    UNUSED(backend);
}

static void ggml_backend_cpu_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
}

static void ggml_backend_cpu_cpy_tensor_from(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));

    UNUSED(backend);
}

static void ggml_backend_cpu_cpy_tensor_to(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst) {
    // for a backend such as CUDA that can queue async calls, it is ok to do this asynchronously, but it may not be the case for other backends
    ggml_backend_tensor_set_async(dst, src->data, 0, ggml_nbytes(src));

    UNUSED(backend);
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = malloc(sizeof(struct ggml_backend_plan_cpu));

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph;

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    UNUSED(backend);
}

static void ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    UNUSED(backend);
}

static void ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);

    if (cpu_ctx->work_size < cplan.work_size) {
        // TODO: may be faster to free and use malloc to avoid the copy
        cpu_ctx->work_data = realloc(cpu_ctx->work_data, cplan.work_size);
        cpu_ctx->work_size = cplan.work_size;
    }

    cplan.work_data = cpu_ctx->work_data;

    ggml_graph_compute(cgraph, &cplan);
}

static bool ggml_backend_cpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return true;
    UNUSED(backend);
    UNUSED(op);
}

static struct ggml_backend_i cpu_backend_i = {
    /* .get_name            = */ ggml_backend_cpu_name,
    /* .free                = */ ggml_backend_cpu_free,
    /* .alloc_buffer        = */ ggml_backend_cpu_alloc_buffer,
    /* .get_alignment       = */ ggml_backend_cpu_get_alignment,
    /* .set_tensor_async    = */ ggml_backend_cpu_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_cpu_get_tensor_async,
    /* .synchronize         = */ ggml_backend_cpu_synchronize,
    /* .cpy_tensor_from     = */ ggml_backend_cpu_cpy_tensor_from,
    /* .cpy_tensor_to       = */ ggml_backend_cpu_cpy_tensor_to,
    /* .graph_plan_create   = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free     = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute  = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute       = */ ggml_backend_cpu_graph_compute,
    /* .supports_op         = */ ggml_backend_cpu_supports_op,
};

ggml_backend_t ggml_backend_cpu_init(void) {
    struct ggml_backend_cpu_context * ctx = malloc(sizeof(struct ggml_backend_cpu_context));

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->work_data = NULL;
    ctx->work_size = 0;

    ggml_backend_t cpu_backend = malloc(sizeof(struct ggml_backend));

    *cpu_backend = (struct ggml_backend) {
        /* .interface = */ cpu_backend_i,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    // TODO: NULL backend?
    // TODO: no free
    return ggml_backend_buffer_init(NULL, cpu_backend_buffer_i_from_ptr, ptr, size);
}

#if 0
// splits

struct ggml_graph_splits ggml_graph_split_init(void) {
    struct ggml_graph_splits splits = {0};
    return splits;
}

// TODO: this can be removed after allocating the graphs in a ggml_context
void ggml_graph_splits_free(struct ggml_graph_splits * splits) {
    for (int i = 0; i < splits->n_splits; i++) {
        if (splits->splits[i].graph) {
            free(splits->splits[i].graph);
        }
    }
}

static void ggml_graph_splits_add_n_va(struct ggml_graph_splits * splits, struct ggml_tensor *** inputs, struct ggml_context * ctx, const char * fmt, va_list args) {
    GGML_ASSERT(splits->n_splits < GGML_MAX_SPLITS);

    struct ggml_graph_split * split = &splits->splits[splits->n_splits];


    if (splits->n_splits == 0) {
        // always add the first split
        int i = 0;
        while (inputs[i] != NULL) {
            GGML_ASSERT(i < GGML_MAX_SPLIT_INPUTS);
            split->src_inputs[i] = *inputs[i];
            split->dst_inputs[i] = *inputs[i];
            i++;
        }
        split->src_inputs[i] = NULL;
        split->dst_inputs[i] = NULL;
        split->ctx = ctx;
    }
    // check if the split is on the same context as the previous one
    else if (splits->n_splits > 0 && splits->splits[splits->n_splits - 1].ctx == ctx) {
        // add to the previous split
        char name[GGML_MAX_NAME - 2];
        int n = vsnprintf(name, sizeof(name), fmt, args);
        char new_name[GGML_MAX_NAME];
        snprintf(new_name, sizeof(new_name), "%.*s,%s", GGML_MAX_NAME - n - 2, splits->splits[splits->n_splits - 1].name, name);
        strcpy(splits->splits[splits->n_splits - 1].name, new_name);
        return;
    } else {
        // add a new split
        int i = 0;
        while (inputs[i] != NULL) {
            GGML_ASSERT(i < GGML_MAX_SPLIT_INPUTS);
            split->src_inputs[i] = *inputs[i];
            split->dst_inputs[i] = ggml_dup_tensor(ctx, *inputs[i]);
            ggml_format_name(split->dst_inputs[i], "%s (split output)", split->src_inputs[i]->name);
            // TODO: maybe support different layouts in ggml_backend_cpy_tensor instead
            for (int j = 0; j < GGML_MAX_DIMS; j++) {
                split->dst_inputs[i]->nb[j] = split->src_inputs[i]->nb[j];
            }
            ggml_set_name(split->dst_inputs[i], ggml_get_name(*inputs[i]));
            *inputs[i] = split->dst_inputs[i];
            i++;
        }
        split->src_inputs[i] = NULL;
        split->dst_inputs[i] = NULL;
        split->ctx = ctx;
    }

    vsnprintf(split->name, GGML_MAX_NAME, fmt, args);
    split->graph = NULL;
    splits->n_splits++;
}

void ggml_graph_splits_add_n(struct ggml_graph_splits * splits, struct ggml_tensor *** input, struct ggml_context * ctx, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ggml_graph_splits_add_n_va(splits, input, ctx, fmt, args);
    va_end(args);
}

void ggml_graph_splits_add(struct ggml_graph_splits * splits, struct ggml_tensor ** input, struct ggml_context * ctx, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ggml_graph_splits_add_n_va(splits, (struct ggml_tensor**[2]){ input, NULL }, ctx, fmt, args);
    va_end(args);
}

void ggml_graph_splits_build_forward(struct ggml_graph_splits * splits, struct ggml_tensor * output) {
    struct ggml_tensor *last_outputs[2] = { output, NULL };
    struct ggml_tensor ** outputs;

    for (int i = 0; i < splits->n_splits; i++) {
        struct ggml_graph_split * split = &splits->splits[i];

        if (i < splits->n_splits - 1) {
            outputs = splits->splits[i + 1].src_inputs;
        } else {
            outputs = last_outputs;
        }

        // build the graph
        // TODO: allocate graphs in context
        split->graph = (struct ggml_cgraph *) malloc(sizeof(struct ggml_cgraph));
        memset(split->graph, 0, sizeof(struct ggml_cgraph));
        for (int j = 0; outputs[j] != NULL; j++) {
            ggml_build_forward_expand(split->graph, outputs[j]);
        }

        for (int j = 1; j < split->graph->n_nodes; j++) {
            if (split->graph->nodes[j]->backend != split->graph->nodes[0]->backend) {
                fprintf(stderr, "split %s: node %s has different backend (%s) than the first node (%s)\n",
                    split->name, split->graph->nodes[j]->name,
                    ggml_backend_name(split->graph->nodes[j]->backend_s),
                    ggml_backend_name(split->graph->nodes[0]->backend_s));
            }
        }
        for (int j = 1; j < split->graph->n_leafs; j++) {
            if (split->graph->leafs[j]->backend != split->graph->leafs[0]->backend) {
                fprintf(stderr, "split %s: leaf %s has different backend (%s) than the first leaf (%s)\n",
                    split->name, split->graph->leafs[j]->name,
                    ggml_backend_name(split->graph->leafs[j]->backend_s),
                    ggml_backend_name(split->graph->leafs[0]->backend_s));
            }
        }
    }
}

void ggml_graph_splits_compute(struct ggml_graph_splits * splits) {
    uint64_t copy_us = 0;
    uint64_t compute_cpu_us = 0;
    uint64_t compute_gpu_us = 0;
    int n_nodes = 0;
    for (int i = 0; i < splits->n_splits; i++) {
        struct ggml_graph_split * split = &splits->splits[i];

        //printf("computing split %i (%s) on backend %s (%i nodes)\n", i, split->name, ggml_backend_name(split->dst_inputs[0]->backend), split->graph->n_nodes);

        // copy the input tensor to the backend
        uint64_t copy_start_us = ggml_time_us();
        for (int j = 0; split->src_inputs[j] != NULL; j++) {
            //printf("\tcopying tensor %d (%s) (%s -> %s) (%lu bytes)\n", j, split->src_inputs[j]->name, ggml_backend_name(split->src_inputs[j]->backend), ggml_backend_name(split->dst_inputs[j]->backend), ggml_nbytes(split->src_inputs[j]));
            //printf("%p %p\n", split->src_inputs[j], split->dst_inputs[j]);
            ggml_backend_tensor_copy(split->src_inputs[j], split->dst_inputs[j]);
        }
        // ggml_backend_synchronize(split->dst_inputs[0]->backend);
        copy_us += ggml_time_us() - copy_start_us;

#if 0
        char split_filename[GGML_MAX_NAME];
        snprintf(split_filename, GGML_MAX_NAME, "split_%i.dot", i);
        ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif
        uint64_t start = ggml_time_us();
        ggml_backend_graph_compute(split->dst_inputs[0]->backend_s, split->graph);
        //ggml_backend_synchronize(split->dst_inputs[0]->backend);
        uint64_t end = ggml_time_us();
        if (strcmp(ggml_backend_name(split->dst_inputs[0]->backend_s), "CPU") == 0) {
            compute_cpu_us += end - start;
        } else {
            compute_gpu_us += end - start;
        }

        n_nodes += split->graph->n_nodes;
    }

    //printf("ggml_graph_splits_compute: n_splits: %d, nodes: %d, copy: %.2fms, compute_cpu: %.2fms, compute_gpu: %.2fms\n", splits->n_splits, n_nodes, copy_us / 1000.0, compute_cpu_us / 1000.0, compute_gpu_us / 1000.0);
    //exit(0);
}

void ggml_graph_splits_allocate_tensors(struct ggml_graph_splits * splits) {
    // splits of the same backend are allocated together to ensure that dependencies from one split to the next
    // are not overwritten when there is another split from a different backend between them (e.g. inpSA in llama.cpp)
    bool visited[GGML_MAX_SPLITS] = {false};
    for (int i = 0; i < splits->n_splits; i++) {
        if (!visited[i]) {
            struct ggml_graph_split * split = &splits->splits[i];
            struct ggml_context * ctx = split->ctx;
            struct ggml_cgraph * backend_graphs[GGML_MAX_SPLITS];
            struct ggml_tensor ** graph_inputs[GGML_MAX_SPLITS];
            struct ggml_tensor ** graph_outputs[GGML_MAX_SPLITS];
            int n_graphs = 0;

            for (int j = i; j < splits->n_splits; j++) {
                if (splits->splits[j].ctx == ctx) {
                    graph_inputs[n_graphs] = splits->splits[j].dst_inputs;
                    graph_outputs[n_graphs] = j < splits->n_splits - 1 ? splits->splits[j + 1].src_inputs : NULL;
                    backend_graphs[n_graphs] = splits->splits[j].graph;
                    visited[j] = true;
                    n_graphs++;
                }
            }

            struct ggml_allocr * alloc = NULL;
            ggml_allocr_alloc_graph_n(alloc, backend_graphs, n_graphs, graph_inputs, graph_outputs);
        }
    }
}
#endif
