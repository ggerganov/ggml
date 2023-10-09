#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED GGML_UNUSED

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
        struct ggml_backend                  * backend,
        struct ggml_backend_buffer_i           iface,
               ggml_backend_buffer_context_t   context,
               size_t                          size) {
    ggml_backend_buffer_t buffer = malloc(sizeof(struct ggml_backend_buffer));

    GGML_ASSERT(iface.get_base != NULL);

    (*buffer) = (struct ggml_backend_buffer) {
        /* .interface = */ iface,
        /* .backend   = */ backend,
        /* .context   = */ context,
        /* .size      = */ size,
    };

    return buffer;
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_get_alignment(buffer->backend);
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buffer->iface.get_alloc_size) {
        return buffer->iface.get_alloc_size(buffer, tensor);
    }
    return ggml_nbytes(tensor);
}

void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

void ggml_backend_buffer_free_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // free_tensor is optional
    if (buffer->iface.free_tensor) {
        buffer->iface.free_tensor(buffer, tensor);
    }
}

// backend

ggml_backend_t ggml_get_backend(const struct ggml_tensor * tensor) {
    return tensor->buffer ? tensor->buffer->backend : NULL;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return backend->iface.alloc_buffer(backend, size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return backend->iface.get_alignment(backend);
}

void ggml_backend_tensor_set_async(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_get_backend(tensor)->iface.set_tensor_async(ggml_get_backend(tensor), tensor, data, offset, size);
}

void ggml_backend_tensor_get_async(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_get_backend(tensor)->iface.get_tensor_async(ggml_get_backend(tensor), tensor, data, offset, size);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_t backend = ggml_get_backend(tensor);

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(backend != NULL && "tensor backend not set");

    backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    backend->iface.synchronize(backend);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_t backend = ggml_get_backend(tensor);

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(backend != NULL && "tensor backend not set");

    backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    backend->iface.synchronize(backend);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_free(backend, plan);
}

void ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_compute(backend, plan);
}

void ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
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

    // fprintf(stderr, "cpy tensor %s from %s to %s (%lu bytes)\n", src->name, ggml_backend_name(src->backend), ggml_backend_name(dst->backend), ggml_nbytes(src));

    if (src == dst) {
        return;
    }

    // TODO: allow backends to support copy to/from same backend

    if (ggml_get_backend(dst)->iface.cpy_tensor_from != NULL) {
        ggml_get_backend(dst)->iface.cpy_tensor_from(ggml_get_backend(dst)->context, src, dst);
    } else if (ggml_get_backend(src)->iface.cpy_tensor_to != NULL) {
        ggml_get_backend(src)->iface.cpy_tensor_to(ggml_get_backend(src)->context, src, dst);
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
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: maybe use GGML_ALIGNED_MALLOC?

    GGML_ASSERT(data != NULL && "failed to allocate buffer");

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
    ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));

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

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_cpu_name;
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(ggml_backend_t backend_cpu, void * ptr, size_t size) {
    return ggml_backend_buffer_init(backend_cpu, cpu_backend_buffer_i_from_ptr, ptr, size);
}

// scheduler

#define GGML_MAX_BACKENDS 4
#define GGML_MAX_SPLITS 64
#define GGML_MAX_SPLIT_INPUTS 16

struct ggml_backend_sched_split {
    ggml_allocr_t allocr;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_MAX_SPLIT_INPUTS];
    int n_inputs;
    struct ggml_cgraph * graph;
};

struct ggml_backend_sched {
    int n_backends;
    ggml_backend_t backends[GGML_MAX_BACKENDS];
    ggml_allocr_t  allocs[GGML_MAX_BACKENDS];

    const struct ggml_tensor *  hash_keys[GGML_GRAPH_HASHTABLE_SIZE];
    ggml_allocr_t               node_allocr[GGML_GRAPH_HASHTABLE_SIZE];
    struct ggml_tensor *        node_copies[GGML_GRAPH_HASHTABLE_SIZE][GGML_MAX_BACKENDS];

    struct ggml_cgraph * graph;
    struct ggml_backend_sched_split splits[GGML_MAX_SPLITS];
    int n_splits;

    struct ggml_context * ctx;

    // align context_buffer to GGML_MEM_ALIGN
    //char padding[0];
    // FIXME: this require too much memory due to the size of the graph, avoid duplicating the graphs, use node ranges instead
    char context_buffer[GGML_MAX_SPLITS*GGML_MAX_SPLIT_INPUTS*sizeof(struct ggml_tensor) + GGML_MAX_SPLITS*sizeof(struct ggml_cgraph)];
};

#define hash_id(node) ggml_hash_find_or_insert(sched->hash_keys, node)
#define node_allocr(node) sched->node_allocr[hash_id(node)]

static bool ggml_is_view_op(enum ggml_op op) {
    return false;
    //return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// returns the priority of the backend, lower is better
static int sched_backend_prio(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return INT_MAX;
}

static int sched_allocr_prio(ggml_backend_sched_t sched, ggml_allocr_t allocr) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->allocs[i] == allocr) {
            return i;
        }
    }
    return INT_MAX;
}

// returns the backend that should be used for the node based on the current locations
char causes[GGML_GRAPH_HASHTABLE_SIZE][128];
static ggml_backend_t sched_backend_from_cur(ggml_backend_sched_t sched, const struct ggml_tensor * node) {
    // if the dst tensor is already allocated in a buffer, we must assume that it is critical to keep it there
    // ie. kv cache updates
    // dst
    ggml_backend_t cur_backend = ggml_get_backend(node);
    if (cur_backend != NULL) {
        sprintf(causes[hash_id(node)], "1.dst");
        return cur_backend;
    }

    // view_src
    if (node->view_src != NULL && ggml_get_backend(node->view_src) != NULL) {
        sprintf(causes[hash_id(node)], "1.vsrc");
        return ggml_get_backend(node->view_src);
    }

    // src
    int cur_prio = INT_MAX;
    size_t cur_size = 0;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = node->src[i];
        if (src == NULL) {
            break;
        }
        ggml_backend_t src_backend = ggml_get_backend(src);
        if (src_backend != NULL) {
            int src_prio = sched_backend_prio(sched, src_backend);
            size_t src_size = ggml_nbytes(src);
            if (src_prio < cur_prio && src_size >= cur_size) {
                cur_prio = src_prio;
                cur_size = src_size;
                cur_backend = src_backend;
                sprintf(causes[hash_id(node)], "1.src%d", i);
            }
        }
    }
    return cur_backend;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        sprintf(buffer, "%zuM", size/1024/1024);
    } else {
        sprintf(buffer, "%zuK", size/1024);
    }
    return buffer;
}

static void sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = ggml_allocr_get_buffer(sched->splits[cur_split].allocr)->backend;
            fprintf(stderr, "\n## SPLIT #%d: %s # %d inputs: ", cur_split, ggml_backend_name(split_backend), sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                fprintf(stderr, "[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name, fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            fprintf(stderr, "\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue; // views are removed from the final graphs
        }
        ggml_allocr_t node_allocr = node_allocr(node);
        ggml_backend_t node_backend = node_allocr ? ggml_allocr_get_buffer(node_allocr)->backend : NULL;
        fprintf(stderr, "node #%3d (%10.10s): %20.20s (%4.4s) [%4.4s %8.8s]:", i, ggml_op_name(node->op), node->name, fmt_size(ggml_nbytes(node)), node_allocr ? ggml_backend_name(node_backend) : "NULL", causes[hash_id(node)]);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_allocr_t src_allocr = node_allocr(src);
            ggml_backend_t src_backend = src_allocr ? ggml_allocr_get_buffer(src_allocr)->backend : NULL;
            fprintf(stderr, " %20.20s (%4.4s) [%4.4s %8.8s]", src->name, fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", causes[hash_id(src)]);
        }
        fprintf(stderr, "\n");
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
// TODO: merge passes
static void sched_split_graph(ggml_backend_sched_t sched) {
    struct ggml_cgraph * graph = sched->graph;

    // reset state
    memset(sched->hash_keys, 0, sizeof(sched->hash_keys));
    memset(sched->node_allocr, 0, sizeof(sched->node_allocr));
    memset(sched->node_copies, 0, sizeof(sched->node_copies));
    sched->n_splits = 0;

    struct ggml_init_params params = {
        /*.mem_size =   */ sizeof(sched->context_buffer),
        /*.mem_buffer = */ sched->context_buffer,
        /*.no_alloc =   */ true
    };

    if (sched->ctx != NULL) {
        ggml_free(sched->ctx);
    }

    sched->ctx = ggml_init(params);

    // pass 1: assign backends to ops with allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        const struct ggml_tensor * leaf = graph->leafs[i];
        ggml_backend_t leaf_backend = ggml_get_backend(leaf);
        if (leaf_backend == NULL && leaf->view_src != NULL) {
            leaf_backend = ggml_get_backend(leaf->view_src);
        }
        if (leaf_backend != NULL) {
            node_allocr(leaf) = ggml_backend_sched_get_allocr(sched, leaf_backend);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_backend_t node_backend = sched_backend_from_cur(sched, node);
        if (node_backend != NULL) {
            node_allocr(node) = ggml_backend_sched_get_allocr(sched, node_backend);
        }
    }
    //printf("PASS 1 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);

    // pass 2: assign backends to ops from current assignments
    // TODO:
    //  - reuse sched_backend_from_cur
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_allocr_t node_allocr = node_allocr(node);
        if (node_allocr == NULL) {
            int    cur_prio = INT_MAX;
            size_t cur_size = 0;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                const struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    break;
                }
                ggml_allocr_t src_allocr = node_allocr(src);
                if (src_allocr != NULL) {
                    int    src_prio = sched_allocr_prio(sched, src_allocr);
                    size_t src_size = ggml_nbytes(src);
                    if (src_prio < cur_prio && src_size >= cur_size) {
                        cur_prio = src_prio;
                        cur_size = src_size;
                        node_allocr = src_allocr;
                        sprintf(causes[hash_id(node)], "2.src%d", j);
                    }
                }
            }
            if (node_allocr != NULL) {
                node_allocr(node) = node_allocr;
            }
        }
    }
    //printf("PASS 2 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);

    // pass 3: assign backends to remaining src from dst (should only be leafs)
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_allocr_t node_allocr = node_allocr(node);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_allocr_t src_allocr = node_allocr(src);
            if (src_allocr == NULL) {
                node_allocr(src) = node_allocr;
            }
        }
    }
    //printf("PASS 3 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);

    // pass 4: split graph, find tensors that need to be copied
    // TODO:
    //  - when switching from a less preferred backend to a more preferred backend, check if it is possible to move the switch to an earlier point for the same cost
    // find first backend
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node->view_src == NULL) {
            sched->splits[0].allocr = node_allocr(node);
            break;
        }
    }
    sched->splits[0].i_start = 0;
    sched->splits[0].n_inputs = 0;
    memset(sched->splits[0].inputs, 0, sizeof(sched->splits[0].inputs)); //HACK
    ggml_allocr_t cur_allocr = sched->splits[0].allocr;
    size_t cur_backend_id = sched_allocr_prio(sched, cur_allocr);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        if (ggml_is_view_op(node->op)) {
            continue; // views are removed from the final graphs
        }

        ggml_allocr_t node_allocr = node_allocr(node);

        if (node_allocr != cur_allocr) {
            sched->splits[cur_split].i_end = i;
            cur_split++;
            GGML_ASSERT(cur_split < GGML_MAX_SPLITS);
            sched->splits[cur_split].allocr = node_allocr;
            sched->splits[cur_split].i_start = i;
            sched->splits[cur_split].n_inputs = 0;
            memset(sched->splits[cur_split].inputs, 0, sizeof(sched->splits[cur_split].inputs)); //HACK
            cur_allocr = node_allocr;
            cur_backend_id = sched_allocr_prio(sched, cur_allocr);
        }

        // find inputs that are not on the same backend
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_allocr_t src_allocr = node_allocr(src);
            if (src_allocr != node_allocr) {
                int n_inputs = sched->splits[cur_split].n_inputs++;
                GGML_ASSERT(n_inputs < GGML_MAX_SPLIT_INPUTS);
                sched->splits[cur_split].inputs[n_inputs] = (struct ggml_tensor *)src;

                // create copies
                size_t id = hash_id(src);
                if (sched->node_copies[id][cur_backend_id] == NULL) {
                    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                    sched->node_copies[id][cur_backend_id] = tensor_copy;
                    // FIXME: there is a small chance that this will cause the hash table to overflow
                    // it is not necessary to set the backend of the input copies, but for now it makes debugging easier
                    node_allocr(tensor_copy) = cur_allocr;
                    ggml_backend_t backend = ggml_allocr_get_buffer(cur_allocr)->backend;
                    ggml_format_name(tensor_copy, "%s#%s", ggml_backend_name(backend), src->name);
                }
                node->src[j] = sched->node_copies[id][cur_backend_id];
            }
        }
    }
    sched->splits[cur_split].i_end = graph->n_nodes;
    sched->n_splits = cur_split + 1;

    fprintf(stderr, "PASS 4 ASSIGNMENTS\n"); sched_print_assignments(sched, graph); fflush(stdout);

#if 1
    // sanity check: all sources should have the same backend as the node
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_allocr_t node_allocr = node_allocr(node);
        if (node_allocr == NULL) {
            fprintf(stderr, "!!!!!!! %s has no backend\n", node->name);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_allocr_t src_allocr = node_allocr(src);
            if (src_allocr != node_allocr /* && src_backend != NULL */) { // ignore nulls for now
                fprintf(stderr, "!!!! %s has backend %s, src %d (%s) has backend %s\n",
                    node->name, node_allocr ? ggml_backend_name(ggml_allocr_get_buffer(node_allocr)->backend) : "NULL",
                    j, src->name, src_allocr ? ggml_backend_name(ggml_allocr_get_buffer(src_allocr)->backend) : "NULL");
            }
        }
    }
#endif

    // create copies of the graph for each split
    // FIXME: this is not really necessary, we should use the range of nodes instead
    struct ggml_cgraph * graph_copy = ggml_new_graph(sched->ctx);
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_new_graph(sched->ctx);

        // add inputs to the graph copy
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            struct ggml_tensor * input_cpy = sched->node_copies[hash_id(input)][sched_allocr_prio(sched, split->allocr)];
            input_cpy->src[0] = input;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            struct ggml_tensor * node = graph->nodes[j];
            if (ggml_is_view_op(node->op)) {
                continue; // views are removed from the final graphs
            }
            split->graph->nodes[split->graph->n_nodes++] = node;
            graph_copy->nodes[graph_copy->n_nodes++] = node;
        }
    }
    sched->graph = graph_copy;
}

static void sched_alloc_splits(ggml_backend_sched_t sched) {
    ggml_allocr_alloc_graph_n(
        sched->graph,
        sched->hash_keys,
        sched->node_allocr);
}

static void sched_compute_splits(ggml_backend_sched_t sched) {
    uint64_t copy_us = 0;
    uint64_t compute_cpu_us = 0;
    uint64_t compute_gpu_us = 0;
    int n_nodes = 0;
    struct ggml_backend_sched_split * splits = sched->splits;
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &splits[i];
        ggml_backend_t split_backend = ggml_allocr_get_buffer(split->allocr)->backend;

        //printf("\ncomputing split %i on backend %s (%i nodes) (%i inputs)\n", i, ggml_backend_name(split->backend), split->graph->n_nodes, split->n_inputs);

        // copy the input tensor to the backend
        uint64_t copy_start_us = ggml_time_us();
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input_cpy = sched->node_copies[hash_id(split->inputs[j])][sched_backend_prio(sched, split_backend)];
            //printf("\ninput %d/%d: %s\n", j + 1, split->n_inputs, split->inputs[j]->name);
            //printf("buffers: %p %p\n", split->inputs[j]->buffer, input_cpy->buffer);
            if (split->inputs[j]->buffer == NULL) {
                if (split->inputs[j]->view_src == NULL) {
                    fprintf(stderr, "input %s has no buffer and no view_src\n", split->inputs[j]->name);
                    exit(1);
                }
                struct ggml_tensor * view = split->inputs[j];
                view->backend = view->view_src->backend;
                view->buffer  = view->view_src->buffer;
                view->data    = (char *)view->view_src->data + view->view_offs;
                ggml_backend_buffer_init_tensor(ggml_backend_sched_get_buffer(sched, view->buffer->backend), view);
            }
            if (input_cpy->buffer == NULL) {
                fprintf(stderr, "input_cpy %s has no buffer\n", input_cpy->name);
                exit(1);
            }
            GGML_ASSERT(split->inputs[j]->buffer->backend != input_cpy->buffer->backend);
            GGML_ASSERT(input_cpy->buffer->backend == split_backend);
            //printf("\tcopying tensor %d (%s) (%s -> %s) (%lu bytes)\n", j, split->inputs[j]->name,
            //    ggml_backend_name(split->inputs[j]->buffer->backend), ggml_backend_name(input_cpy->buffer->backend),
            //    ggml_nbytes(split->inputs[j]));
            fflush(stdout);
            ggml_backend_tensor_copy(split->inputs[j], input_cpy);
        }
        // ggml_backend_synchronize(split->dst_inputs[0]->backend);
        copy_us += ggml_time_us() - copy_start_us;

#if 0
        char split_filename[GGML_MAX_NAME];
        snprintf(split_filename, GGML_MAX_NAME, "split_%i.dot", i);
        ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif
        uint64_t start = ggml_time_us();
        ggml_backend_graph_compute(split_backend, split->graph);
        ggml_backend_synchronize(split_backend);
        uint64_t end = ggml_time_us();
        if (strcmp(ggml_backend_name(split_backend), "CPU") == 0) {
            compute_cpu_us += end - start;
        } else {
            compute_gpu_us += end - start;
        }

        n_nodes += split->graph->n_nodes;
    }

    //printf("ggml_graph_splits_compute: n_splits: %d, nodes: %d, copy: %.2fms, compute_cpu: %.2fms, compute_gpu: %.2fms\n", sched->n_splits, n_nodes, copy_us / 1000.0, compute_cpu_us / 1000.0, compute_gpu_us / 1000.0);
    //exit(0);
}

static void sched_reset(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_allocr_reset(sched->allocs[i]);
    }
}

ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, int n_backends) {
    GGML_ASSERT(n_backends <= GGML_MAX_BACKENDS);

    struct ggml_backend_sched * sched = malloc(sizeof(struct ggml_backend_sched));

    fprintf(stderr, "ggml_backend_sched size: %lu KB\n", sizeof(struct ggml_backend_sched)/1024);

    sched->n_backends = n_backends;
    for (int i = 0; i < n_backends; i++) {
        sched->backends[i] = backends[i];
    }

    // init measure allocs for each backend
    for (int i = 0; i < n_backends; i++) {
        sched->allocs[i] = ggml_allocr_new_measure_from_backend(backends[i]);
    }

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_allocr_free(sched->allocs[i]);
    }
    free(sched);
}

void ggml_backend_sched_init_measure(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    sched->graph = measure_graph;
    sched_split_graph(sched);
    sched_alloc_splits(sched);

    // allocate buffers and reset allocators
    for (int i = 0; i < sched->n_backends; i++) {
        size_t size = ggml_allocr_max_size(sched->allocs[i]);
        ggml_allocr_free(sched->allocs[i]);
        sched->allocs[i] = ggml_allocr_new_from_backend(sched->backends[i], size);
    }

    sched_reset(sched);
}

void ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    sched->graph = graph;
    sched_split_graph(sched);
    sched_alloc_splits(sched);
    sched_compute_splits(sched);
    sched_reset(sched);
}

ggml_allocr_t ggml_backend_sched_get_allocr(ggml_backend_sched_t sched, ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    return sched->allocs[backend_index];
}

ggml_backend_buffer_t ggml_backend_sched_get_buffer(ggml_backend_sched_t sched, ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    return ggml_allocr_get_buffer(sched->allocs[backend_index]);
}
