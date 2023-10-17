#include "ggml-wgpu.h"
#include "ggml.h"

#include <webgpu/webgpu.h>
#include "framework.h"

#include <stdarg.h>


#define ASSERT_CHECK(x) \
    if (!(x)) { \
        GGML_WGPU_LOG_ERROR("%s: error: assertion failed: %s\n", __func__, #x); \
        return NULL; \
    }

#define LOG_PREFIX "[compute]"
static void handle_request_adapter(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter, char const *message,
                                   void *userdata) {
  UNUSED(status)
  UNUSED(message)
  *(WGPUAdapter *)userdata = adapter;
}
static void handle_request_device(WGPURequestDeviceStatus status,
                                  WGPUDevice device, char const *message,
                                  void *userdata) {
  UNUSED(status)
  UNUSED(message)
  *(WGPUDevice *)userdata = device;
}
static void handle_buffer_map(WGPUBufferMapAsyncStatus status, void *userdata) {
  UNUSED(userdata)
  printf(LOG_PREFIX " buffer_map status=%#.8x\n", status);
}


#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef GGML_WGPU_NDEBUG
#define GGML_WGPU_LOG_INFO(...)
#define GGML_WGPU_LOG_WARN(...)
#define GGML_WGPU_LOG_ERROR(...)
#else
#define GGML_WGPU_LOG_INFO(...)  ggml_wgpu_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_WGPU_LOG_WARN(...)  ggml_wgpu_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_WGPU_LOG_ERROR(...) ggml_wgpu_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#endif

#define UNUSED(x) (void)(x)

#define GGML_MAX_CONCUR (2*GGML_MAX_NODES)

struct ggml_wgpu_buffer {
    const char * name;

    void   * data;
    size_t   size;

    WGPUBuffer wgpu;
};


struct ggml_wgpu_context {
    int n_cb;

    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUShaderModule shader_module;
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;

    // id<MTLCommandBuffer>         command_buffers [GGML_WGPU_MAX_COMMAND_BUFFERS];
    // id<MTLComputeCommandEncoder> command_encoders[GGML_WGPU_MAX_COMMAND_BUFFERS];

    // dispatch_queue_t d_queue;

    int n_buffers;
    struct ggml_wgpu_buffer buffers[GGML_WGPU_MAX_BUFFERS];

    int concur_list[GGML_MAX_CONCUR];
    int concur_list_len;

    // custom kernels
#define GGML_WGPU_DECL_KERNEL(name) \
    WGPUComputePipeline pipeline_##name

    GGML_WGPU_DECL_KERNEL(add);
    GGML_WGPU_DECL_KERNEL(scale);
    GGML_WGPU_DECL_KERNEL(silu);
    GGML_WGPU_DECL_KERNEL(relu);
    GGML_WGPU_DECL_KERNEL(gelu);
    GGML_WGPU_DECL_KERNEL(soft_max);
    GGML_WGPU_DECL_KERNEL(sqr);

#undef GGML_WGPU_DECL_KERNEL
};


ggml_log_callback ggml_wgpu_log_callback = NULL;
void * ggml_wgpu_log_user_data = NULL;

void ggml_wgpu_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_wgpu_log_callback  = log_callback;
    ggml_wgpu_log_user_data = user_data;
}

static void wgpu_log_callback(WGPULogLevel level, char const *message,
                         void *userdata) {
  UNUSED(userdata);
  char *level_str;
  switch (level) {
  case WGPULogLevel_Error:
    level_str = "error";
    break;
  case WGPULogLevel_Warn:
    level_str = "warn";
    break;
  case WGPULogLevel_Info:
    level_str = "info";
    break;
  case WGPULogLevel_Debug:
    level_str = "debug";
    break;
  case WGPULogLevel_Trace:
    level_str = "trace";
    break;
  default:
    level_str = "unknown_level";
  }
  fprintf(stderr, "[wgpu] [%s] %s\n", level_str, message);
}

static void ggml_wgpu_log(enum ggml_log_level level, const char* format, ...){
    if (ggml_wgpu_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_wgpu_log_callback(level, buffer, ggml_wgpu_log_user_data);
        } else {
            char* buffer2 = malloc(len+1);
            vsnprintf(buffer2, len+1, format, args);
            buffer2[len] = 0;
            ggml_wgpu_log_callback(level, buffer2, ggml_wgpu_log_user_data);
            free(buffer2);
        }
        va_end(args);
    }
}



struct ggml_wgpu_context * ggml_wgpu_init(int n_cb) {
    GGML_WGPU_LOG_INFO("%s: allocating\n", __func__);

    wgpuSetLogCallback(wgpu_log_callback, NULL);
    wgpuSetLogLevel(WGPULogLevel_Info);


    // Configure context
    struct ggml_wgpu_context * ctx = malloc(sizeof(struct ggml_wgpu_context));

    ctx->instance = wgpuCreateInstance(NULL);
    ASSERT_CHECK(ctx->instance);

    wgpuInstanceRequestAdapter(ctx->instance, NULL, handle_request_adapter,
                                (void *)&(ctx->adapter));
    ASSERT_CHECK(ctx->adapter);

    wgpuAdapterRequestDevice(ctx->adapter, NULL, handle_request_device,
                            (void *)&(ctx->device));
    ASSERT_CHECK(ctx->device);

    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    ASSERT_CHECK(ctx->queue);

    ctx->n_cb   = MIN(n_cb, GGML_WGPU_MAX_BUFFERS);
    ctx->n_buffers = 0;
    ctx->concur_list_len = 0;

    // load library
    ctx->shader_module = frmwrk_load_shader_module(ctx->device, "ggml-wgpu.wgsl");
    ASSERT_CHECK(ctx->shader_module);


    ctx->bind_group_layout = wgpuDeviceCreateBindGroupLayout(ctx->device, &(const WGPUBindGroupLayoutDescriptor){
                           .label = "ggml-wgpu-bind-group-layout",
                           .entryCount = 0,
                       });
    ASSERT_CHECK(ctx->bind_group_layout);

    ctx->pipeline_layout = wgpuDeviceCreatePipelineLayout(ctx->device, &(const WGPUPipelineLayoutDescriptor){
                           .label = "ggml-wgpu-pipeline-layout",
                           .bindGroupLayoutCount = 1,
                           .bindGroupLayouts = &(ctx->bind_group_layout),
                       });
    ASSERT_CHECK(ctx->pipeline_layout);




    // load kernels
    {
#define GGML_WGPU_ADD_KERNEL(name) \
        ctx->pipeline_##name = wgpuDeviceCreateComputePipeline(     \
            ctx->device, &(const WGPUComputePipelineDescriptor){    \
                        .label = "compute_pipeline_##name",         \
                        .compute =                                  \
                            (const WGPUProgrammableStageDescriptor){\
                                .module = ctx->shader_module,       \
                                .entryPoint = "kernel_##name",      \
                            },                                      \
                    });                                             \
        ASSERT_CHECK(ctx->pipeline_##name);


        GGML_WGPU_ADD_KERNEL(add);
        GGML_WGPU_ADD_KERNEL(scale);
        GGML_WGPU_ADD_KERNEL(silu);
        GGML_WGPU_ADD_KERNEL(relu);
        GGML_WGPU_ADD_KERNEL(gelu);
        GGML_WGPU_ADD_KERNEL(soft_max);
        GGML_WGPU_ADD_KERNEL(sqr);

#undef GGML_WGPU_ADD_KERNEL
    }

    return ctx;
}
#if 0

void ggml_metal_free(struct ggml_metal_context * ctx) {
    GGML_METAL_LOG_INFO("%s: deallocating\n", __func__);
#define GGML_METAL_DEL_KERNEL(name) \
    [ctx->function_##name release]; \
    [ctx->pipeline_##name release];

    GGML_METAL_DEL_KERNEL(add);
    GGML_METAL_DEL_KERNEL(scale);
    GGML_METAL_DEL_KERNEL(silu);
    GGML_METAL_DEL_KERNEL(relu);
    GGML_METAL_DEL_KERNEL(gelu);
    GGML_METAL_DEL_KERNEL(soft_max);
    GGML_METAL_DEL_KERNEL(sqr);

#undef GGML_METAL_DEL_KERNEL

    for (int i = 0; i < ctx->n_buffers; ++i) {
        [ctx->buffers[i].metal release];
    }

    [ctx->library release];
    [ctx->queue release];
    [ctx->device release];

    dispatch_release(ctx->d_queue);

    free(ctx);
}
#endif


void * ggml_wgpu_host_malloc(struct ggml_wgpu_context * ctx, size_t n) {
    // TODO: proper buffer label
    WGPUBuffer storage_buffer = wgpuDeviceCreateBuffer(
        ctx->device, &(const WGPUBufferDescriptor){
                    .label = "storage_buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                            WGPUBufferUsage_CopySrc,
                    .size = n,
                    .mappedAtCreation = false,
                });

    if (!storage_buffer) {
        GGML_WGPU_LOG_ERROR("%s: error: wgpuDeviceCreateBuffer failed\n", __func__);
        return NULL;
    }

    return storage_buffer;
}

void ggml_wgpu_host_free(struct ggml_wgpu_context * ctx, void * data) {
    WGPUBuffer storage_buffer = (WGPUBuffer)data;
    wgpuBufferRelease(storage_buffer);
    // TODO: figure out different between release/destroy and which one to use
    // wgpuBufferDestroy(storage_buffer)
}

void ggml_wgpu_set_n_cb(struct ggml_wgpu_context * ctx, int n_cb) {
    ctx->n_cb = MIN(n_cb, GGML_WGPU_MAX_BUFFERS);
}

int ggml_wgpu_if_optimized(struct ggml_wgpu_context * ctx) {
    return ctx->concur_list_len;
}

int * ggml_wgpu_get_concur_list(struct ggml_wgpu_context * ctx) {
    return ctx->concur_list;
}


// finds the WebGPU buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// WebGPU buffer based on the host memory pointer
//
static WGPUBuffer ggml_wgpu_get_buffer(struct ggml_wgpu_context * ctx, struct ggml_tensor * t, size_t * offs) {
    //GGML_WGPU_LOG_INFO("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        //GGML_WGPU_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, ctx->buffers[%d].size = %10ld, name = %s\n", ioffs, tsize, ioffs + tsize, i, ctx->buffers[i].size, ctx->buffers[i].name);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //GGML_WGPU_LOG_INFO("%s: '%s' tensor '%16s', offs = %8ld\n", __func__, ctx->buffers[i].name, t->name, *offs);

            return ctx->buffers[i].wgpu;
        }
    }

    GGML_WGPU_LOG_ERROR("%s: error: buffer is null\n", __func__);

    return NULL;
}

#if 0

bool ggml_metal_add_buffer(
        struct ggml_metal_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= GGML_METAL_MAX_BUFFERS) {
        GGML_METAL_LOG_ERROR("%s: error: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                GGML_METAL_LOG_ERROR("%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = sysconf(_SC_PAGESIZE);

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->device.maxBufferLength) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                GGML_METAL_LOG_ERROR("%s: error: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }

            GGML_METAL_LOG_INFO("%s: allocated '%-16s' buffer, size = %8.2f MB", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->device.maxBufferLength - size_ovlp;
            const size_t size_view = ctx->device.maxBufferLength;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    GGML_METAL_LOG_ERROR("%s: error: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }

                GGML_METAL_LOG_INFO("%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    GGML_METAL_LOG_INFO("\n");
                }

                ++ctx->n_buffers;
            }
        }

        GGML_METAL_LOG_INFO(", (%8.2f)\n", ctx->device.currentAllocatedSize / 1024.0 / 1024.0);
    }

    return true;
}

void ggml_metal_set_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    size_t offs;
    id<MTLBuffer> id_dst = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy((void *) ((uint8_t *) id_dst.contents + offs), t->data, ggml_nbytes(t));
}

void ggml_metal_get_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    size_t offs;
    id<MTLBuffer> id_src = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy(t->data, (void *) ((uint8_t *) id_src.contents + offs), ggml_nbytes(t));
}

void ggml_metal_graph_find_concurrency(
        struct ggml_metal_context * ctx,
        struct ggml_cgraph * gf, bool check_mem) {
    int search_depth = gf->n_nodes; //we only find concurrency in this range to avoid wasting too much time
    int nodes_unused[GGML_MAX_CONCUR];

    for (int i = 0; i < GGML_MAX_CONCUR; i++) { ctx->concur_list[i] = 0; }
    for (int i = 0; i < gf->n_nodes;     i++) { nodes_unused[i]     = 1; }
    ctx->concur_list_len = 0;

    int n_left    = gf->n_nodes;
    int n_start   = 0; // all nodes before n_start at nodes_unused array have been sorted and store back to ctx->concur_list
    int level_pos = 0; // at ctx->concur_list, the last layer (level) ends at level_pos

    while (n_left > 0) {
        // number of nodes at a layer (that can be issued concurrently)
        int concurrency = 0;
        for (int i = n_start; i < ((n_start + search_depth > gf->n_nodes) ? gf->n_nodes : n_start + search_depth); i++) {
            if (nodes_unused[i]) {
                // if the requirements for gf->nodes[i] are satisfied
                int exe_flag = 1;

                // scan all srcs
                for (int src_ind = 0; src_ind < GGML_MAX_SRC; src_ind++) {
                    struct ggml_tensor * src_cur = gf->nodes[i]->src[src_ind];
                    if (src_cur) {
                        // if is leaf nodes it's satisfied.
                        // TODO: ggml_is_leaf()
                        if (src_cur->op == GGML_OP_NONE && src_cur->grad == NULL) {
                            continue;
                        }

                        // otherwise this src should be the output from previous nodes.
                        int is_found = 0;

                        // scan 2*search_depth back because we inserted barrier.
                        //for (int j = ((level_pos - 2*search_depth) < 0 ? 0 : (level_pos - 2*search_depth)); j < level_pos; j++) {
                        for (int j = MAX(0, level_pos - 2*search_depth); j < level_pos; j++) {
                            if (ctx->concur_list[j] >= 0 && gf->nodes[ctx->concur_list[j]] == src_cur) {
                                is_found = 1;
                                break;
                            }
                        }
                        if (is_found == 0) {
                            exe_flag = 0;
                            break;
                        }
                    }
                }
                if (exe_flag && check_mem) {
                    // check if nodes[i]'s data will be overwritten by a node before nodes[i].
                    // if node[5] and node[3] write to the same memory region, then we can't issue node[5] before node[3]
                    int64_t data_start = (int64_t) gf->nodes[i]->data;
                    int64_t length     = (int64_t) ggml_nbytes(gf->nodes[i]);
                    for (int j = n_start; j < i; j++) {
                        if (nodes_unused[j] && gf->nodes[j]->op != GGML_OP_RESHAPE \
                                            && gf->nodes[j]->op != GGML_OP_VIEW \
                                            && gf->nodes[j]->op != GGML_OP_TRANSPOSE \
                                            && gf->nodes[j]->op != GGML_OP_PERMUTE) {
                            if (((int64_t)gf->nodes[j]->data) >= data_start + length || \
                                ((int64_t)gf->nodes[j]->data) + (int64_t) ggml_nbytes(gf->nodes[j]) <= data_start) {
                                continue;
                            }

                            exe_flag = 0;
                        }
                    }
                }
                if (exe_flag) {
                    ctx->concur_list[level_pos + concurrency] = i;
                    nodes_unused[i] = 0;
                    concurrency++;
                    ctx->concur_list_len++;
                }
            }
        }
        n_left -= concurrency;
        // adding a barrier different layer
        ctx->concur_list[level_pos + concurrency] = -1;
        ctx->concur_list_len++;
        // jump all sorted nodes at nodes_bak
        while (!nodes_unused[n_start]) {
            n_start++;
        }
        level_pos += concurrency + 1;
    }

    if (ctx->concur_list_len > GGML_MAX_CONCUR) {
        GGML_METAL_LOG_WARN("%s: too many elements for metal ctx->concur_list!\n", __func__);
    }
}

void ggml_metal_graph_compute(
        struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {
    @autoreleasepool {

    // if there is ctx->concur_list, dispatch concurrently
    // else fallback to serial dispatch
    MTLComputePassDescriptor * edesc = MTLComputePassDescriptor.computePassDescriptor;

    const bool has_concur = ctx->concur_list_len && ctx->concur_list_len <= GGML_MAX_CONCUR;

    const int n_nodes  = has_concur ? ctx->concur_list_len      : gf->n_nodes;
    edesc.dispatchType = has_concur ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;

    // create multiple command buffers and enqueue them
    // then, we encode the graph into the command buffers in parallel

    const int n_cb = ctx->n_cb;

    for (int i = 0; i < n_cb; ++i) {
        ctx->command_buffers[i] = [ctx->queue commandBuffer];

        // enqueue the command buffers in order to specify their execution order
        [ctx->command_buffers[i] enqueue];

        ctx->command_encoders[i] = [ctx->command_buffers[i] computeCommandEncoderWithDescriptor: edesc];
    }

    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

        dispatch_async(ctx->d_queue, ^{
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            id<MTLCommandBuffer> command_buffer  = ctx->command_buffers[cb_idx];
            id<MTLComputeCommandEncoder> encoder = ctx->command_encoders[cb_idx];

            const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
            const int node_end   = MIN((cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb, n_nodes);

            for (int ind = node_start; ind < node_end; ++ind) {
                const int i = has_concur ? ctx->concur_list[ind] : ind;

                if (i == -1) {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    continue;
                }

                //GGML_METAL_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

                struct ggml_tensor * src0 = gf->nodes[i]->src[0];
                struct ggml_tensor * src1 = gf->nodes[i]->src[1];
                struct ggml_tensor * dst  = gf->nodes[i];

                const int64_t  ne00 = src0 ? src0->ne[0] : 0;
                const int64_t  ne01 = src0 ? src0->ne[1] : 0;
                const int64_t  ne02 = src0 ? src0->ne[2] : 0;
                const int64_t  ne03 = src0 ? src0->ne[3] : 0;

                const uint64_t nb00 = src0 ? src0->nb[0] : 0;
                const uint64_t nb01 = src0 ? src0->nb[1] : 0;
                const uint64_t nb02 = src0 ? src0->nb[2] : 0;
                const uint64_t nb03 = src0 ? src0->nb[3] : 0;

                const int64_t  ne10 = src1 ? src1->ne[0] : 0;
                const int64_t  ne11 = src1 ? src1->ne[1] : 0;
                const int64_t  ne12 = src1 ? src1->ne[2] : 0;
                const int64_t  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

                const uint64_t nb10 = src1 ? src1->nb[0] : 0;
                const uint64_t nb11 = src1 ? src1->nb[1] : 0;
                const uint64_t nb12 = src1 ? src1->nb[2] : 0;
                const uint64_t nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

                const int64_t  ne0  = dst ? dst->ne[0] : 0;
                const int64_t  ne1  = dst ? dst->ne[1] : 0;
                const int64_t  ne2  = dst ? dst->ne[2] : 0;
                const int64_t  ne3  = dst ? dst->ne[3] : 0;

                const uint64_t nb0  = dst ? dst->nb[0] : 0;
                const uint64_t nb1  = dst ? dst->nb[1] : 0;
                const uint64_t nb2  = dst ? dst->nb[2] : 0;
                const uint64_t nb3  = dst ? dst->nb[3] : 0;

                const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
                const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
                const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

                id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(ctx, src0, &offs_src0) : nil;
                id<MTLBuffer> id_src1 = src1 ? ggml_metal_get_buffer(ctx, src1, &offs_src1) : nil;
                id<MTLBuffer> id_dst  = dst  ? ggml_metal_get_buffer(ctx, dst,  &offs_dst)  : nil;

                //GGML_METAL_LOG_INFO("%s: op - %s\n", __func__, ggml_op_name(dst->op));
                //if (src0) {
                //    GGML_METAL_LOG_INFO("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02,
                //            ggml_is_contiguous(src0), src0->name);
                //}
                //if (src1) {
                //    GGML_METAL_LOG_INFO("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12,
                //            ggml_is_contiguous(src1), src1->name);
                //}
                //if (dst) {
                //    GGML_METAL_LOG_INFO("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, ggml_type_name(dstt),  ne0,  ne1,  ne2,
                //            dst->name);
                //}

                switch (dst->op) {
                    case GGML_OP_NONE:
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_TRANSPOSE:
                    case GGML_OP_PERMUTE:
                        {
                            // noop
                        } break;
                    case GGML_OP_ADD:
                        {
                            GGML_ASSERT(ggml_is_contiguous(src0));
                            GGML_ASSERT(ggml_is_contiguous(src1));

                            bool bcast_row = false;

                            int64_t nb = ne00;

                            if (ggml_nelements(src1) == ne10 && ne00 % 4 == 0) {
                                // src1 is a row
                                GGML_ASSERT(ne11 == 1);

                                nb = ne00 / 4;
                                [encoder setComputePipelineState:ctx->pipeline_add_row];

                                bcast_row = true;
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_add];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                            [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:6];
                            [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:7];
                            [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:8];
                            [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:9];
                            [encoder setBytes:&nb03 length:sizeof(nb03) atIndex:10];
                            [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:11];
                            [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:12];
                            [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:13];
                            [encoder setBytes:&ne13 length:sizeof(ne13) atIndex:14];
                            [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:15];
                            [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:16];
                            [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:17];
                            [encoder setBytes:&nb13 length:sizeof(nb13) atIndex:18];
                            [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:19];
                            [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:20];
                            [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:21];
                            [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:22];
                            [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:23];
                            [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:24];
                            [encoder setBytes:&nb2  length:sizeof(nb2)  atIndex:25];
                            [encoder setBytes:&nb3  length:sizeof(nb3)  atIndex:26];
                            [encoder setBytes:&nb   length:sizeof(nb)   atIndex:27];

                            if (bcast_row) {
                                const int64_t n = ggml_nelements(dst)/4;

                                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            } else {
                                const int nth = MIN(1024, ne0);

                                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                            }
                        } break;
                    case GGML_OP_SCALE:
                        {
                            GGML_ASSERT(ggml_is_contiguous(src0));

                            const float scale = *(const float *) src1->data;

                            [encoder setComputePipelineState:ctx->pipeline_scale];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                            const int64_t n = ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_UNARY:
                        switch (ggml_get_unary_op(gf->nodes[i])) {
                            case GGML_UNARY_OP_SILU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_silu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst)/4;

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_RELU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_relu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_GELU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_gelu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst)/4;

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            default:
                                {
                                    GGML_METAL_LOG_WARN("%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                    GGML_ASSERT(false);
                                }
                        } break;
                    case GGML_OP_SQR:
                        {
                            GGML_ASSERT(ggml_is_contiguous(src0));

                            [encoder setComputePipelineState:ctx->pipeline_sqr];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                            const int64_t n = ggml_nelements(dst);
                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_SOFT_MAX:
                        {
                            const int nth = MIN(32, ne00);

                            if (ne00%4 == 0) {
                                [encoder setComputePipelineState:ctx->pipeline_soft_max_4];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_soft_max];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_DUP:
                    case GGML_OP_CPY:
                    case GGML_OP_CONT:
                        {
                            const int nth = MIN(1024, ne00);

                            switch (src0t) {
                                case GGML_TYPE_F32:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f16]; break;
                                            case GGML_TYPE_F32: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f32]; break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                case GGML_TYPE_F16:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f16_f16]; break;
                                            case GGML_TYPE_F32: GGML_ASSERT(false && "cpy_f16_f32 not implemented"); break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    default:
                        {
                            GGML_METAL_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                            GGML_ASSERT(false);
                        }
                }
            }

            if (encoder != nil) {
                [encoder endEncoding];
                encoder = nil;
            }

            [command_buffer commit];
        });
    }

    // wait for all threads to finish
    dispatch_barrier_sync(ctx->d_queue, ^{});

    // check status of command buffers
    // needed to detect if the device ran out-of-memory for example (#1881)
    for (int i = 0; i < n_cb; i++) {
        [ctx->command_buffers[i] waitUntilCompleted];

        MTLCommandBufferStatus status = (MTLCommandBufferStatus) [ctx->command_buffers[i] status];
        if (status != MTLCommandBufferStatusCompleted) {
            GGML_METAL_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
            GGML_ASSERT(false);
        }
    }

    }
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

static const char * ggml_backend_wgpu_name(ggml_backend_t backend) {
    return "WebGPU";

    UNUSED(backend);
}

static void ggml_backend_wgpu_free(ggml_backend_t backend) {
    struct ggml_wgpu_context * ctx = (struct ggml_wgpu_context *)backend->context;
    ggml_wgpu_free(ctx);
    free(backend);
}

static void * ggml_backend_wgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}

static void ggml_backend_wgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
    UNUSED(buffer);
}

static struct ggml_backend_buffer_i wgpu_backend_buffer_i = {
    /* .free_buffer    = */ ggml_backend_wgpu_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_wgpu_buffer_get_base,
    /* .get_alloc_size = */ NULL, // defaults to ggml_nbytes
    /* .init_tensor    = */ NULL, // no initialization required
    /* .free_tensor    = */ NULL, // no cleanup required
};

static ggml_backend_buffer_t ggml_backend_wgpu_alloc_buffer(ggml_backend_t backend, size_t size) {
    struct ggml_wgpu_context * ctx = (struct ggml_wgpu_context *)backend->context;

    void * data = ggml_wgpu_host_malloc(size);

    // TODO: set proper name of the buffers
    ggml_wgpu_add_buffer(ctx, "backend", data, size, 0);

    return ggml_backend_buffer_init(backend, wgpu_backend_buffer_i, data, size);
}

static size_t ggml_backend_wgpu_get_alignment(ggml_backend_t backend) {
    return 32;
    UNUSED(backend);
}

static void ggml_backend_wgpu_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(backend);
}

static void ggml_backend_wgpu_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy(data, (const char *)tensor->data + offset, size);

    UNUSED(backend);
}

static void ggml_backend_wgpu_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
}

static void ggml_backend_wgpu_cpy_tensor_from(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));

    UNUSED(backend);
}

static void ggml_backend_wgpu_cpy_tensor_to(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_set_async(dst, src->data, 0, ggml_nbytes(src));

    UNUSED(backend);
}

static void ggml_backend_wgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_wgpu_context * wgpu_ctx = (struct ggml_wgpu_context *)backend->context;

    ggml_wgpu_graph_compute(wgpu_ctx, cgraph);
}

static bool ggml_backend_wgpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return true;
    UNUSED(backend);
    UNUSED(op);
}

static struct ggml_backend_i wgpu_backend_i = {
    /* .get_name            = */ ggml_backend_wgpu_name,
    /* .free                = */ ggml_backend_wgpu_free,
    /* .alloc_buffer        = */ ggml_backend_wgpu_alloc_buffer,
    /* .get_alignment       = */ ggml_backend_wgpu_get_alignment,
    /* .set_tensor_async    = */ ggml_backend_wgpu_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_wgpu_get_tensor_async,
    /* .synchronize         = */ ggml_backend_wgpu_synchronize,
    /* .cpy_tensor_from     = */ ggml_backend_wgpu_cpy_tensor_from,
    /* .cpy_tensor_to       = */ ggml_backend_wgpu_cpy_tensor_to,
    /* .graph_plan_create   = */ NULL, // the wgpu implementation does not require creating graph plans atm
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_wgpu_graph_compute,
    /* .supports_op         = */ ggml_backend_wgpu_supports_op,
};

ggml_backend_t ggml_backend_wgpu_init(void) {
    struct ggml_wgpu_context * ctx = malloc(sizeof(struct ggml_wgpu_context));

    ctx = ggml_wgpu_init(GGML_DEFAULT_N_THREADS);

    ggml_backend_t wgpu_backend = malloc(sizeof(struct ggml_backend));

    *wgpu_backend = (struct ggml_backend) {
        /* .interface = */ wgpu_backend_i,
        /* .context   = */ ctx,
    };

    return wgpu_backend;
}

bool ggml_backend_is_wgpu(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_wgpu_name;
}

void ggml_backend_wgpu_set_n_cb(ggml_backend_t backend, int n_cb) {
    struct ggml_wgpu_context * ctx = (struct ggml_wgpu_context *)backend->context;

    ggml_wgpu_set_n_cb(ctx, n_cb);
}
#endif