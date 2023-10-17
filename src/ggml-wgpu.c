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

struct ggml_wgpu_buffer {
    const char * name;

    void   * data;
    size_t   size;

    WGPUBuffer wgpu;
};


struct ggml_wgpu_context {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUSupportedLimits limits;
    WGPUQueue queue;
    WGPUShaderModule shader_module;
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;

    WGPUBuffer tensor_dimension_params;
    int64_t tensor_dimension_params_host[24];

    WGPUBindGroupEntry bind_group_entries[4];

    int n_buffers;
    struct ggml_wgpu_buffer buffers[GGML_WGPU_MAX_BUFFERS];

    // custom kernels
#define GGML_WGPU_DECL_KERNEL(name) \
    WGPUComputePipeline pipeline_##name

    GGML_WGPU_DECL_KERNEL(silu);

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



struct ggml_wgpu_context * ggml_wgpu_init() {
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

    ASSERT_CHECK(wgpuDeviceGetLimits(ctx->device, &(ctx->limits)));

    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    ASSERT_CHECK(ctx->queue);

    ctx->tensor_dimension_params = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                            .label = "tensor_dimension_params",
                                                            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                                                            .size = 192,
                                                            .mappedAtCreation = false,
                                                         });
    ASSERT_CHECK(ctx->tensor_dimension_params);

    ctx->bind_group_entries[3].binding = 3;
    ctx->bind_group_entries[3].buffer = ctx->tensor_dimension_params;
    ctx->bind_group_entries[3].offset = 0;
    ctx->bind_group_entries[3].size = 192;



    ctx->n_buffers = 0;

    // load library
    ctx->shader_module = frmwrk_load_shader_module(ctx->device, "ggml-wgpu.wgsl");
    ASSERT_CHECK(ctx->shader_module);

    WGPUBindGroupLayoutEntry bindGroupLayoutEntries[4];
    {
        bindGroupLayoutEntries[0].binding = 0;
        bindGroupLayoutEntries[0].visibility = WGPUShaderStage_Compute;
        bindGroupLayoutEntries[0].buffer.type = WGPUBufferBindingType_Storage;
        bindGroupLayoutEntries[0].buffer.hasDynamicOffset = false;
        bindGroupLayoutEntries[0].buffer.minBindingSize = 0;

        bindGroupLayoutEntries[1].binding = 1;
        bindGroupLayoutEntries[1].visibility = WGPUShaderStage_Compute;
        bindGroupLayoutEntries[1].buffer.type = WGPUBufferBindingType_Storage;
        bindGroupLayoutEntries[1].buffer.hasDynamicOffset = false;
        bindGroupLayoutEntries[1].buffer.minBindingSize = 0;

        bindGroupLayoutEntries[2].binding = 2;
        bindGroupLayoutEntries[2].visibility = WGPUShaderStage_Compute;
        bindGroupLayoutEntries[2].buffer.type = WGPUBufferBindingType_Storage;
        bindGroupLayoutEntries[2].buffer.hasDynamicOffset = false;
        bindGroupLayoutEntries[2].buffer.minBindingSize = 0;

        bindGroupLayoutEntries[3].binding = 3;
        bindGroupLayoutEntries[3].visibility = WGPUShaderStage_Compute;
        bindGroupLayoutEntries[3].buffer.type = WGPUBufferBindingType_Uniform;
        bindGroupLayoutEntries[3].buffer.hasDynamicOffset = false;
        bindGroupLayoutEntries[3].buffer.minBindingSize = 192;
    }


    ctx->bind_group_layout = wgpuDeviceCreateBindGroupLayout(ctx->device, &(const WGPUBindGroupLayoutDescriptor){
                           .label = "ggml-wgpu-bind-group-layout",
                           .entries = bindGroupLayoutEntries,
                           .entryCount = 4,
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


        GGML_WGPU_ADD_KERNEL(silu);

#undef GGML_WGPU_ADD_KERNEL
    }

    return ctx;
}

void ggml_wgpu_free(struct ggml_wgpu_context * ctx) {
    GGML_WGPU_LOG_INFO("%s: deallocating\n", __func__);
#if 0
#define GGML_WGPU_DEL_KERNEL(name) \
    [ctx->function_##name release]; \
    [ctx->pipeline_##name release];

    GGML_WGPU_DEL_KERNEL(silu);

#undef GGML_WGPU_DEL_KERNEL

    for (int i = 0; i < ctx->n_buffers; ++i) {
        [ctx->buffers[i].wgpu release];
    }

    [ctx->library release];
    [ctx->queue release];
    [ctx->device release];

    dispatch_release(ctx->d_queue);

    free(ctx);
#endif
}


void * ggml_wgpu_host_malloc(size_t n) {
    void * data = NULL;
    const int result = posix_memalign((void **) &data, 4, n);
    if (result != 0) {
        GGML_WGPU_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }

    return data;
}

void ggml_wgpu_host_free(void * data) {
    free(data);
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

bool ggml_wgpu_add_buffer(
        struct ggml_wgpu_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= GGML_WGPU_MAX_BUFFERS) {
        GGML_WGPU_LOG_ERROR("%s: error: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                GGML_WGPU_LOG_ERROR("%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = 4; // TODO: figure out if this needs a real value like on metal // sysconf(_SC_PAGESIZE);

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->limits.limits.maxBufferSize) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            // TODO: proper buffer label
            ctx->buffers[ctx->n_buffers].wgpu = wgpuDeviceCreateBuffer(
                                                    ctx->device, &(const WGPUBufferDescriptor){
                                                                .label = "storage_buffer",
                                                                .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                                                        WGPUBufferUsage_CopySrc,
                                                                .size = size_aligned,
                                                                .mappedAtCreation = true,
                                                });

            if (ctx->buffers[ctx->n_buffers].wgpu == NULL) {
                GGML_WGPU_LOG_ERROR("%s: error: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }
            void *buf = wgpuBufferGetMappedRange(ctx->buffers[ctx->n_buffers].wgpu, 0, size);
            memcpy(buf, data, size);
            wgpuBufferUnmap(ctx->buffers[ctx->n_buffers].wgpu);

            GGML_WGPU_LOG_INFO("%s: allocated '%-16s' buffer, size = %8.2f MB", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->limits.limits.maxBufferSize - size_ovlp;
            const size_t size_view = ctx->limits.limits.maxBufferSize;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                // TODO: proper buffer label
                ctx->buffers[ctx->n_buffers].wgpu = wgpuDeviceCreateBuffer(
                                                        ctx->device, &(const WGPUBufferDescriptor){
                                                                    .label = "storage_buffer",
                                                                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                                                            WGPUBufferUsage_CopySrc,
                                                                    .size = size_step_aligned,
                                                                    .mappedAtCreation = true,
                                                    });



                if (ctx->buffers[ctx->n_buffers].wgpu == NULL) {
                    GGML_WGPU_LOG_ERROR("%s: error: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }
                void *buf = wgpuBufferGetMappedRange(ctx->buffers[ctx->n_buffers].wgpu, 0, size_step_aligned);
                memcpy(buf, (void *) ((uint8_t *) data + i), size_step_aligned); // TODO: might be copying bytes out of range if the original alignment is not at 4bytes
                wgpuBufferUnmap(ctx->buffers[ctx->n_buffers].wgpu);

                GGML_WGPU_LOG_INFO("%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    GGML_WGPU_LOG_INFO("\n");
                }

                ++ctx->n_buffers;
            }
        }

        // GGML_WGPU_LOG_INFO(", (%8.2f)\n", ctx->device.currentAllocatedSize / 1024.0 / 1024.0);
    }

    return true;
}

void ggml_wgpu_set_tensor(
        struct ggml_wgpu_context * ctx,
        struct ggml_tensor * t) {
    size_t offs;
    WGPUBuffer id_dst = ggml_wgpu_get_buffer(ctx, t, &offs);

    GGML_ASSERT(ggml_is_contiguous(t));
    wgpuQueueWriteBuffer(ctx->queue, id_dst, offs, t->data, ggml_nbytes(t));
}

void ggml_wgpu_get_tensor(
        struct ggml_wgpu_context * ctx,
        struct ggml_tensor * t) {
    GGML_ASSERT(ggml_is_contiguous(t));
    size_t offs;
    WGPUBuffer id_src = ggml_wgpu_get_buffer(ctx, t, &offs);

    const size_t nbytes = ggml_nbytes(t);

    wgpuBufferMapAsync(id_src, WGPUMapMode_Read, offs, nbytes,
                     handle_buffer_map, NULL);
    wgpuDevicePoll(ctx->device, true, NULL);

    void * buf = wgpuBufferGetMappedRange(id_src, offs, nbytes);
    GGML_ASSERT(buf);

    memcpy(t->data, buf, nbytes);
    wgpuBufferUnmap(id_src);
}

void ggml_wgpu_graph_compute(
        struct ggml_wgpu_context * ctx,
               struct ggml_cgraph * gf) {

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(
            ctx->device, &(const WGPUCommandEncoderDescriptor){
                        .label = "ggml_command_encoder",
                    });
    ASSERT_CHECK(command_encoder);


    for (int i = 0; i < gf->n_nodes; ++i) {
        GGML_WGPU_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

        struct ggml_tensor * src0 = gf->nodes[i]->src[0];
        struct ggml_tensor * src1 = gf->nodes[i]->src[1];
        struct ggml_tensor * dst  = gf->nodes[i];

        ctx->tensor_dimension_params_host[0]  = src0 ? src0->ne[0] : 0;
        ctx->tensor_dimension_params_host[1]  = src0 ? src0->ne[1] : 0;
        ctx->tensor_dimension_params_host[2]  = src0 ? src0->ne[2] : 0;
        ctx->tensor_dimension_params_host[3]  = src0 ? src0->ne[3] : 0;

        ctx->tensor_dimension_params_host[4]  = src0 ? src0->nb[0] : 0;
        ctx->tensor_dimension_params_host[5]  = src0 ? src0->nb[1] : 0;
        ctx->tensor_dimension_params_host[6]  = src0 ? src0->nb[2] : 0;
        ctx->tensor_dimension_params_host[7]  = src0 ? src0->nb[3] : 0;

        ctx->tensor_dimension_params_host[8]  = src1 ? src1->ne[0] : 0;
        ctx->tensor_dimension_params_host[9]  = src1 ? src1->ne[1] : 0;
        ctx->tensor_dimension_params_host[10] = src1 ? src1->ne[2] : 0;
        ctx->tensor_dimension_params_host[11] = src1 ? src1->ne[3] : 0;

        ctx->tensor_dimension_params_host[12] = src1 ? src1->nb[0] : 0;
        ctx->tensor_dimension_params_host[13] = src1 ? src1->nb[1] : 0;
        ctx->tensor_dimension_params_host[14] = src1 ? src1->nb[2] : 0;
        ctx->tensor_dimension_params_host[15] = src1 ? src1->nb[3] : 0;

        ctx->tensor_dimension_params_host[16] = dst ? dst->ne[0] : 0;
        ctx->tensor_dimension_params_host[17] = dst ? dst->ne[1] : 0;
        ctx->tensor_dimension_params_host[18] = dst ? dst->ne[2] : 0;
        ctx->tensor_dimension_params_host[19] = dst ? dst->ne[3] : 0;

        ctx->tensor_dimension_params_host[20] = dst ? dst->nb[0] : 0;
        ctx->tensor_dimension_params_host[21] = dst ? dst->nb[1] : 0;
        ctx->tensor_dimension_params_host[22] = dst ? dst->nb[2] : 0;
        ctx->tensor_dimension_params_host[23] = dst ? dst->nb[3] : 0;

        wgpuQueueWriteBuffer(ctx->queue, ctx->tensor_dimension_params, 0, ctx->tensor_dimension_params_host, 192);

        const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
        const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
        const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

        size_t offs_src0 = 0;
        size_t offs_src1 = 0;
        size_t offs_dst  = 0;
        WGPUBuffer id_src0 = src0 ? ggml_wgpu_get_buffer(ctx, src0, &offs_src0) : NULL;
        WGPUBuffer id_src1 = src1 ? ggml_wgpu_get_buffer(ctx, src1, &offs_src1) : NULL;
        WGPUBuffer id_dst  = dst  ? ggml_wgpu_get_buffer(ctx, dst,  &offs_dst)  : NULL;


        ctx->bind_group_entries[0].binding = 0;
        ctx->bind_group_entries[0].buffer = id_src0;
        ctx->bind_group_entries[0].offset = offs_src0;
        ctx->bind_group_entries[0].size = id_src0 ? (wgpuBufferGetSize(id_src0) - offs_src0) : 0;

        ctx->bind_group_entries[1].binding = 1;
        ctx->bind_group_entries[1].buffer = id_src1;
        ctx->bind_group_entries[1].offset = offs_src1;
        ctx->bind_group_entries[1].size = id_src1 ? (wgpuBufferGetSize(id_src1) - offs_src1) : 0;

        ctx->bind_group_entries[2].binding = 2;
        ctx->bind_group_entries[2].buffer = id_dst;
        ctx->bind_group_entries[2].offset = offs_dst;
        ctx->bind_group_entries[2].size = id_dst ? (wgpuBufferGetSize(id_dst) - offs_dst) : 0;


        WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
            ctx->device, &(const WGPUBindGroupDescriptor){
                        .label = "bind_group",
                        .layout = ctx->bind_group_layout,
                        .entryCount = 4,
                        .entries = ctx->bind_group_entries,
                    });
        ASSERT_CHECK(bind_group);

        WGPUComputePassEncoder compute_pass_encoder;


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
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(gf->nodes[i])) {
                    case GGML_UNARY_OP_SILU:
                        {
                            compute_pass_encoder = wgpuCommandEncoderBeginComputePass(
                                command_encoder, &(const WGPUComputePassDescriptor){
                                                    .label = "compute_pass",
                                                });
                            ASSERT_CHECK(compute_pass_encoder);

                            wgpuComputePassEncoderSetPipeline(compute_pass_encoder, ctx->pipeline_silu);
                            wgpuComputePassEncoderSetBindGroup(compute_pass_encoder, 0, bind_group, 0, NULL);
                            wgpuComputePassEncoderDispatchWorkgroups(compute_pass_encoder, ggml_nelements(dst), 1, 1);
                            wgpuComputePassEncoderEnd(compute_pass_encoder);
                        } break;
                    default:
                        {
                            GGML_WGPU_LOG_WARN("%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                            GGML_ASSERT(false);
                        }
                } break;
            default:
                {
                    GGML_WGPU_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                    GGML_ASSERT(false);
                }
        }

        if (bind_group) wgpuBindGroupRelease(bind_group);
        if (compute_pass_encoder) wgpuComputePassEncoderRelease(compute_pass_encoder);
    }

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
                                        command_encoder, &(const WGPUCommandBufferDescriptor){
                                                            .label = "command_buffer",
                       });
    ASSERT_CHECK(command_buffer);

    wgpuQueueSubmit(ctx->queue, 1, &command_buffer);

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

