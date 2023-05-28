#import "main-mtl.h"

#import "ggml/ggml.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

struct ggml_mtl_context {
    struct ggml_context * ctx_data;
    struct ggml_context * ctx_eval;
    struct ggml_context * ctx_work;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLHeap> heap_data;
    id<MTLHeap> heap_eval;

    // custom kernels
    id<MTLFunction>             function_add;
    id<MTLComputePipelineState> pipeline_add;
};

// MSL code
NSString * const msl_library_mnist = @"\
#include <metal_stdlib>                                                                 \n\
using namespace metal;                                                                  \n\
                                                                                        \n\
kernel void kernel_add(                                                                 \n\
        device const float * src0,                                                      \n\
        device const float * src1,                                                      \n\
        device float * dst,                                                             \n\
        uint gid[[thread_position_in_grid]]) {                                          \n\
    dst[gid] = src0[gid] + src1[gid];                                                   \n\
}                                                                                       \n\
";

struct ggml_mtl_context * mnist_mtl_init(
    struct ggml_context * ctx_data,
    struct ggml_context * ctx_eval,
    struct ggml_context * ctx_work,
    struct ggml_cgraph  * gf) {
    fprintf(stderr, "%s: allocating\n", __func__);

    struct ggml_mtl_context * ctx = malloc(sizeof(struct ggml_mtl_context));

    ctx->ctx_data = ctx_data;
    ctx->ctx_eval = ctx_eval;
    ctx->ctx_work = ctx_work;

    ctx->device  = MTLCreateSystemDefaultDevice();
    ctx->queue   = [ctx->device newCommandQueue];

    // compile from source string and show compile log
    NSError * error = nil;
    ctx->library = [ctx->device newLibraryWithSource:msl_library_mnist options:nil error:&error];
    if (error) {
        fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
        exit(1);
    }

    // load kernels
    ctx->function_add = [ctx->library newFunctionWithName:@"kernel_add"];
    ctx->pipeline_add = [ctx->device newComputePipelineStateWithFunction:ctx->function_add error:nil];

    // pin ctx_data memory to GPU
    // use MTLStorageModeShared to allow us to initialize the weights from the CPU
    // TODO: how to use MTLStorageModeManaged?
    // TODO: see if we can avoid this copy somehow
    {
        const void * mem_buffer = ggml_get_mem_buffer(ctx_data);
        const size_t mem_size   = ggml_get_mem_size(ctx_data);

        MTLHeapDescriptor * heap_desc = [MTLHeapDescriptor new];
        heap_desc.storageMode = MTLStorageModeShared;
        heap_desc.size        = mem_size;

        ctx->heap_data = [ctx->device newHeapWithDescriptor:heap_desc];
        [ctx->heap_data setPurgeableState:MTLPurgeableStateNonVolatile];

        id<MTLBuffer> buffer = [ctx->heap_data newBufferWithLength:mem_size options:MTLResourceStorageModeShared];

        // copy data from CPU to GPU
        memcpy([buffer contents], mem_buffer, mem_size);
    }

    // pin ctx_eval memory to GPU
    // this heap will be used for the intermediate results of the evaluation
    {
        const size_t mem_size = ggml_get_mem_size(ctx_eval);

        MTLHeapDescriptor * heap_desc = [MTLHeapDescriptor new];
        heap_desc.storageMode = MTLStorageModePrivate; // GPU only
        heap_desc.size        = mem_size;

        ctx->heap_eval = [ctx->device newHeapWithDescriptor:heap_desc];
        [ctx->heap_eval setPurgeableState:MTLPurgeableStateNonVolatile];
    }

    return ctx;
}

void mnist_mtl_free(struct ggml_mtl_context * ctx) {
    fprintf(stderr, "%s: deallocating\n", __func__);

    free(ctx);
}

// make a view of the respective MTL heap
id<MTLBuffer> mnist_mtl_get_buffer(struct ggml_mtl_context * ctx, struct ggml_tensor * t) {
    const int64_t offs_data = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_data);
    const int64_t offs_eval = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_eval);

    const bool is_data = (offs_eval < 0) || (offs_data >= 0 && offs_data < offs_eval);

    const size_t t_size = ggml_nbytes(t);
    const size_t t_offs = is_data ? offs_data : offs_eval;

    if (is_data) {
        fprintf(stderr, "%s: data tensor '%s'\n", __func__, t->name);
        return [ctx->heap_data newBufferWithLength:t_size options:MTLResourceStorageModeShared offset:t_offs];
    } else {
        fprintf(stderr, "%s: eval tensor '%s'\n", __func__, t->name);
        return [ctx->heap_eval newBufferWithLength:t_size options:MTLResourceStorageModePrivate offset:t_offs];
    }
}

int mnist_mtl_eval(
        struct ggml_mtl_context * ctx,
        struct ggml_cgraph      * gf) {
    fprintf(stderr, "%s: evaluating\n", __func__);

    id<MTLCommandBuffer> command_buffer  = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    for (int i = 0; i < gf->n_nodes; ++i) {
        fprintf(stderr, "%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

        switch (gf->nodes[i]->op) {
            case GGML_OP_ADD:
                {
                    id<MTLBuffer> id_src0 = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src0);
                    id<MTLBuffer> id_src1 = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src1);
                    id<MTLBuffer> id_dst  = mnist_mtl_get_buffer(ctx, gf->nodes[i]);

                    [encoder setComputePipelineState:ctx->pipeline_add];
                    [encoder setBuffer:id_src0 offset:0 atIndex:0];
                    [encoder setBuffer:id_src1 offset:0 atIndex:1];
                    [encoder setBuffer:id_dst  offset:0 atIndex:2];

                    const int64_t n = ggml_nelements(gf->nodes[i]);
                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_RELU:
                {
                } break;
            case GGML_OP_SOFT_MAX:
                {
                } break;
            case GGML_OP_MUL_MAT:
                {
                } break;
            default:
                fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(gf->nodes[i]->op));
                GGML_ASSERT(false);
                return -1;
        }
    }

    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    {
        const double time_elapsed = [command_buffer GPUEndTime] - [command_buffer GPUStartTime];
        fprintf(stderr, "%s: time elapsed = %f\n", __func__, time_elapsed);
    }

    return 0;
}
