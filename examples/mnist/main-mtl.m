#import "main-mtl.h"

#import "ggml/ggml.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

struct ggml_mtl_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command_buffer;

    id<MTLHeap> heap_data;
    id<MTLHeap> heap_eval;
};

struct ggml_mtl_context * mnist_mtl_init(
    struct ggml_context * ctx_data,
    struct ggml_context * ctx_eval,
    struct ggml_context * ctx_work,
    struct ggml_cgraph  * gf) {
    fprintf(stderr, "%s: allocating\n", __func__);

    struct ggml_mtl_context * ctx = malloc(sizeof(struct ggml_mtl_context));

    ctx->device         = MTLCreateSystemDefaultDevice();
    ctx->queue          = [ctx->device newCommandQueue];
    ctx->command_buffer = [ctx->queue commandBuffer];

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

int mnist_mtl_eval(struct ggml_mtl_context * ctx, struct ggml_cgraph * gf) {
    fprintf(stderr, "%s: evaluating\n", __func__);

    // create a new encoder for the command buffer
    id<MTLComputeCommandEncoder> encoder = [ctx->command_buffer computeCommandEncoder];

    for (int i = 0; i < gf->n_nodes; ++i) {
        fprintf(stderr, "%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

        // TODO ...
    }

    // finish encoding
    [encoder endEncoding];

    [ctx->command_buffer commit];
    [ctx->command_buffer waitUntilCompleted];

    {
        const double time_elapsed = [ctx->command_buffer GPUEndTime] - [ctx->command_buffer GPUStartTime];
        fprintf(stderr, "%s: time elapsed = %f\n", __func__, time_elapsed);
    }

    return 0;
}
