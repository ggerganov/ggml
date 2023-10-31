#import "main-mtl.h"

#import "ggml/ggml.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// TODO: couldn't get this to work
//#define GGML_MTL_HEAP

struct ggml_mtl_context {
    struct ggml_context * ctx_data;
    struct ggml_context * ctx_eval;
    struct ggml_context * ctx_work;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

#ifdef GGML_MTL_HEAP
    id<MTLHeap> heap_data;
    id<MTLHeap> heap_eval;
#else
    id<MTLBuffer> buffer_data;
    id<MTLBuffer> buffer_eval;
#endif

    id<MTLBuffer> out;

    // custom kernels
    id<MTLFunction>             function_add;
    id<MTLComputePipelineState> pipeline_add;

    id<MTLFunction>             function_relu;
    id<MTLComputePipelineState> pipeline_relu;

    id<MTLFunction>             function_soft_max;
    id<MTLComputePipelineState> pipeline_soft_max;
};

// MSL code
NSString * const msl_library_mnist = @"\
#include <metal_stdlib>                                                                 \n\
using namespace metal;                                                                  \n\
                                                                                        \n\
#define MAX(x, y) ((x) > (y) ? (x) : (y))                                               \n\
                                                                                        \n\
constant int k_digits [[function_constant(0)]];                                         \n\
                                                                                        \n\
kernel void kernel_add(                                                                 \n\
        device const float * src0,                                                      \n\
        device const float * src1,                                                      \n\
        device float * dst,                                                             \n\
        uint gid[[thread_position_in_grid]]) {                                          \n\
    dst[gid] = src0[gid] + src1[gid];                                                   \n\
}                                                                                       \n\
                                                                                        \n\
kernel void kernel_relu(                                                                \n\
        device const float * src,                                                       \n\
        device       float * dst,                                                       \n\
        uint gid[[thread_position_in_grid]]) {                                          \n\
    dst[gid] = max(0.0f, src[gid]);                                                     \n\
}                                                                                       \n\
                                                                                        \n\
kernel void kernel_soft_max(                                                            \n\
        device const float * src,                                                       \n\
        device       float * dst,                                                       \n\
        uint gid[[thread_position_in_grid]]) {                                          \n\
    float max = 0.0f;                                                                   \n\
    for (int i = 0; i < k_digits; i++) {                                                \n\
        max = MAX(max, src[i]);                                                         \n\
    }                                                                                   \n\
    float sum = 0.0f;                                                                   \n\
    for (int i = 0; i < k_digits; i++) {                                                \n\
        dst[i] = exp(src[i] - max);                                                     \n\
        sum += dst[i];                                                                  \n\
    }                                                                                   \n\
    for (int i = 0; i < k_digits; i++) {                                                \n\
        dst[i] /= sum;                                                                  \n\
    }                                                                                   \n\
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

    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->queue  = [ctx->device newCommandQueue];

    // determine if we can use MPS
    if (MPSSupportsMTLDevice(ctx->device)) {
        fprintf(stderr, "%s: using MPS\n", __func__);
    } else {
        fprintf(stderr, "%s: not using MPS\n", __func__);
        GGML_ASSERT(false && "MPS not supported");
    }

    // compile from source string and show compile log
    {
        NSError * error = nil;
        ctx->library = [ctx->device newLibraryWithSource:msl_library_mnist options:nil error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }

    // load kernels
    {
        const int k_digits = ggml_graph_get_tensor(gf, "probs")->ne[0];

        MTLFunctionConstantValues * constants = [MTLFunctionConstantValues new];
        [constants setConstantValue:&k_digits type:MTLDataTypeInt withName:@"k_digits"];

        ctx->function_add = [ctx->library newFunctionWithName:@"kernel_add"];
        ctx->pipeline_add = [ctx->device newComputePipelineStateWithFunction:ctx->function_add error:nil];
        fprintf(stderr, "%s: loaded kernel_add: %p\n", __func__, (void *) ctx->pipeline_add);

        ctx->function_relu = [ctx->library newFunctionWithName:@"kernel_relu"];
        ctx->pipeline_relu = [ctx->device newComputePipelineStateWithFunction:ctx->function_relu error:nil];
        fprintf(stderr, "%s: loaded kernel_relu: %p\n", __func__, (void *) ctx->pipeline_relu);

        ctx->function_soft_max = [ctx->library newFunctionWithName:@"kernel_soft_max" constantValues:constants error:nil];
        ctx->pipeline_soft_max = [ctx->device newComputePipelineStateWithFunction:ctx->function_soft_max error:nil];
        fprintf(stderr, "%s: loaded kernel_soft_max: %p\n", __func__, (void *) ctx->pipeline_soft_max);
    }

#ifdef GGML_MTL_HEAP
    // MTLHeap approach

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

        printf("heap_desc.size = %zu\n", mem_size);

        ctx->heap_data = [ctx->device newHeapWithDescriptor:heap_desc];
        [ctx->heap_data setPurgeableState:MTLPurgeableStateNonVolatile]; // TODO: is this needed?
        ctx->heap_data.label = @"heap_data";

        printf("ctx->heap_data.size = %zu\n", [ctx->heap_data size]);

        id<MTLBuffer> buffer = [ctx->heap_data newBufferWithLength:mem_size options:MTLResourceStorageModeShared];
        if (!buffer) {
            fprintf(stderr, "%s: error: failed to allocate buffer\n", __func__);
            exit(1);
        }

        // copy data from CPU to GPU
        memcpy([buffer contents], mem_buffer, mem_size);

        fprintf(stderr, "%s: allocated data heap, size = %zu\n", __func__, mem_size);
    }

    // pin ctx_eval memory to GPU
    // this heap will be used for the intermediate results of the evaluation
    {
        const size_t mem_size = ggml_get_mem_size(ctx_eval);

        MTLHeapDescriptor * heap_desc = [MTLHeapDescriptor new];
        heap_desc.storageMode = MTLStorageModePrivate; // GPU only
        heap_desc.size        = mem_size;

        ctx->heap_eval = [ctx->device newHeapWithDescriptor:heap_desc];
        [ctx->heap_eval setPurgeableState:MTLPurgeableStateNonVolatile]; // TODO: is this needed?

        fprintf(stderr, "%s: allocated eval heap, size = %zu\n", __func__, mem_size);
    }
#else
    // MTLBuffer approach

    // pin ctx_data memory to GPU
    // use MTLStorageModeShared to allow us to initialize the weights from the CPU
    // TODO: how to use MTLStorageModeManaged?
    // TODO: see if we can avoid this copy somehow
    {
        const void * mem_buffer = ggml_get_mem_buffer(ctx_data);
        const size_t mem_size   = ggml_get_mem_size(ctx_data);

        ctx->buffer_data = [ctx->device newBufferWithBytes:mem_buffer length:mem_size options:MTLResourceStorageModeShared];

        fprintf(stderr, "%s: allocated data buffer, size = %zu\n", __func__, mem_size);
    }

    // pin ctx_eval memory to GPU
    // this buffer will be used for the intermediate results of the evaluation
    {
        const size_t mem_size = ggml_get_mem_size(ctx_eval);

        ctx->buffer_eval = [ctx->device newBufferWithLength:mem_size options:MTLResourceStorageModePrivate];

        fprintf(stderr, "%s: allocated eval buffer, size = %zu\n", __func__, mem_size);
    }
#endif

    // allocate buffer for result extraction
    {
        const size_t mem_size = ggml_nbytes(gf->nodes[gf->n_nodes - 1]);

        ctx->out = [ctx->device newBufferWithLength:mem_size options:MTLResourceStorageModeShared];

        fprintf(stderr, "%s: allocated out buffer, size = %zu\n", __func__, mem_size);
    }

    return ctx;
}

void mnist_mtl_free(struct ggml_mtl_context * ctx) {
    fprintf(stderr, "%s: deallocating\n", __func__);

    free(ctx);
}

#ifdef GGML_MTL_HEAP

// make a view of the respective MTL heap
id<MTLBuffer> mnist_mtl_get_buffer_on_heap(struct ggml_mtl_context * ctx, struct ggml_tensor * t) {
    const int64_t offs_data = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_data);
    const int64_t offs_eval = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_eval);

    const bool is_data = (offs_eval < 0) || (offs_data >= 0 && offs_data < offs_eval);

    const size_t t_size = ggml_nbytes(t);
    const size_t t_offs = is_data ? offs_data : offs_eval;

    id<MTLBuffer> result;

    if (is_data) {
        fprintf(stderr, "%s: data tensor '%16s', offs = %8ld, size = %8ld\n", __func__, t->name, t_offs, t_size);
        result = [ctx->heap_data newBufferWithLength:t_size options:MTLResourceStorageModeShared offset:t_offs];
    } else {
        fprintf(stderr, "%s: eval tensor '%16s', offs = %8ld, size = %8ld\n", __func__, t->name, t_offs, t_size);
        result = [ctx->heap_eval newBufferWithLength:t_size options:MTLResourceStorageModePrivate offset:t_offs];
    }

    if (result == nil) {
        fprintf(stderr, "%s: error: buffer is nil\n", __func__);
        GGML_ASSERT(false);
    }

    return result;
}

#else

// get data / eval buffer + offset
id<MTLBuffer> mnist_mtl_get_buffer(struct ggml_mtl_context * ctx, struct ggml_tensor * t, size_t * offs) {
    const int64_t offs_data = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_data);
    const int64_t offs_eval = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_eval);

    const bool is_data = (offs_eval < 0) || (offs_data >= 0 && offs_data < offs_eval);

    const size_t t_size = ggml_nbytes(t);
    const size_t t_offs = is_data ? offs_data : offs_eval;

    id<MTLBuffer> result;

    if (is_data) {
        fprintf(stderr, "%s: data tensor '%16s', offs = %8ld, size = %8ld\n", __func__, t->name, t_offs, t_size);
        result = ctx->buffer_data;
    } else {
        fprintf(stderr, "%s: eval tensor '%16s', offs = %8ld, size = %8ld\n", __func__, t->name, t_offs, t_size);
        result = ctx->buffer_eval;
    }

    if (result == nil) {
        fprintf(stderr, "%s: error: buffer is nil\n", __func__);
        GGML_ASSERT(false);
    }

    if (offs != nil) {
        *offs = t_offs;
    }

    return result;
}

#endif

int mnist_mtl_eval(
        struct ggml_mtl_context * ctx,
        struct ggml_cgraph      * gf) {
    fprintf(stderr, "%s: evaluating\n", __func__);

    id<MTLCommandBuffer> command_buffer  = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = nil;

    size_t offs_src0;
    size_t offs_src1;
    size_t offs_dst;

    // copy the input data to the GPU
    {
        struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "input");

        id<MTLBuffer> id_dst = mnist_mtl_get_buffer(ctx, inp, &offs_src0);

        memcpy((char *) id_dst.contents + offs_src0, inp->data, ggml_nbytes(inp));
    }

    for (int i = 0; i < gf->n_nodes; ++i) {
        fprintf(stderr, "%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

        switch (gf->nodes[i]->op) {
            case GGML_OP_ADD:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src0 = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src[0], &offs_src0);
                    id<MTLBuffer> id_src1 = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src[1], &offs_src1);
                    id<MTLBuffer> id_dst  = mnist_mtl_get_buffer(ctx, gf->nodes[i],         &offs_dst);

                    [encoder setComputePipelineState:ctx->pipeline_add];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                    const int64_t n = ggml_nelements(gf->nodes[i]);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(gf->nodes[i])) {
                    case GGML_UNARY_OP_RELU:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            id<MTLBuffer> id_src = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src[0], &offs_src0);
                            id<MTLBuffer> id_dst = mnist_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                            [encoder setComputePipelineState:ctx->pipeline_relu];
                            [encoder setBuffer:id_src offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst offset:offs_dst  atIndex:1];

                            const int64_t n = ggml_nelements(gf->nodes[i]);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    default:
                        {
                            fprintf(stderr, "%s: node %3d, op = %8s, unary op %d not implemented\n", __func__, i, ggml_op_name(gf->nodes[i]->op), (int) ggml_get_unary_op(gf->nodes[i]));
                            GGML_ASSERT(false);
                            return -1;
                        }
                        break;
                } break;
            case GGML_OP_SOFT_MAX:
                {
#if 0
                    // NOTE: MPSMatrixSoftMax is not working properly, probably there is a bug

                    if (encoder != nil) {
                        [encoder endEncoding];
                        encoder = nil;
                    }

                    // use MPSMatrixSoftMax
                    id<MTLBuffer> id_src = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_dst = mnist_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    MPSMatrixDescriptor * desc = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:1 columns:gf->nodes[i]->ne[0] rowBytes:gf->nodes[i]->nb[1] dataType:MPSDataTypeFloat32];

                    MPSMatrix * mat_src = [[MPSMatrix alloc] initWithBuffer:id_src offset:offs_src0 descriptor:desc];
                    MPSMatrix * mat_dst = [[MPSMatrix alloc] initWithBuffer:id_dst offset:offs_dst  descriptor:desc];

                    MPSMatrixSoftMax * softmax = [[MPSMatrixSoftMax alloc] initWithDevice:ctx->device];

                    [softmax encodeToCommandBuffer:command_buffer inputMatrix:mat_src resultMatrix:mat_dst];
#else
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src[0], &offs_src0);
                    id<MTLBuffer> id_dst = mnist_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    [encoder setComputePipelineState:ctx->pipeline_soft_max];
                    [encoder setBuffer:id_src offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
#endif
                } break;
            case GGML_OP_MUL_MAT:
                {
                    if (encoder != nil) {
                        [encoder endEncoding];
                        encoder = nil;
                    }

                    // use MPSMatrixMultiplication
                    id<MTLBuffer> id_src0 = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src[0], &offs_src0);
                    id<MTLBuffer> id_src1 = mnist_mtl_get_buffer(ctx, gf->nodes[i]->src[1], &offs_src1);
                    id<MTLBuffer> id_dst  = mnist_mtl_get_buffer(ctx, gf->nodes[i],         &offs_dst);

                    const int64_t ncols0 = gf->nodes[i]->src[0]->ne[0];
                    const int64_t nrows0 = gf->nodes[i]->src[0]->ne[1];

                    const int64_t ncols1 = gf->nodes[i]->src[1]->ne[0];
                    const int64_t nrows1 = gf->nodes[i]->src[1]->ne[1];

                    const int64_t ncols2 = gf->nodes[i]->ne[0];
                    const int64_t nrows2 = gf->nodes[i]->ne[1];

                    GGML_ASSERT(ncols0 == ncols1);

                    MPSMatrixDescriptor * desc0 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:nrows0 columns:ncols0 rowBytes:gf->nodes[i]->src[0]->nb[1] dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor * desc1 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:nrows1 columns:ncols1 rowBytes:gf->nodes[i]->src[1]->nb[1] dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor * desc2 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:nrows2 columns:ncols2 rowBytes:gf->nodes[i]->nb[1] dataType:MPSDataTypeFloat32];

                    MPSMatrix * mat_src0 = [[MPSMatrix alloc] initWithBuffer:id_src0 offset:offs_src0 descriptor:desc0];
                    MPSMatrix * mat_src1 = [[MPSMatrix alloc] initWithBuffer:id_src1 offset:offs_src1 descriptor:desc1];
                    MPSMatrix * mat_dst  = [[MPSMatrix alloc] initWithBuffer:id_dst  offset:offs_dst  descriptor:desc2];

                    MPSMatrixMultiplication * mul = [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                        transposeLeft:false transposeRight:true resultRows:nrows1 resultColumns:nrows0 interiorColumns:ncols0 alpha:1.0 beta:0.0];

                    [mul encodeToCommandBuffer:command_buffer leftMatrix:mat_src1 rightMatrix:mat_src0 resultMatrix:mat_dst];
                } break;
            default:
                {
                    fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(gf->nodes[i]->op));
                    GGML_ASSERT(false);
                    return -1;
                }
        }
    }

    // extract results from the GPU
    {
        if (encoder != nil) {
            [encoder endEncoding];
            encoder = nil;
        }

        struct ggml_tensor * out = gf->nodes[gf->n_nodes - 1];

        id<MTLBuffer> id_src = mnist_mtl_get_buffer(ctx, out, &offs_src0);
        id<MTLBuffer> id_dst = ctx->out;

        id<MTLBlitCommandEncoder> encoder_blit = [command_buffer blitCommandEncoder];
        [encoder_blit copyFromBuffer:id_src sourceOffset:offs_src0 toBuffer:id_dst destinationOffset:0 size:ggml_nbytes(out)];
        [encoder_blit endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    {
        const double time_elapsed = [command_buffer GPUEndTime] - [command_buffer GPUStartTime];
        fprintf(stderr, "%s: time elapsed = %f\n", __func__, time_elapsed);
    }

    // select the most probable digit
    int result = -1;
    {
        const float * probs = ctx->out.contents;

        float prob = probs[0];

        for (int i = 0; i < 10; ++i) {
            fprintf(stderr, "%s: probs[%2d] = %f\n", __func__, i, probs[i]);

            if (probs[i] > prob) {
                result = i;
                prob = probs[i];
            }
        }
    }

    return result;
}
