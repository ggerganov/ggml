#import "main-mtl.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

int mnist_mtl_eval(
    struct ggml_context * ctx_data,
    struct ggml_context * ctx_eval,
    struct ggml_context * ctx_work,
    struct ggml_cgraph  * gf) {
    printf("mnist_mtl_eval\n");
    return 0;
}
