// An interface allowing to compute ggml_cgraph with WebGPU
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, OpenCL, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_wgpu_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_wgpu_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_wgpu_set_tensor() and ggml_wgpu_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

// max memory buffers that can be mapped to the device
#define GGML_WGPU_MAX_BUFFERS 16

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// internal API
// temporary exposed to user-code
//

struct ggml_wgpu_context;

void ggml_wgpu_log_set_callback(ggml_log_callback log_callback, void * user_data);

// number of command buffers to use
struct ggml_wgpu_context * ggml_wgpu_init();
void ggml_wgpu_free(struct ggml_wgpu_context * ctx);

void * ggml_wgpu_host_malloc(size_t n);
void   ggml_wgpu_host_free  (void * data);

// creates a mapping between a host memory buffer and a device memory buffer
// - make sure to map all buffers used in the graph before calling ggml_wgpu_graph_compute
// - the mapping is used during computation to determine the arguments of the compute kernels
// - you don't need to keep the host memory buffer allocated as it is never accessed by WebGPU
// - max_size specifies the maximum size of a tensor and is used to create shared views such
//   that it is guaranteed that the tensor will fit in at least one of the views
//
bool ggml_wgpu_add_buffer(
        struct ggml_wgpu_context * ctx,
                       const char * name,
                             void * data,
                           size_t   size,
                           size_t   max_size);

// set data from host memory into the device
void ggml_wgpu_set_tensor(struct ggml_wgpu_context * ctx, struct ggml_tensor * t);

// get data from the device into host memory
void ggml_wgpu_get_tensor(struct ggml_wgpu_context * ctx, struct ggml_tensor * t);

// same as ggml_graph_compute but uses WebGPU
void ggml_wgpu_graph_compute(struct ggml_wgpu_context * ctx, struct ggml_cgraph * gf);

//
// backend API
// user-code should use only these functions
//

GGML_API ggml_backend_t ggml_backend_wgpu_init(void);

GGML_API bool ggml_backend_is_wgpu(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif

